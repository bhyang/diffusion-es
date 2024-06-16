import time
from typing import List, Optional, Type, cast
import os
import itertools
import logging

import numpy as np
import numpy.typing as npt
from PIL import Image
from omegaconf import OmegaConf

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.callbacks.utils.visualization_utils import get_generic_raster_from_vector_map
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper

logger = logging.getLogger(__name__)


class MLPlanner(AbstractPlanner):
    """
    Implements abstract planner interface.
    Used for simulating any ML planner trained through the nuPlan training framework.
    """

    requires_scenario: bool = True

    def __init__(
        self, 
        # model: TorchModuleWrapper, 
        checkpoint_path: str,
        scenario: AbstractScenario, 
        replan_freq: int = 1,
        dump_gifs: bool = False,
        dump_gifs_path: str = None,
        log_replay: bool = False
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._checkpoint_path = checkpoint_path

        self._future_horizon = 8.0 # model.future_trajectory_sampling.time_horizon
        self._step_interval = 0.5 # model.future_trajectory_sampling.step_time
        self._num_output_dim = 16 # model.future_trajectory_sampling.num_poses

        # self._model_loader = ModelLoader(model)

        self._initialization: Optional[PlannerInitialization] = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self._observations = []

        self.states = None
        self.replan_freq = replan_freq
        self.replan_counter = 0

        self.dump_gifs = dump_gifs
        self.dump_gifs_path = dump_gifs_path
        self._counter = 0
        self._frames = []

        self._scenario = scenario
        self._log_replay = log_replay

    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        # Propagate model
        predictions = self._model_loader.infer(features)

        # Extract trajectory prediction
        trajectory_predicted = cast(Trajectory, predictions['trajectory'])
        trajectory_tensor = trajectory_predicted.data
        trajectory = trajectory_tensor.cpu().detach().numpy()[0]  # retrive first (and only) batch as a numpy array

        predictions['trajectory'] = cast(npt.NDArray[np.float32], trajectory)

        return predictions

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        config_path = '/'.join(self._checkpoint_path.split('/')[:-2])
        config_path = os.path.join(config_path, 'code/hydra/config.yaml')
        model_config = OmegaConf.load(config_path).model
        torch_module_wrapper = build_torch_module_wrapper(model_config)
        model = LightningModuleWrapper.load_from_checkpoint(
            self._checkpoint_path, model=torch_module_wrapper
        ).model

        self._model_loader = ModelLoader(model)
        self._model_loader.initialize()
        self._initialization = initialization

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        # Extract history
        history = current_input.history

        # Construct input features
        start_time = time.perf_counter()
        features = self._model_loader.build_features(current_input, self._initialization)
        self._feature_building_runtimes.append(time.perf_counter() - start_time)

        # Infer model
        start_time = time.perf_counter()

        if self.replan_counter % self.replan_freq == 0:
            if not self._log_replay:
                out = self._infer_model(features)
                predictions = out['trajectory']

                if 'attentions' in out:
                    attentions = out['attentions'][0,0].cpu().numpy()
                    # Just vehicles for now
                    attentions = attentions[1:31]
                    attentions = attentions / attentions.max()

                    # import matplotlib.pyplot as plt
                    # plt.hist(np.arange(attentions.shape[0]), weights=attentions, bins=range(attentions.shape[0]))
                    # plt.savefig('/zfsauton2/home/brianyan/nuplan-devkit/test.png')
                    # import pdb; pdb.set_trace()
                    # raise
                
                # Convert relative poses to absolute states and wrap in a trajectory object.
                self.states = transform_predictions_to_states(
                    predictions, history.ego_states, self._future_horizon, self._step_interval
                )
            else:
                current_state = self._scenario.get_ego_state_at_iteration(current_input.iteration.index)
                try:
                    states = self._scenario.get_ego_future_trajectory(
                        current_input.iteration.index, 8.0, 16
                    )
                    self.states = list(itertools.chain([current_state], states))
                except AssertionError:
                    logger.warning("Cannot retrieve future ego trajectory. Using previous computed trajectory.")
                    if self._trajectory is None:
                        raise RuntimeError("Future ego trajectory cannot be retrieved from the scenario!")

            if self.dump_gifs:
                frame = get_generic_raster_from_vector_map(
                    features['vector_set_map'].to_device('cpu'),
                    features['generic_agents'].to_device('cpu'),
                    trajectory=predictions if not self._log_replay else None,
                    agent_weights=attentions if not self._log_replay and 'attentions' in out else None,
                    pixel_size=0.1,
                    radius=60
                )
                self._counter += 1
                # cv2.imwrite(path, frame[...,::-1])
                self._frames.append(frame)
                if self._counter == 149:
                    # Save to gif
                    frames = [Image.fromarray(frame) for frame in self._frames]
                    # fname_dir = '/zfsauton2/home/brianyan/nuplan-devkit/nuplan/planning/simulation/planner/ml_planner/viz/unimodal/'
                    fname_dir = f'{self.dump_gifs_path}'
                    if not os.path.exists(fname_dir):
                        os.makedirs(fname_dir)
                    fname = f'{fname_dir}{time.strftime("%Y%m%d-%H%M%S")}.gif'
                    # fname = f'{fname_dir}viz.gif'
                    frames[0].save(fname, save_all=True, append_images=frames[1:], duration=int(self._counter * .4), loop=0)
                    # self._frames = []
                    # raise
                    print(f'SAVING GIF TO {fname}')
            
                print(self._counter)

        self.replan_counter += 1

        self._inference_runtimes.append(time.perf_counter() - start_time)

        self._observations.append({key: features[key].to_device('cpu') for key in features})

        trajectory = InterpolatedTrajectory(self.states)

        return trajectory

    def get_observations(self):
        return self._observations

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []
        return report
