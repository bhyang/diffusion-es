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
from nuplan.planning.training.callbacks.utils.visualization_utils import visualize
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper

from nuplan.planning.simulation.planner.ml_planner.constraints import (
    make_go_fast_constraint,
    make_stop_constraint,
    make_goal_constraint,
    make_lane_following_constraint,
    make_drivable_area_compliance_constraint,
    make_speed_constraint
)

from nuplan.planning.simulation.planner.ml_planner.helpers import (
    convert_to_local,
    convert_to_global,
    _get_lanes,
    _get_current_lane,
    _get_next_lane,
    _get_next_of,
    _get_left_lane,
    _get_right_lane,
    _get_points_of_lane
)

logger = logging.getLogger(__name__)


class FactorizedDiffusionPlanner(AbstractPlanner):

    requires_scenario: bool = True

    def __init__(
        self, 
        checkpoint_path: str,
        scenario: AbstractScenario, 
        replan_freq: int = 1,
        dump_gifs: bool = False,
        dump_gifs_path: str = None
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._checkpoint_path = checkpoint_path

        self._future_horizon = 8.0
        self._step_interval = 0.5

        self._initialization: Optional[PlannerInitialization] = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self.predictions_global = None
        self.replan_freq = replan_freq
        self.replan_counter = 0

        self.dump_gifs = dump_gifs
        self.dump_gifs_path = dump_gifs_path
        self._counter = 0
        self._counter_max = 30 # (150 // self.replan_counter) - 1
        self._frames = []

        self._plan_history = []

        self._scenario = scenario

        self._prev_ego_trajectory = None
    
    def update_goal(self):
        if self.goal is None:
            return

        if self.goal_type == 'static':
            # Convert global to local
            global_goal, threshold = self.goal
            self.local_goal = convert_to_local(self, global_goal)
            curr_pos = self.features['agent_history'].ego[0].cpu().numpy()[-1][:2]
            self.goal_reached = np.linalg.norm(self.local_goal[:, :2] - curr_pos) < threshold

        elif self.goal_type == 'dynamic':
            # Let's just say the closest to the dynamic target is the same one
            # TODO: Do something less stupid for dynamic tracking
            global_anchor, offset, threshold = self.goal
            local_anchor = convert_to_local(self, global_anchor)
            agent_features = self.features['generic_agents'].agents['VEHICLE'][0][-1].cpu().numpy()
            dists = np.linalg.norm(local_anchor[None] - agent_features[:,:2], axis=-1)
            new_local_anchor = agent_features[np.argmin(dists)][:2]
            # Check if reached
            self.local_goal = new_local_anchor + offset
            curr_pos = self.features['generic_agents'].ego[0].cpu().numpy()[-1][:2]
            self.goal_reached = np.linalg.norm(self.local_goal - curr_pos) < threshold
            # Update global anchor poss
            new_global_anchor = convert_to_global(self, new_local_anchor)
            self.goal = (new_global_anchor, offset, threshold)

        else:
            raise NotImplementedError

    def _infer_model(self, features: FeaturesType, disable_grad_guidance: bool = False) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """

        # Build constraints
        features['constraints'] = self.generate_constraints(features)
        features['disable_grad_guidance'] = disable_grad_guidance

        if self._prev_ego_trajectory is not None:
            features['warm_start'] = self._prev_ego_trajectory
        #     features['warm_start_steps'] = 90

        # Propagate model
        predictions = self._model_loader._model.cem_test(features)
        # predictions = self._model_loader.infer(features)
        # visualize_denoising(features, predictions['intermediate'])

        # Extract trajectory prediction
        trajectory_predicted = cast(Trajectory, predictions['trajectory'])
        trajectory_tensor = trajectory_predicted.data
        trajectory = trajectory_tensor.cpu().detach().numpy()[0]  # retrieve first (and only) batch as a numpy array

        predictions['trajectory'] = cast(npt.NDArray[np.float32], trajectory)

        self._prev_ego_trajectory = predictions['multimodal_trajectories']

        return predictions

    def generate_constraints(self, features):
        """
        Each constraint is a function that maps an ego-trajectory to some scalar cost to be minimized.

        Since we use the constraint to guide diffusion sampling, we need to compute this constraint cost 
        at every step of the diffusion sampling process, so we provide the model with callable functions
        which will be called iteratively in model inference.

        Even though the constraint functions the model sees only take the ego-trajectory as input, they can
        be conditioned on arbitrary information since we use closures.
        """
        self.update_goal()

        return [
            # Test constraint
            # make_go_fast_constraint(weight=10.0),
            # make_stop_constraint(),

            make_speed_constraint(weight=100.0),

            # TODO: Goal reaching
            make_goal_constraint(features, self.local_goal, weight=1.0),

            # TODO: Lane following
            # make_lane_following_constraint(
            #     features, _get_current_lane(self), _get_next_lane(self),
            #     weight=10.0
            # ),

            # TODO: drivable area compliance
            # make_drivable_area_compliance_constraint(features, self._initialization.map_api, )
        ]
            
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        config_path = '/'.join(self._checkpoint_path.split('/')[:-2])
        config_path = os.path.join(config_path, 'code/hydra/config.yaml')
        model_config = OmegaConf.load(config_path).model
        torch_module_wrapper = build_torch_module_wrapper(model_config)
        model = LightningModuleWrapper.load_from_checkpoint(
            self._checkpoint_path, model=torch_module_wrapper
        ).model

        model.predictions_per_sample = 64

        self._model_loader = ModelLoader(model)
        self._model_loader.initialize()
        self._initialization = initialization
        self.goal = None

    def set_static_goal(self, pos, threshold=5.0):
        self.goal_type = 'static'
        self.goal = (convert_to_global(self, pos), threshold)
        self.update_goal()

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
        self.current_input = current_input
        # Extract history
        history = current_input.history

        # Construct input features
        start_time = time.perf_counter()
        features = self._model_loader.build_features(current_input, self._initialization)
        self._feature_building_runtimes.append(time.perf_counter() - start_time)
        self.features = features
        if self.goal is None:
            goal = []
            current = _get_lanes(self, 24)
            for _ in range(6):
                next = _get_next_of(self, current)
                goal.append(_get_points_of_lane(self, current))
                current = next
            
            goal = np.concatenate(goal)
            self.set_static_goal(goal)
            # self.set_static_goal(np.array([[18.0, -20.0]]))

        # Infer model
        start_time = time.perf_counter()

        if self.replan_counter % self.replan_freq == 0:
            out = self._infer_model(features, disable_grad_guidance=False)

            predictions = out['trajectory']

            multimodal_pred = out['multimodal_trajectories'].detach().cpu().numpy()
            multimodal_pred = multimodal_pred.reshape(-1,16,3)

            # gradient = out['grad']

            # Convert relative poses to absolute states and wrap in a trajectory object
            self.states = transform_predictions_to_states(
                predictions, history.ego_states, self._future_horizon, self._step_interval
            )
            self.multimodal_pred = convert_to_global(self, multimodal_pred.reshape(-1,3))

            if self.dump_gifs:
                states_array = [(state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading) for state in self.states]
                states_array = np.array(states_array)
                states_array = convert_to_local(self, states_array)

                mm_states_array = convert_to_local(self, self.multimodal_pred)
                mm_states_array = mm_states_array.reshape(-1,16,3)

                frame = visualize(
                    features['vector_set_map'].to_device('cpu'),
                    features['agent_history'].to_device('cpu'),

                    allpred=mm_states_array,
                    allpred_color='blue',

                    bestpred=states_array,
                    bestpred_color='green',
                    # bestpred_grad=gradient,

                    goal=self.local_goal,
                    goal_color='orange',

                    pixel_size=0.1,
                    radius=60,
                )

                self._counter += 1
                self._frames.append(frame)

                if self._counter == self._counter_max:
                    # Save to gif
                    frames = [Image.fromarray(frame) for frame in self._frames]
                    fname_dir = f'{self.dump_gifs_path}'
                    if not os.path.exists(fname_dir):
                        os.makedirs(fname_dir)
                    fname = f'{fname_dir}/{time.strftime("%Y%m%d-%H%M%S")}.gif'
                    frames[0].save(fname, save_all=True, append_images=frames[1:], duration=int(self._counter * .4), loop=0)
                    print(f'SAVING GIF TO {fname}')
            
                print(self._counter)

        self.replan_counter += 1

        self._inference_runtimes.append(time.perf_counter() - start_time)

        trajectory = InterpolatedTrajectory(self.states)

        return trajectory

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


def visualize_denoising(features, trajs):
    frames = []
    for intermediate_trajs in trajs:
        traj = intermediate_trajs.squeeze(0).reshape(-1,16,3).cpu().numpy()

        frame = visualize(
            features['vector_set_map'].to_device('cpu'),
            features['agent_history'].to_device('cpu'),

            traj=traj,
            traj_color='blue',

            pixel_size=0.1,
            radius=60,
        )

        frames.append(frame)

    frames = [Image.fromarray(frame) for frame in frames]
    fname_dir = '/zfsauton2/home/brianyan/nuplan-devkit/nuplan/planning/simulation/planner/ml_planner/viz'
    fname = f'{fname_dir}/denoise.gif'
    frames[0].save(fname, save_all=True, append_images=frames[1:], duration=int(32 * 0.25), loop=0)

    print(f'Visualization dumped to {fname}')

    raise