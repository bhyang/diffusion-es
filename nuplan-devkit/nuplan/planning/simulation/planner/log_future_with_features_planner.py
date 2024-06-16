import itertools
import logging
from typing import List, Optional, Type

from omegaconf.dictconfig import DictConfig

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import (
    GenericAgentsFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

logger = logging.getLogger(__name__)


class LogFutureWithFeaturesPlanner(AbstractPlanner):
    """
    Planner which just looks as future GT and returns it as a desired trajectory
    the input to this planner are detections.
    """

    # Inherited property, see superclass.
    requires_scenario: bool = True

    def __init__(self, scenario: AbstractScenario, num_poses: int, future_time_horizon: float, 
                 model: TorchModuleWrapper):
        """
        Constructor of LogFuturePlanner.
        :param scenario: The scenario the planner is running on.
        :param num_poses: The number of poses to plan for.
        :param future_time_horizon: [s] The horizon length to plan for.
        """
        self._scenario = scenario

        self._num_poses = num_poses
        self._future_time_horizon = future_time_horizon
        self._trajectory: Optional[AbstractTrajectory] = None
        
        self._model = model
        self._feature_builders = model.get_list_of_required_feature()
        self._observations = []

        self._counter = 0

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        self._initialization = initialization

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        current_state = self._scenario.get_ego_state_at_iteration(current_input.iteration.index)
        try:
            states = self._scenario.get_ego_future_trajectory(
                current_input.iteration.index, self._future_time_horizon, self._num_poses
            )
            self._trajectory = InterpolatedTrajectory(list(itertools.chain([current_state], states)))
        except AssertionError:
            logger.warning("Cannot retrieve future ego trajectory. Using previous computed trajectory.")
            if self._trajectory is None:
                raise RuntimeError("Future ego trajectory cannot be retrieved from the scenario!")

        # Store observation
        observation = self.get_observation(current_input)
        self._observations.append(observation)

        # print(self._counter)
        # self._counter += 1

        return self._trajectory

    def get_observation(self, current_input):
        features = {
            builder.get_feature_unique_name(): builder.get_features_from_simulation(current_input, self._initialization)
            for builder in self._feature_builders
        }
        features = {name: feature.to_feature_tensor() for name, feature in features.items()}
        features = {name: feature.collate([feature]) for name, feature in features.items()}
        features = {name: feature.to_device('cpu') for name, feature in features.items()}
        return features

    def get_observations(self):
        return self._observations
