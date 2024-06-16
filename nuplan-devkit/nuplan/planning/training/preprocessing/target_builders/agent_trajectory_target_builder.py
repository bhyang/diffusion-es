from __future__ import annotations

from typing import Dict, List, Tuple, Type, cast

import torch
import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.agents_trajectory import AgentTrajectory
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features_from_tensor,
    compute_yaw_rate_from_state_tensors,
    convert_absolute_quantities_to_relative,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list,
    AgentInternalIndex,
    extract_track_token_ids,
    build_trajectory_features
)


class AgentTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, 
        trajectory_sampling: TrajectorySampling, 
        future_trajectory_sampling: TrajectorySampling,
        max_agents: int
    ) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__()
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon

        self.max_agents = max_agents

        self.num_future_poses = future_trajectory_sampling.num_poses
        self.future_time_horizon = future_trajectory_sampling.time_horizon

        self._agents_states_dim = Agents.agents_states_dim()

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "agent_trajectories"

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return AgentTrajectory  # type: ignore

    @torch.jit.unused
    def get_targets(self, scenario: AbstractScenario) -> Agents:
        """Inherited, see superclass."""
        # Retrieve present/past ego states and agent boxes
        with torch.no_grad():
            anchor_ego_state = scenario.initial_ego_state

            past_ego_states = scenario.get_ego_past_trajectory(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
            sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
            time_stamps = list(scenario.get_past_timestamps(
                    iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
                )
            ) + [scenario.start_time]

            # Retrieve present/future agent boxes
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
                )
            ]
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]

            sampled_past_observations = past_tracked_objects + [present_tracked_objects]

            present_agents = present_tracked_objects.get_tracked_objects_of_type(TrackedObjectType['VEHICLE'])
            if len(present_agents) > self.max_agents:
                # keep the closest agents to ego
                dists = [anchor_ego_state.rear_axle.distance_to(agent.center) for agent in present_agents]
                present_agents = [present_agents[i] for i in np.argsort(dists)[:self.max_agents]]

            track_token_ids = extract_track_token_ids(present_agents)
            trajectory_data, trajectory_masks = build_trajectory_features(future_tracked_objects, track_token_ids, anchor_ego_state)

            if len(present_agents) < self.max_agents:
                # pad to self.max_agents
                pad_length = self.max_agents - len(present_agents)
                trajectory_data = torch.cat([
                    trajectory_data,
                    torch.zeros(self.num_future_poses,pad_length,trajectory_data.shape[2])
                ], dim=1)
                trajectory_masks = torch.cat([
                    trajectory_masks,
                    torch.zeros(self.num_future_poses, pad_length, dtype=torch.bool)
                ], dim=1)

            trajectory_data = trajectory_data.numpy()
            trajectory_masks = trajectory_masks.numpy()

            return AgentTrajectory(data=trajectory_data, mask=trajectory_masks)
