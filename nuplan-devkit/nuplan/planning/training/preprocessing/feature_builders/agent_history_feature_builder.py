from typing import Dict, List, Tuple, Type, cast

import torch
import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature, AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.agent_history import AgentHistory
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_generic_ego_features_from_tensor,
    compute_yaw_rate_from_state_tensors,
    convert_absolute_quantities_to_relative,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list,
    extract_track_token_ids,
    build_history_features
)


class AgentHistoryFeatureBuilder(AbstractFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, trajectory_sampling: TrajectorySampling, max_agents: int) -> None:
        """
        Initializes AgentHistoryFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__()
        self._agent_features = ['VEHICLE']
        self._num_past_poses = trajectory_sampling.num_poses
        self._past_time_horizon = trajectory_sampling.time_horizon
        self._max_agents = max_agents

        self._agents_states_dim = GenericAgents.agents_states_dim()

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "agent_history"

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return AgentHistory  # type: ignore

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> AgentHistory:
        """Inherited, see superclass."""
        # Retrieve present/past ego states and agent boxes
        with torch.no_grad():
            anchor_ego_state = scenario.initial_ego_state

            # Ego features
            past_ego_states = scenario.get_ego_past_trajectory(
                iteration=0, num_samples=self._num_past_poses, time_horizon=self._past_time_horizon
            )
            sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
            past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)
            ego_tensor = build_generic_ego_features_from_tensor(past_ego_states_tensor, reverse=True)

            # Agent features
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self._past_time_horizon, num_samples=self._num_past_poses
                )
            ]
            sampled_past_observations = past_tracked_objects + [present_tracked_objects]

            present_agents = present_tracked_objects.get_tracked_objects_of_type(TrackedObjectType['VEHICLE'])
            if len(present_agents) > self._max_agents:
                # keep the closest agents to ego
                dists = [anchor_ego_state.rear_axle.distance_to(agent.center) for agent in present_agents]
                present_agents = [present_agents[i] for i in np.argsort(dists)[:self._max_agents]]

            track_token_ids = extract_track_token_ids(present_agents)
            history_data, history_masks = build_history_features(sampled_past_observations, track_token_ids, anchor_ego_state)

            if len(present_agents) < self._max_agents:
                pad_length = self._max_agents - len(present_agents)
                history_data = torch.cat([
                    history_data,
                    torch.zeros(self._num_past_poses+1, pad_length, history_data.shape[2], dtype=torch.bool)
                ], dim=1)
                history_masks = torch.cat([
                    history_masks,
                    torch.zeros(self._num_past_poses+1, pad_length, dtype=torch.bool)
                ], dim=1)

            ego_tensor = ego_tensor.numpy()
            history_data = history_data.numpy()
            history_masks = history_masks.numpy()

            return AgentHistory(ego=ego_tensor, data=history_data, mask=history_masks)


    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AgentHistory:
        """Inherited, see superclass."""
        with torch.no_grad():
            history = current_input.history
            assert isinstance(
                history.observations[0], DetectionsTracks
            ), f"Expected observation of type DetectionTracks, got {type(history.observations[0])}"

            anchor_ego_state, present_observation = history.current_state

            past_observations = history.observations[:-1]
            past_ego_states = history.ego_states[:-1]

            indices = sample_indices_with_time_horizon(
                self._num_past_poses, self._past_time_horizon, history.sample_interval
            )

            # Ego features
            sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
            sampled_past_ego_states = sampled_past_ego_states + [anchor_ego_state]
            # sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
            past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)
            ego_tensor = build_generic_ego_features_from_tensor(past_ego_states_tensor, reverse=True)

            # Agent features
            sampled_past_observations = [
                cast(DetectionsTracks, past_observations[-idx]).tracked_objects for idx in reversed(indices)
            ]
            sampled_present_observations = cast(DetectionsTracks, present_observation).tracked_objects
            sampled_past_observations = sampled_past_observations + [sampled_present_observations]
            present_agents = sampled_present_observations.get_tracked_objects_of_type(TrackedObjectType['VEHICLE'])
            if len(present_agents) > self._max_agents:
                # keep the closest agents to ego
                dists = [anchor_ego_state.rear_axle.distance_to(agent.center) for agent in present_agents]
                present_agents = [present_agents[i] for i in np.argsort(dists)[:self._max_agents]]

            track_token_ids = extract_track_token_ids(present_agents)
            history_data, history_masks = build_history_features(sampled_past_observations, track_token_ids, anchor_ego_state)

            if len(present_agents) < self._max_agents:
                pad_length = self._max_agents - len(present_agents)
                history_data = torch.cat([
                    history_data,
                    torch.zeros(self._num_past_poses+1, pad_length, history_data.shape[2], dtype=torch.bool)
                ], dim=1)
                history_masks = torch.cat([
                    history_masks,
                    torch.zeros(self._num_past_poses+1, pad_length, dtype=torch.bool)
                ], dim=1)

            ego_tensor = ego_tensor.numpy()
            history_data = history_data.numpy()
            history_masks = history_masks.numpy()

            return AgentHistory(ego=ego_tensor, data=history_data, mask=history_masks)

    # @torch.jit.unused
    # def _pack_to_feature_tensor_dict(
    #     self,
    #     past_ego_states: List[EgoState],
    #     past_time_stamps: List[TimePoint],
    #     past_tracked_objects: List[TrackedObjects],
    # ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
    #     """
    #     Packs the provided objects into tensors to be used with the scriptable core of the builder.
    #     :param past_ego_states: The past states of the ego vehicle.
    #     :param past_time_stamps: The past time stamps of the input data.
    #     :param past_tracked_objects: The past tracked objects.
    #     :return: The packed tensors.
    #     """
    #     list_tensor_data: Dict[str, List[torch.Tensor]] = {}
    #     past_ego_states_tensor = sampled_past_ego_states_to_tensor(past_ego_states)
    #     past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

    #     for feature_name in self._agent_features:
    #         past_tracked_objects_tensor_list = sampled_tracked_objects_to_tensor_list(
    #             past_tracked_objects, TrackedObjectType[feature_name]
    #         )
    #         list_tensor_data[f"past_tracked_objects.{feature_name}"] = past_tracked_objects_tensor_list

    #     return (
    #         {
    #             "past_ego_states": past_ego_states_tensor,
    #             "past_time_stamps": past_time_stamps_tensor,
    #         },
    #         list_tensor_data,
    #         {},
    #     )

    # @torch.jit.unused
    # def _unpack_feature_from_tensor_dict(
    #     self,
    #     tensor_data: Dict[str, torch.Tensor],
    #     list_tensor_data: Dict[str, List[torch.Tensor]],
    #     list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    # ) -> AgentHistory:
    #     """
    #     Unpacks the data returned from the scriptable core into an AgentHistory feature class.
    #     :param tensor_data: The tensor data output from the scriptable core.
    #     :param list_tensor_data: The List[tensor] data output from the scriptable core.
    #     :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
    #     :return: The packed AgentHistory object.
    #     """
    #     ego_features = list_tensor_data["generic_agents.ego"][0].detach().numpy()
    #     agent_features = list_tensor_data['generic_agents.agents.VEHICLE'][0].detach().numpy()
    #     num_agents_present = agent_features.shape[1]
    #     if num_agents_present > self._max_agents:
    #         agent_features = agent_features[:,:self._max_agents]
    #     else:
    #         agent_features = np.concatenate([
    #             agent_features, 
    #             np.zeros((self._num_past_poses+1, self._max_agents - num_agents_present, agent_features.shape[2]))
    #         ], axis=1)
    #     agent_masks = np.zeros((self._num_past_poses+1, self._max_agents), dtype=bool)
    #     agent_masks[:,:num_agents_present] = True

    #     out = AgentHistory(
    #         ego=ego_features[None],
    #         data=agent_features[None],
    #         mask=agent_masks[None]
    #     )
    #     return out

    # @torch.jit.export
    # def scriptable_forward(
    #     self,
    #     tensor_data: Dict[str, torch.Tensor],
    #     list_tensor_data: Dict[str, List[torch.Tensor]],
    #     list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    # ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
    #     """
    #     Inherited. See interface.
    #     """
    #     output_dict: Dict[str, torch.Tensor] = {}
    #     output_list_dict: Dict[str, List[torch.Tensor]] = {}
    #     output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}

    #     ego_history: torch.Tensor = tensor_data["past_ego_states"]
    #     time_stamps: torch.Tensor = tensor_data["past_time_stamps"]
    #     anchor_ego_state = ego_history[-1, :].squeeze()

    #     # ego features
    #     ego_tensor = build_generic_ego_features_from_tensor(ego_history, reverse=True)
    #     output_list_dict["generic_agents.ego"] = [ego_tensor]

    #     # agent features
    #     for feature_name in self._agent_features:

    #         if f"past_tracked_objects.{feature_name}" in list_tensor_data:
    #             agents: List[torch.Tensor] = list_tensor_data[f"past_tracked_objects.{feature_name}"]
    #             agent_history = filter_agents_tensor(agents, reverse=True)

    #             if agent_history[-1].shape[0] == 0:
    #                 # Return zero array when there are no agents in the scene
    #                 agents_tensor: torch.Tensor = torch.zeros((len(agent_history), 0, self._agents_states_dim)).float()
    #             else:
    #                 padded_agent_states = pad_agent_states(agent_history, reverse=True)

    #                 local_coords_agent_states = convert_absolute_quantities_to_relative(
    #                     padded_agent_states, anchor_ego_state
    #                 )

    #                 # Calculate yaw rate
    #                 yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)

    #                 agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)

    #             output_list_dict[f"generic_agents.agents.{feature_name}"] = [agents_tensor]

    #     return output_dict, output_list_dict, output_list_list_dict

    # @torch.jit.export
    # def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
    #     """
    #     Inherited. See interface.
    #     """
    #     return {
    #         "past_ego_states": {
    #             "iteration": "0",
    #             "num_samples": str(self._num_past_poses),
    #             "time_horizon": str(self._past_time_horizon),
    #         },
    #         "past_time_stamps": {
    #             "iteration": "0",
    #             "num_samples": str(self._num_past_poses),
    #             "time_horizon": str(self._past_time_horizon),
    #         },
    #         "past_tracked_objects": {
    #             "iteration": "0",
    #             "time_horizon": str(self._past_time_horizon),
    #             "num_samples": str(self._num_past_poses),
    #             "agent_features": ",".join(self._agent_features),
    #         },
    #     }
