import itertools
import time
from typing import List, Optional, Type, cast

import numpy as np
import numpy.typing as npt
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
# import imageio
from tqdm import tqdm

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
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.callbacks.utils.visualization_utils import get_generic_raster_from_vector_map
from nuplan.common.maps.abstract_map import SemanticMapLayer

def dfs(adj_matrix, visited, node, component):
    visited[node] = True
    component.append(int(node))
    
    neighbors = torch.nonzero(adj_matrix[node]).flatten()
    for neighbor in neighbors:
        if not visited[neighbor]:
            dfs(adj_matrix, visited, neighbor, component)

def connected_components(adj_matrix):
    n = adj_matrix.size(0)
    visited = torch.zeros(n, dtype=torch.bool)
    components = []
    
    for node in range(n):
        if not visited[node]:
            component = []
            dfs(adj_matrix, visited, node, component)
            components.append(component)
    
    return components


def prune_obs_features(features):
    # Filter map features
    map_keys = list(features['vector_set_map'].coords.keys())
    for key in map_keys:
        points = features['vector_set_map'].coords[key]
        mask = filter_points(points) # (points.norm(dim=-1) < 15) # .any(dim=2, keepdim=True).repeat(1,1,20)
        features['vector_set_map'].coords[key][~mask] = 0
        features['vector_set_map'].availabilities[key][~mask] = False

    agent_keys = list(features['generic_agents'].agents.keys())
    for key in agent_keys:
        batch_size = len(features['generic_agents'].agents[key])
        for batch_idx in range(batch_size):
            points = features['generic_agents'].agents[key][batch_idx][-1,:,:2]
            # num_timesteps = features['generic_agents'].agents[key][batch_idx].shape[0]
            mask = filter_points(points) # (points.norm(dim=-1) < 15) # [None].repeat(num_timesteps,1)
            features['generic_agents'].agents[key][batch_idx] = \
                features['generic_agents'].agents[key][batch_idx][:,mask]

    return features


def filter_points(points):
    # warp so that +x distance counts for less
    # then filter by distance/threshold
    points = points.clone()
    points[...,0] -= 10
    points[...,0] *= 0.5
    mask = points.norm(dim=-1) < 10
    return mask


def visualize_denoising(predictions, features):
    colors = np.array([
        np.linspace(0,0,16),
        np.linspace(0,1,16),
        np.linspace(1,0,16)
    ]).T

    frames = []
    for i, intermediate_trajs in enumerate(predictions['intermediate']):
        trajs = intermediate_trajs.squeeze(0).reshape(-1,16,3).cpu().numpy()

        frame = get_generic_raster_from_vector_map(
            features['vector_set_map'].to_device('cpu'),
            features['generic_agents'].to_device('cpu'),
            trajectories=trajs,
            # agent_weights=agent_weights[i],
            pixel_size=0.1,
            radius=60
        )

        frames.append(frame)
        # plt.close()

    frames = [Image.fromarray(frame) for frame in frames]
    fname_dir = '/zfsauton2/home/brianyan/nuplan-devkit/nuplan/planning/simulation/planner/ml_planner/viz/'
    fname = f'{fname_dir}denoise.gif'
    frames[0].save(fname, save_all=True, append_images=frames[1:], duration=int(32 * 0.25), loop=0)

    raise


class DiffusionPlanner(MLPlanner):

    requires_scenario: bool = True

    def __init__(self, 
        model: TorchModuleWrapper, 
        scenario: AbstractScenario,
        goal_mode: str,
        constraint_mode: str,
        replan_freq: int
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval = model.future_trajectory_sampling.step_time
        self._num_output_dim = model.future_trajectory_sampling.num_poses

        self._model_loader = ModelLoader(model)

        self._initialization: Optional[PlannerInitialization] = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self.plan = None
        self.plan_status = None

        self.goal = None
        self.goal_type = None
        self.goal_reached = False
        self.local_goal = None

        self._counter = 0
        self._frames = []

        self._scenario = scenario
        self._gt_ego_state = None
        self._best_gt_goal_dist = np.inf

        assert goal_mode in ('llm', 'gt')
        assert constraint_mode in ('ours', 'motiondiffuser', 'ctg', 'inpaint')
        self.goal_mode = goal_mode
        self.constraint_mode = constraint_mode

        self.replan_frequency = replan_freq
        self.replan_counter = 0
        self.states = None

    def _get_lanes(self, idx):
        ego_state = self.current_input.history.current_state[0]
        radius = 100.0
        lanes = list(self._initialization.map_api.get_proximal_map_objects(ego_state.rear_axle, radius, [SemanticMapLayer.LANE]).values())[0]
        
        lane = []
        for node in lanes[idx].baseline_path.discrete_path:
            lane.append([node.x, node.y])

        return self.convert_to_local(np.array(lane[::5]))
    
    def _get_left_lane(self):
        ego_state = self.current_input.history.current_state[0]
        radius = 10.0
        current = list(self._initialization.map_api.get_proximal_map_objects(ego_state.center, radius, [SemanticMapLayer.LANE]).values())[0][-1]  
        left = current.adjacent_edges[0]
        lane = []
        for node in left.baseline_path.discrete_path:
            lane.append([node.x, node.y])

        return self.convert_to_local(np.array(lane[::2])) 
    

    def _get_right_lane(self):
        ego_state = self.current_input.history.current_state[0]
        radius = 20.0
        current = list(self._initialization.map_api.get_proximal_map_objects(ego_state.center, radius, [SemanticMapLayer.LANE]).values())[0][-2]  
        right = current.adjacent_edges[1]
        lane = []
        for node in right.baseline_path.discrete_path:
            lane.append([node.x, node.y])

        return self.convert_to_local(np.array(lane[::10])) 
    
    def _get_next_lane(self):
        ego_state = self.current_input.history.current_state[0]
        radius = 0.3
        while True:
            lanes = list(self._initialization.map_api.get_proximal_map_objects(ego_state.center, radius, [SemanticMapLayer.LANE]).values())[0]
            connectors = list(self._initialization.map_api.get_proximal_map_objects(ego_state.center, radius, [SemanticMapLayer.LANE_CONNECTOR]).values())[0]
            if len(lanes) > 0:
                current = lanes[0]
                connector = current.outgoing_edges[0]
                next = connector.outgoing_edges[0]
                connector_lane = []
                for node in connector.baseline_path.discrete_path:
                    connector_lane.append([node.x, node.y])
                break
            elif len(connectors) > 0: 
                current = connectors[0]
                next = current.outgoing_edges[0]
                break
            else:
                radius += 0.2
        
        current_lane = []
        for node in current.baseline_path.discrete_path:
            current_lane.append([node.x, node.y])

        lane = []
        for node in next.baseline_path.discrete_path:
            lane.append([node.x, node.y])
        
        if len(lanes) > 0:
            return self.convert_to_local(np.array(current_lane[::5])), self.convert_to_local(np.array(connector_lane[::5])), self.convert_to_local(np.array(lane[::5])) 
        else:
            return self.convert_to_local(np.array(current_lane[::5])), None, self.convert_to_local(np.array(lane[::5])) 

    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        # Manually setting guidance parameters
        if self.local_goal is not None:
            raise
            features['guidance_target'] = self.local_goal # goal
            features['guidance_weight'] = 1.0
            features['w'] = -1.0
            features['guidance_mode'] = self.constraint_mode
            # if self.constraint_mode in ('motiondiffuser', 'ctg'):
            #     features['guidance_weight'] = 1.0

        self._model_loader._model.set_sampling_steps(32, 80.0, 2e-3, 7, strategy='edm', freeze_t=0, freeze_steps=0)
        self._model_loader._model.predictions_per_sample = 16

        predictions = self._model_loader.infer(features)
        trajectories = predictions['multimodal_trajectories'][0].reshape(-1,16,3).cpu().numpy()

        # Visualize attention weights
        # 0:31 => agents (ego first)
        # 32:191 => map features
        # 192:207 => trajectory
        # Right now we just take the average over all samples
        # TODO: right now this is wrong
        # raise
        # agent_weights = predictions['attention_weights'][:,:,:,:31].mean(1)[:,-16:].mean(1).cpu().numpy()
        # agent_weights /= agent_weights.max(axis=1)[:,None]

        # # Visualize denoising process
        if self._counter == 7:
            visualize_denoising(predictions, features)

        # Extract trajectory prediction
        trajectory_predicted = cast(Trajectory, predictions['trajectory'])
        trajectory_tensor = trajectory_predicted.data
        trajectory = trajectory_tensor.cpu().detach().numpy()[0]  # retrieve first (and only) batch as a numpy array
        
        return (
            cast(npt.NDArray[np.float32], trajectory),
            trajectories,
            predictions['probabilities'][0].cpu().numpy(),
            # agent_weights
        )

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._model_loader.initialize()
        self._initialization = initialization

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def get_plan(self):
        """
        LLM HELPERS
        """

        # FEATURE HELPERS

        def get_ego_features():
            return self.features['generic_agents'].ego[0].cpu().numpy()[-1]

        def get_agent_features(radius=35):
            agent_features = self.features['generic_agents'].agents['VEHICLE'][0][-1].cpu().numpy()
            dists = np.linalg.norm(agent_features[:,:2], axis=-1)
            agent_features = agent_features[dists < radius]
            return agent_features

        def get_lane_features():
            # Load features
            vector_set_map = cast(VectorSetMap, self.features['vector_set_map'])
            all_road_segments = vector_set_map.coords['LANE'][0].cpu().numpy()
            all_road_segments_mask = vector_set_map.availabilities['LANE'][0].cpu().numpy()
            all_road_segments = all_road_segments[all_road_segments_mask]
            road_segments_on_route = vector_set_map.coords['ROUTE_LANES'][0]
            mask = vector_set_map.availabilities['ROUTE_LANES'][0]
            # Ignore empty segments
            road_segments_on_route = road_segments_on_route[mask.any(dim=1)]
            mask = mask[mask.any(dim=1)]
            # Get endpoints
            endpoints = []
            num_segments = road_segments_on_route.shape[0]
            for i in range(num_segments):
                non_masked_indices = torch.nonzero(mask[i])
                endpoints.append(
                    torch.cat([
                        road_segments_on_route[i,non_masked_indices[0]],
                        road_segments_on_route[i,non_masked_indices[-1]]
                    ], dim=0)
                )
            endpoints = torch.stack(endpoints, dim=0)
            
            # Compute orientations + pairwise orientation differences
            deltas = endpoints[:,1] - endpoints[:,0]
            orientations = torch.atan2(deltas[:,1], deltas[:,0])
            pairwise_orientations = orientations[:,None] - orientations[None]
            adjacencies_orientation = pairwise_orientations < 0.05 # TODO: magic number
            
            # Compute pairwise distances
            pairwise_dists = (endpoints[:,None,:,None] - endpoints[None,:,None]).norm(dim=-1).reshape(num_segments,num_segments,-1).min(dim=-1).values
            adjacencies_position = pairwise_dists < 2.0 # TODO: magic number
            # torch.eye(num_segments, device=pairwise_dists.device)
            # adjacencies_1 = adjacencies_position * adjacencies_orientation
            adjacencies_1 = torch.eye(num_segments, device=pairwise_dists.device) # adjacencies_position

            # Connect close segments
            component_idxs = connected_components(adjacencies_1)
            components = []
            for group in component_idxs:
                points = []
                for segment_idx in group:
                    points.append(road_segments_on_route[segment_idx][mask[segment_idx]])
                points = torch.cat(points, dim=0)
                components.append(points)
            lane_features_1 = [component.cpu().numpy() for component in components]

            # component_idxs = connected_components(adjacencies_2)
            # components = []
            # for group in component_idxs:
            #     points = []
            #     for segment_idx in group:
            #         points.append(road_segments_on_route[segment_idx][mask[segment_idx]])
            #     points = torch.cat(points, dim=0)
            #     components.append(points)
            # lane_features_2 = [component.cpu().numpy() for component in components]
            # return lane_features_1, lane_features_2
            return lane_features_1

        # GOAL HELPERS

        def clear_goal():
            self.clear_goal()

        def goal_reached():
            return self.goal_reached

        def set_static_goal(pos, threshold=5.0):
            self.goal_type = 'static'
            self.goal = (self.convert_to_global(pos), threshold)
            self.update_goal()

        def set_dynamic_goal(anchor_id, offset, threshold=5.0):
            self.goal_type = 'dynamic'
            anchor_pos = get_agent_features()[anchor_id][:2]
            global_anchor_pos = self.convert_to_global(anchor_pos)
            self.goal = (global_anchor_pos, offset, threshold)
            self.update_goal()

        def get_adjacent_lane_centers(thresh=3.0):
            """
            Find lanes that intersect with line perpendicular to ego heading
            """
            lanes = get_lane_features()
            lane_poses = []
            for lane in lanes:
                lon_dists = np.abs(lane[:,0])
                if np.min(lon_dists) > thresh:
                    continue
                position = lane[np.argmin(lon_dists)]
                delta = lane[-1] - lane[0]
                heading = np.arctan2(delta[1], delta[0])
                lane_poses.append(np.array([
                    position[0], position[1], heading
                ]))

            lane_poses = np.stack(lane_poses)
            lane_poses = lane_poses[np.argsort(lane_poses[:,1])]
            
            current_lane_id = np.argmin(np.abs(lane_poses[:,1]))
            right_lanes = lane_poses[:current_lane_id][::-1]
            current_lane = lane_poses[current_lane_id]
            left_lanes = lane_poses[current_lane_id+1:]

            # if len(right_lanes) == 0:
            #     right_lanes = current_lane.reshape(1,3)
            # if len(left_lanes) == 0:
            #     left_lanes = current_lane.reshape(1,3)

            return current_lane, left_lanes, right_lanes

        def get_agent_poses():
            """
            Computes relative position and orientation of agents
            """
            return get_agent_features()[...,:3]

        def get_heading_and_distance_to_agents():
            """
            Computes heading relative to the ego for each agent, plus distance to ego
            Useful for stuff like "to the left", "up head", etc.
            """
            agent_poses = get_agent_poses()
            headings = np.arctan2(agent_poses[:,1], agent_poses[:,0])
            distances = np.linalg.norm(agent_poses[:,:2], axis=-1)
            return headings, distances

        def get_agent_speeds():
            """
            Computes speed of agents
            """
            agent_speeds = get_agent_features()[...,3:5]
            return np.linalg.norm(agent_speeds, axis=-1)

        def get_rotation_matrix(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])

        def rotate_to_pose_frame(position, pose):
            R = get_rotation_matrix(pose[2])
            position = R @ position
            return position

        def transform_to_pose_frame(position, pose):
            """
            Transform position to pose frame
            """
            position = rotate_to_pose_frame(position, pose)
            position = position + pose[:2]
            return position

        def seconds_to_steps(seconds):
            return seconds * 2

        """
        INSERT LLM CODE HERE
        """
        def make_plan():
            # set_static_goal(np.array([[15.0, -18.0]]))
            while True:
                yield 'running'

        return make_plan()

    def convert_to_global(self, pos):
        ego_state = self.current_input.history.current_state[0]
        ref = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])
        rot = np.array([
            [np.cos(ref[2]), -np.sin(ref[2])],
            [np.sin(ref[2]),  np.cos(ref[2])]
        ])

        target = np.empty_like(pos)
        for i in range(pos.shape[0]):
            target[i] = rot @ pos[i]    
            target[i] += ref[:2]

        # target = rot @ pos
        # target += ref[:2]
        return target

    def convert_to_local(self, pos):
        ego_state = self.current_input.history.current_state[0]
        ref = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])
        rot = np.array([
            [np.cos(ref[2]), -np.sin(ref[2])],
            [np.sin(ref[2]),  np.cos(ref[2])]
        ])

        target = np.empty_like(pos)
        for i in range(pos.shape[0]):
            target[i] = pos[i] - ref[:2]
            target[i] = rot.T @ target[i]

        return target


    def clear_goal(self):
        self.goal = None
        self.goal_reached = False
        self.local_goal = None

    def update_goal(self):
        if self.goal is None:
            return

        if self.goal_type == 'static':
            # Convert global to local
            global_goal, threshold = self.goal
            self.local_goal = self.convert_to_local(global_goal)
            curr_pos = self.features['generic_agents'].ego[0].cpu().numpy()[-1][:2]
            self.goal_reached = np.linalg.norm(self.local_goal - curr_pos) < threshold

        elif self.goal_type == 'dynamic':
            # Let's just say the closest to the dynamic target is the same one
            # TODO: Do something less stupid for dynamic tracking
            global_anchor, offset, threshold = self.goal
            local_anchor = self.convert_to_local(global_anchor)
            agent_features = self.features['generic_agents'].agents['VEHICLE'][0][-1].cpu().numpy()
            dists = np.linalg.norm(local_anchor[None] - agent_features[:,:2], axis=-1)
            new_local_anchor = agent_features[np.argmin(dists)][:2]
            # Check if reached
            self.local_goal = new_local_anchor + offset
            curr_pos = self.features['generic_agents'].ego[0].cpu().numpy()[-1][:2]
            self.goal_reached = np.linalg.norm(self.local_goal - curr_pos) < threshold
            # Update global anchor poss
            new_global_anchor = self.convert_to_global(new_local_anchor)
            self.goal = (new_global_anchor, offset, threshold)

        else:
            raise NotImplementedError

    def get_gt_goal_error(self):
        global_gt_goal = self._gt_ego_state
        local_gt_goal = self.convert_to_local(global_gt_goal)
        return np.linalg.norm(local_gt_goal)

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

        # Query scenario for actual GT waypoints (used for some evals)
        if self._gt_ego_state is None:
            current_state = self._scenario.get_ego_state_at_iteration(current_input.iteration.index)
            states = self._scenario.get_ego_future_trajectory(
                current_input.iteration.index, 8, 16
            )
            self._gt_ego_state = list(states)[-1]
            self._gt_ego_state = np.array([
                self._gt_ego_state.rear_axle.x,
                self._gt_ego_state.rear_axle.y
            ])

        if self.goal_mode == 'llm':
            # Run LLM planner if we don't have one
            if self.plan is None:
                self.plan = self.get_plan()

            # Execute LLM plan
            if self.plan_status != 'done':
                self.update_goal()
                self.plan_status = next(self.plan)
                if self.plan_status == 'done':
                    self.clear_goal()
                    print('Plan done!')
        elif self.goal_mode == 'gt':
            self.local_goal = self.convert_to_local(self._gt_ego_state)
        else:
            raise NotImplementedError

        # Infer model
        start_time = time.perf_counter()
        # predictions, trajectories, probabilities = self._infer_model(features)
        # self._inference_runtimes.append(time.perf_counter() - start_time)

        # Convert relative poses to absolute states and wrap in a trajectory object.
        if self.replan_counter % self.replan_frequency == 0:
            predictions, trajectories, probabilities = self._infer_model(features)
            self.states = transform_predictions_to_states(
                predictions, history.ego_states, self._future_horizon, self._step_interval
            )

            # features = prune_obs_features(features)

            # current_lane, connector, next_lane = self._get_next_lane()

            frame = get_generic_raster_from_vector_map(
                features['vector_set_map'].to_device('cpu'),
                features['generic_agents'].to_device('cpu'),
                trajectories=trajectories,
                probabilities=probabilities,
                # goal=self.local_goal,
                # current=current_lane,
                # connector=connector,
                # next=next_lane,
                # agent_weights=agent_weights,
                pixel_size=0.1,
                radius=60
            )
            # frame = get_generic_raster_from_vector_map(
            #     features['vector_set_map'].to_device('cpu'),
            #     features['generic_agents'].to_device('cpu'),
            #     trajectories=trajectories,
            #     probabilities=probabilities,
            #     pixel_size=0.1,
            #     radius=35
            # )
            # path = f'/zfsauton2/home/brianyan/nuplan-diffusion/nuplan/planning/simulation/planner/ml_planner/viz/test_{self._counter}.png'
            self._counter += 1
            # cv2.imwrite(path, frame[...,::-1])
            self._frames.append(frame)
            if self._counter == 29:
                # Save to gif
                frames = [Image.fromarray(frame) for frame in self._frames]
                fname_dir = '/zfsauton2/home/brianyan/nuplan-devkit/nuplan/planning/simulation/planner/ml_planner/viz/'
                # fname = f'{fname_dir}{time.strftime("%Y%m%d-%H%M%S")}.gif'
                fname = f'{fname_dir}viz.gif'
                frames[0].save(fname, save_all=True, append_images=frames[1:], duration=int(self._counter * 1.5), loop=0)
                # self._frames = []
                # raise
        
            print(self._counter)

        self.replan_counter += 1

        trajectory = InterpolatedTrajectory(self.states)

        self._inference_runtimes.append(time.perf_counter() - start_time)

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


import io
import cv2

def fig_to_numpy(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = img[:,:,::-1]
    return img
