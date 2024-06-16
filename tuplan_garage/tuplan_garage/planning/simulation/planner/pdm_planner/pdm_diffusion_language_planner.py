import os
import traceback
import threading
import json

import numpy as np
import torch
import dspy
from dspy.teleprompt import LabeledFewShot

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_diffusion_planner import PDMDiffusionPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.diffusion_utils import convert_to_local
from tuplan_garage.planning.simulation.planner.pdm_planner.language.helpers import (
    Vehicle, 
    VehicleGraph,
    Lane,
    LaneGraph
)
from tuplan_garage.planning.simulation.planner.pdm_planner.language.examples import load_examples


class GenerateCodeFromInstruction(dspy.Signature):
    """Use the provided language instruction to write code for guiding a lower-level driving policy."""
    instruction = dspy.InputField()
    code=dspy.OutputField()


class GenerateCode(dspy.Module):
    def __init__(self):
        super().__init__()
        
        self.predict = dspy.Predict(GenerateCodeFromInstruction)

    def forward(self, instruction):
        prediction = self.predict(instruction=instruction)
        return dspy.Prediction(code=prediction.code)


def reformat_output_as_generator(code):
    """
    Since `exec` cannot handle yield statements outside of function definitions,
    reformat the code to add a function definition.
    """
    lines = code.split('\n')
    lines = [(' ' * 4) + line for line in lines]
    lines = ['def make_plan(self):'] + lines
    new_code = '\n'.join(lines)
    return new_code


class PDMDiffusionLanguagePlanner(PDMDiffusionPlanner):
    def __init__(
        self, 
        language_config, 
        experiment_log_path,
        nuplan_output_dir,
        *args, 
        **kwargs
    ):
        super().__init__(use_pdm_proposals=False ,*args, **kwargs)

        self.language_config = language_config
        self.plan = None
        self.goal_reached = False

        self.experiment_log_path = experiment_log_path
        self.experiment_logger = ExperimentLogger(experiment_log_path)
        self.nuplan_output_dir = nuplan_output_dir

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__
    
    def build_llm_module(self):
        # Configure LM
        self.llm = dspy.OpenAI(
            model=self.language_config['model'],
            temperature=self.language_config['temperature'],
            max_tokens=self.language_config['max_tokens']
        )
        dspy.settings.configure(lm=self.llm)

        # Load examples
        examples = load_examples()

        # Build and compile module
        llm_module = GenerateCode()
        tp = LabeledFewShot(k=len(examples))
        self.llm_module = tp.compile(llm_module, trainset=examples)

    def generate_plan(self):
        """
        Produce language-conditioned plan
        """
        # Invoke LLM here
        self.build_llm_module()
        instruction = self.language_config['instruction']

        output = self.llm_module(instruction=instruction)
        code = output.code
        print(f'LANGUAGE INSTRUCTION: {instruction}')

        try:
            print('LLM GENERATED CODE:')
            print(code)
            code = reformat_output_as_generator(code)
            exec(code, globals())
            self.plan = make_plan(self)
        except Exception:
            print('*' * 40)
            print('Exception encountered while executing LLM code, trace below')
            print('*' * 40)
            print(traceback.format_exc())
            print('*' * 40)
            import pdb; pdb.set_trace()

    """
    PERCEPTION HELPERS
    """
    def update_graphs(self, current_input):
        self.lane_graph, lane_object_to_node = self.generate_lane_graph()
        self.current_lane = lane_object_to_node[self._get_starting_lane(self.ego_state).id]
        self.ego_vehicle, self.vehicle_graph = self.generate_vehicle_graph(current_input)

        for vehicle in self.vehicle_graph.get_vehicles():
            closest_lane = vehicle.get_closest_lane(self.lane_graph)
            closest_lane.vehicles.append(vehicle)

    @property
    def left_lane(self):
        return self.current_lane.get_left_lane()

    @property
    def right_lane(self):
        return self.current_lane.get_right_lane()
    
    def get_vehicle(self, idx):
        token = self.vehicle_idx_to_token[idx]
        return self.vehicle_graph.get_by_token(token)
    
    def get_lane(self, idx):
        token = self.lane_idx_to_token[idx]
        return self.lane_graph.get_by_token(token)

    """
    CONTROL HELPERS
    """
    def follow_lane(self, lane):
        centerline, tokens = lane.extend_centerline(return_tokens=True)
        self._scorer.add_lane_metric(centerline, use_dense=False)

        def check_lane_callback():
            done = self.current_lane.token in tokens
            return done

        return check_lane_callback
    
    def stop_following(self):
        self._scorer.remove_lane_metric()
    
    def yield_to_vehicle(self, vehicle):
        self._scorer.add_yield_metric(vehicle.token)

        def check_yield_callback():
            # TODO: implement this
            return False

        return check_yield_callback
    
    def stop_yielding(self):
        self._scorer.remove_yield_metric()

    def set_ego_speed_limit(self, speed):
        self._scorer.add_soft_speed_metric(speed)

    def unset_ego_speed_limit(self):
        self._scorer.remove_soft_speed_metric()

    def adjust_constant_velocity_prediction(self, vehicle, ratio):
        self._observation.set_velocity_ratio(vehicle.token, ratio)

    """
    OTHER PLANNER FUNCTIONS
    """

    def _get_closed_loop_trajectory(self, current_input):
        self.update_graphs(current_input)

        # Save idx to token at initial timestep for future reference
        if self._iteration == 0:
            self.vehicle_idx_to_token = {
                idx: self.vehicle_graph.get_by_idx(idx).token 
                for idx in range(self.vehicle_graph.get_num_vehicles())
            }
            self.lane_idx_to_token = {
                idx: self.lane_graph.get_by_idx(idx).token 
                for idx in range(self.lane_graph.get_num_lanes())
            }

        if not self.language_config['disable']:
            if self._iteration == 0:
                self.generate_plan()

            if self._iteration % self._replan_freq == 0:
                if self.plan is not None:
                    next(self.plan, None)

        if not self.goal_reached:
            self.goal_reached = self.check_task_completion()
            if self.goal_reached:
                print(f'Goal reached!')

        if self._iteration == 148:
            self.experiment_logger.log_value({
                'success': int(self.goal_reached),
                'metric_path': self.nuplan_output_dir
            })
            print(f'Saved results to {self.experiment_log_path}')

        return super()._get_closed_loop_trajectory(current_input)
    
    def generate_lane_graph(self):
        map_object_types = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR] # self._map_api.get_available_map_objects()
        map_objects = self._map_api.get_proximal_map_objects(self.ego_state.center.point, 100.0, layers=map_object_types)
        nodes = []
        for map_object_type in map_object_types:
            for map_object in map_objects[map_object_type]:
                # if map_object.id in self._route_lane_dict:
                nodes.append(map_object)
    
        # Parse map objects to get lane graph
        nodes = [Lane(node) for node in nodes]
        lane_object_to_node = {node.lane_object.id: node for node in nodes}

        # Extract connectivity information from NuPlan map objects
        for node in nodes:
            node.incoming = [
                lane_object_to_node[obj.id] for obj in node.lane_object.incoming_edges
                if obj.id in lane_object_to_node
            ]
            node.outgoing = [
                lane_object_to_node[obj.id] for obj in node.lane_object.outgoing_edges
                if obj.id in lane_object_to_node
            ]
            node.adjacent = [
                lane_object_to_node[obj.id] if (obj is not None) and (obj.id in lane_object_to_node) else None
                for obj in node.lane_object.adjacent_edges
            ]

        lane_graph = LaneGraph(nodes)
        return lane_graph, lane_object_to_node

    def generate_vehicle_graph(self, current_input):
        observation = current_input.history.current_state[1]
        vehicles = []
        for object in observation.tracked_objects:
            if object.tracked_object_type == TrackedObjectType.VEHICLE:
                vehicles.append(object)
        vehicles = [Vehicle(obj) for obj in vehicles]
        
        ego_vehicle = Vehicle(self.ego_state.agent)
        vehicles = [vehicle for vehicle in vehicles if vehicle.distance_to(ego_vehicle) < 100]
        vehicle_graph = VehicleGraph(vehicles)

        return ego_vehicle, vehicle_graph

    def plot_scene_graph(self, path):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(20,20))

        # Lane graph
        for i, lane in enumerate(self.lane_graph.get_lanes()):
            centerline = convert_to_local(self.ego_state, lane.centerline)
            plt.plot(centerline[:,0], centerline[:,1], color='black')
            median_idx = centerline.shape[0] // 2
            plt.text(centerline[median_idx,0], centerline[median_idx,1], f'{i}')

        left_lane = self.current_lane.get_left_lane()
        right_lane = self.current_lane.get_right_lane()

        lanes = [self.current_lane, left_lane, right_lane]
        colors = ['purple', 'yellow', 'orange']

        for color, lane in zip(colors, lanes):
            if lane is None:
                continue
            centerline = convert_to_local(self.ego_state, lane.extend_centerline())
            plt.plot(centerline[:,0], centerline[:,1], color=color)

        # Vehicle graph
        for vidx, vehicle in enumerate(self.vehicle_graph.get_vehicles()):
            position = convert_to_local(self.ego_state, vehicle.position)
            plt.scatter(position[0], position[1], color='red')
            plt.text(position[0], position[1], str(vidx))

        ego_position = convert_to_local(self.ego_state, self.ego_vehicle.position)
        plt.scatter(ego_position[0], ego_position[1], color='blue')
        plt.text(ego_position[0], ego_position[1], 'ego')

        plt.xlim(-100,100)
        plt.ylim(-100,100)

        plt.savefig(path)
        plt.close(fig)

    def check_task_completion(self):
        try:
            if self._log_name == '2021.07.24.18.06.35_veh-35_00016_03642':
                # Scenario 1 -- Lane change
                lane = self.get_lane(34)
                return self.lane_goal_fn(lane.token)

            elif self._log_name == '2021.09.14.15.03.51_veh-45_01205_01789':
                # Scenario 2 -- Unprotected left turn
                # goal = np.array([587677.1590276195, 4475514.034451011])
                # return self.point_goal_fn(goal)
                return self.ego_vehicle.is_ahead_of(self.get_vehicle(18))
            
            elif self._log_name == '2021.06.07.19.29.59_veh-38_01949_02349':
                # Scenario 3 -- Unprotected right turn
                lane = self.get_lane(33)
                return self.lane_goal_fn(lane.token)

            elif self._log_name == '2021.06.07.18.29.03_veh-16_00049_00824':
                # Scenario 4 -- Overtaking
                # Must be ahead of vehicle in same lane
                vehicle = self.get_vehicle(4)
                is_ahead = self.ego_vehicle.is_ahead_of(vehicle)
                _, ego_lane_tokens = self.current_lane.extend_centerline(return_tokens=True)
                other_closest_lane = vehicle.get_closest_lane(self.lane_graph)
                is_same_lane = other_closest_lane.token in ego_lane_tokens
                return is_ahead and is_same_lane
            
            elif self._log_name == '2021.07.24.00.36.59_veh-47_00439_02454':
                # Scenario 5 -- Extended overtaking
                vehicle = self.get_vehicle(3)
                is_ahead = self.ego_vehicle.is_ahead_of(vehicle)
                _, ego_lane_tokens = self.current_lane.extend_centerline(return_tokens=True)
                other_closest_lane = vehicle.get_closest_lane(self.lane_graph)
                is_same_lane = other_closest_lane.token in ego_lane_tokens
                return is_ahead and is_same_lane
            
            elif self._log_name == '2021.07.24.16.07.03_veh-35_03033_05899':
                # Scenario 6 -- Yielding
                vehicle = self.get_vehicle(8)
                is_behind = vehicle.is_ahead_of(self.ego_vehicle)
                _, ego_lane_tokens = self.current_lane.extend_centerline(return_tokens=True)
                other_closest_lane = vehicle.get_closest_lane(self.lane_graph)
                is_same_lane = other_closest_lane.token in ego_lane_tokens
                return is_behind and is_same_lane
            
            elif self._log_name == '2021.07.24.23.59.52_veh-12_01548_02862':
                # Scenario 7 -- Cut in
                vehicle = self.get_vehicle(2)
                is_ahead = self.ego_vehicle.is_ahead_of(vehicle)
                _, ego_lane_tokens = self.current_lane.extend_centerline(return_tokens=True)
                other_closest_lane = vehicle.get_closest_lane(self.lane_graph)
                is_same_lane = other_closest_lane.token in ego_lane_tokens
                return is_ahead and is_same_lane
        
            elif self._log_name == '2021.06.07.12.42.11_veh-38_04779_06284':
                # TODO
                # Scenario 8 -- Lane weaving
                vehicle = self.get_vehicle(2)
                is_ahead = self.ego_vehicle.is_ahead_of(vehicle)
                vehicle2 = self.get_vehicle(9)
                is_behind = vehicle2.is_ahead_of(self.ego_vehicle)
                _, ego_lane_tokens = self.current_lane.extend_centerline(return_tokens=True)
                other_closest_lane = vehicle.get_closest_lane(self.lane_graph)
                is_same_lane = other_closest_lane.token in ego_lane_tokens
                return is_ahead and is_behind and is_same_lane

        except KeyError:
            print('Could not find goal key')
            return False

    def point_goal_fn(self, goal, thresh=3.0):
        curr = np.array([self.ego_state.center.x, self.ego_state.center.y])
        dist = np.linalg.norm(curr - goal)
        return dist < thresh

    def lane_goal_fn(self, goal_lane_token):
        _, tokens = self.current_lane.extend_centerline(return_tokens=True)
        return goal_lane_token in tokens


class ExperimentLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.lock = threading.Lock()

    def log_value(self, value):
        with self.lock:
            # Load data from log file, or create new data if log file doesn't exist
            if not os.path.exists(self.log_file):
                data = []
            else:
                with open(self.log_file, 'r') as file:
                    data = json.load(file)

            # Update dict with new values
            data.append(value)

            # Save to log file
            with open(self.log_file, 'w') as file:
                json.dump(data, file)
