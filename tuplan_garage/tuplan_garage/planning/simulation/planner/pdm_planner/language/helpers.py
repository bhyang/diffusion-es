import numpy as np
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import states_se2_to_array


class Vehicle:
    def __init__(self, vehicle_object):
        self.vehicle_object = vehicle_object
        self.position = np.array([vehicle_object.center.x, vehicle_object.center.y])
        self.heading = vehicle_object.center.heading
        self.speed = np.sqrt(vehicle_object.velocity.x**2 + vehicle_object.velocity.y**2)
        self.token = vehicle_object.track_token

    def copy_from(self, vehicle):
        self.vehicle_object = vehicle.vehicle_object
        self.position = vehicle.position
        self.heading = vehicle.heading
        self.speed = vehicle.speed
        self.token = vehicle.token
    
    def distance_to(self, other):
        if isinstance(other, Vehicle):
            return np.linalg.norm(self.position - other.position)
        elif isinstance(other, Lane):
            dists = np.linalg.norm(self.position[None] - other.centerline[...,:2], axis=-1)
            return dists.min()
    
    def is_ahead_of(self, other, thresh=0.0):
        dxdy = other.position - self.position
        dxdy = dxdy @ np.array([
            [np.cos(self.heading), -np.sin(self.heading)],
            [np.sin(self.heading),  np.cos(self.heading)]
        ])
        return dxdy[0] < -thresh
    
    def get_closest_lane(self, lane_graph):
        lanes = lane_graph.get_lanes()
        dists = [self.distance_to(lane) for lane in lanes]
        closest_lane = lanes[np.argmin(dists)]
        return closest_lane

    def is_stopped(self, thresh=0.2):
        print(self.speed)
        return self.speed < thresh
    

class VehicleGraph:
    def __init__(self, vehicles):
        self.vehicles = vehicles
        self.token_to_idx = {vehicle.token: idx for idx, vehicle in enumerate(vehicles)}

    def get_by_idx(self, idx):
        return self.vehicles[idx]
    
    def get_by_token(self, token):
        return self.vehicles[self.token_to_idx[token]]
    
    def get_vehicles(self):
        return self.vehicles
    
    def get_num_vehicles(self):
        return len(self.vehicles)


class Lane:
    def __init__(self, lane_object):
        self.lane_object = lane_object
        self.centerline = states_se2_to_array(lane_object.baseline_path.discrete_path)
        self.heading = lane_object.baseline_path.discrete_path[-1].heading
        self.speed_limit = lane_object.speed_limit_mps
        self.token = lane_object.id

        self.vehicles = []
        
        self.incoming = []
        self.outgoing = []
        self.adjacent = []

    def extend_centerline(self, return_tokens=False):
        """
        Since the lane segments are short, extend centerline with outgoing edges
        """
        nodes = [self]
        centerline = self.centerline.copy()
        current_node = self
        while True:
            if len(current_node.outgoing) == 0:
                break
            else:
                headings = [calc_angle_diff(current_node.heading, node.heading)
                            for node in current_node.outgoing]
                current_node = current_node.outgoing[np.argmin(headings)]
                centerline = np.concatenate([centerline, current_node.centerline], axis=0)
                nodes = nodes + [current_node]

        # Also extend backwards
        current_node = self
        while True:
            if len(current_node.incoming) == 0:
                break
            else:
                headings = [calc_angle_diff(current_node.heading, node.heading)
                            for node in current_node.incoming]
                current_node = current_node.incoming[np.argmin(headings)]
                centerline = np.concatenate([current_node.centerline, centerline], axis=0)
                nodes = [current_node] + nodes
        
        if return_tokens:
            tokens = [node.token for node in nodes]
            return centerline, tokens
        else:
            return centerline
    
    def get_left_lane(self):
        return self._get_adjacent_lane(0)
    
    def get_right_lane(self):
        return self._get_adjacent_lane(1)

    def _get_adjacent_lane(self, idx):
        """
        If lane_object is LaneConnector, we fetch adjacent lane from next lane
        If there's one next lane, return adjacent lane if it exists. If it doesn't, just return None (no multi-hop)
        If there's multiple next lanes, just take the corresponding left/right one
        If nothing works, just return None
        """
        if isinstance(self.lane_object, NuPlanLane):
            return self.adjacent[idx]
        else:
            if len(self.outgoing) == 0:
                return None
            elif len(self.outgoing) == 1:
                next_lane = self.outgoing[0]
            else:
                headings = [calc_angle_diff(self.heading, node.heading) 
                            for node in self.outgoing]
                next_lane = self.outgoing[np.argmin(headings)]
            return next_lane.adjacent[idx]
        
    def get_vehicles(self):
        return self.vehicles


def estimate_length(centerline):
    start = centerline[0,:2]
    end = centerline[-1,:2]
    return np.linalg.norm(start-end, axis=-1)


def calc_angle_diff(theta1, theta2):
    return ((theta1 - theta2) + np.pi) % (2 * np.pi) - np.pi


class LaneGraph:
    def __init__(self, lanes):
        self.lanes = lanes
        self.token_to_idx = {lane.lane_object.id: idx for idx, lane in enumerate(lanes)}

    def get_by_idx(self, idx):
        return self.lanes[idx]
    
    def get_by_token(self, token):
        return self.lanes[self.token_to_idx[token]]
    
    def get_lanes(self):
        return self.lanes
    
    def get_num_lanes(self):
        return len(self.lanes)
