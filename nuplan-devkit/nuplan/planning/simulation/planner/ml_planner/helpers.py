import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer


def convert_to_global(planner, pos):
    ego_state = planner.current_input.history.current_state[0]
    ref = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])
    rot = np.array([
        [np.cos(ref[2]), -np.sin(ref[2])],
        [np.sin(ref[2]),  np.cos(ref[2])]
    ])

    target = np.empty_like(pos)
    for i in range(pos.shape[0]):
        target[i, :2] = rot @ pos[i, :2]    
        target[i] += ref #[:2]

    return target


def convert_to_local(planner, pos):
    ego_state = planner.current_input.history.current_state[0]
    ref = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])
    rot = np.array([
        [np.cos(ref[2]), -np.sin(ref[2])],
        [np.sin(ref[2]),  np.cos(ref[2])]
    ])

    target = np.empty_like(pos)
    for i in range(pos.shape[0]):
        target[i, :2] = pos[i, :2] - ref[:2]
        target[i, :2] = rot.T @ target[i, :2]
        target[i,  2] = pos[i, 2] - ego_state.rear_axle.heading

    return target


def _get_lanes(planner, idx):
    ego_state = planner.current_input.history.current_state[0]
    radius = 100.0
    lanes = list(planner._initialization.map_api.get_proximal_map_objects(ego_state.rear_axle, radius, [SemanticMapLayer.LANE]).values())[0]
        
    lane = []
    # for node in lanes[idx].baseline_path.discrete_path:
    #     lane.append([node.x, node.y])

    return lanes[idx] # convert_to_local(planner, np.array(lane[::5]))
    

def _get_current(planner):
    ego_state = planner.current_input.history.current_state[0]
    radius = 0.05
    while True:
        lanes = list(planner._initialization.map_api.get_proximal_map_objects(ego_state.center, radius, [SemanticMapLayer.LANE]).values())[0]
        connectors = list(planner._initialization.map_api.get_proximal_map_objects(ego_state.center, radius, [SemanticMapLayer.LANE_CONNECTOR]).values())[0]
        if len(lanes) > 0:
            current = lanes[0]
            break
        elif len(connectors) > 0: 
            current = connectors[0]
            break
        else:
            radius += 0.05
        
    return current
    

def _get_left(self, current):
    # Check if current lane is the left-most lane
    left = current.adjacent_edges[0] if current.adjacent_edges[0] else current

    return left
    

def _get_next_left_lane(self):
    ego_state = self.current_input.history.current_state[0]
    radius = 0.05
    while True:
        lanes = list(self._initialization.map_api.get_proximal_map_objects(ego_state.center, radius, [SemanticMapLayer.LANE]).values())[0]  
        if len(lanes) > 0:
            current = lanes[0]
            break
        else:
            radius += 0.05

    left = current.adjacent_edges[0]
    connector = left.outgoing_edges[0]
    next = connector.outgoing_edges[0]
    lane = []
    for node in next.baseline_path.discrete_path:
        lane.append([node.x, node.y])

    return self.convert_to_local(np.array(lane[::5])) 
    

def _get_right(self, current):
    right = current.adjacent_edges[1] if current.adjacent_edges[1] else current

    return right
    

def _get_next_right_lane(planner):
    ego_state = planner.current_input.history.current_state[0]
    radius = 0.05
    while True:
        lanes = list(planner._initialization.map_api.get_proximal_map_objects(ego_state.center, radius, [SemanticMapLayer.LANE]).values())[0]
        if len(lanes) > 0:
            current = lanes[0]
            break
        else:
            radius += 0.2
    right = current.adjacent_edges[1]
    lane = []
    if right is not None:
        connector = right.outgoing_edges[0]
        next = connector.outgoing_edges[0]
    else:
        connector = current.outgoing_edges[0]
        next = connector.outgoing_edges[0]

    for node in next.baseline_path.discrete_path:
        lane.append([node.x, node.y])

    return convert_to_local(planner, np.array(lane[::5])) 
    

def _get_current_lane(planner, current=None):
    if current is None: current = _get_current(planner)
    current_lane = []
    for node in current.baseline_path.discrete_path:
        current_lane.append([node.x, node.y, node.heading])
    
    return convert_to_local(planner, np.array(current_lane[::5])) 
    

def _get_next_lane(planner, current=None):
    if current is None: current = _get_current(planner)
    next = current.outgoing_edges[0]
    current_lane = []
    for node in current.baseline_path.discrete_path:
        current_lane.append([node.x, node.y])

    return np.array(_get_points_of_lane(planner, next))

def _get_next(planner, lane):
    """
    Given lane/lane connector object, return next object.
    
    """

    return lane.outgoing_edges[0]

def _get_points_of_lane(planner, lane):
    """
    Given lane/lane connector, return points.

    """
    nodes = []
    for node in lane.baseline_path.discrete_path:
        nodes.append([node.x, node.y, node.heading])

    return convert_to_local(planner, np.array(nodes[::5]))
