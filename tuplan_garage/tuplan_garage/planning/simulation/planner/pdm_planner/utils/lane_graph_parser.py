import numpy as np

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import states_se2_to_array


def parse_lane_graph(nodes):
    """
    Reduce list of lane nodes (e.g LANE, LANE CONNECTOR objects)
    Returns list of lane polylines (with connections)
    """
    nodes = [LaneNode(node) for node in nodes]
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
            lane_object_to_node[obj.id] for obj in node.lane_object.adjacent_edges
            if (obj is not None) and (obj.id in lane_object_to_node)
        ]

    # Reduce nodes by grouping
    # Can group A and B if A uniquely -> B or vice versa
    checked_nodes = []
    while len(nodes) > 0:
        node = nodes.pop()
        # Merged with incoming/outgoing edges if possible
        if len(node.incoming) == 1:
            incoming_node = node.incoming[0]
            if len(incoming_node.outgoing) == 1:
                # Merge nodes
                incoming_node.outgoing = node.outgoing
                for new_outgoing_node in incoming_node.outgoing:
                    index = new_outgoing_node.incoming.index(node)
                    new_outgoing_node.incoming[index] = incoming_node

                for adj_node in node.adjacent:
                    if adj_node not in incoming_node.adjacent:
                        incoming_node.adjacent.append(adj_node)
                    if node in adj_node.adjacent:
                        index = adj_node.adjacent.index(node)
                        adj_node.adjacent[index] = incoming_node

                incoming_node.centerline = np.concatenate([
                    incoming_node.centerline,
                    node.centerline
                ], axis=0)
                continue

        if len(node.outgoing) == 1:
            outgoing_node = node.outgoing[0]
            if len(outgoing_node.incoming) == 1:
                # Merge nodes
                outgoing_node.incoming = node.incoming
                for new_incoming_node in outgoing_node.incoming:
                    index = new_incoming_node.outgoing.index(node)
                    new_incoming_node.outgoing[index] = outgoing_node

                for adj_node in node.adjacent:
                    if adj_node not in outgoing_node.adjacent:
                        outgoing_node.adjacent.append(adj_node)
                    if node in adj_node.adjacent:
                        index = adj_node.adjacent.index(node)
                        adj_node.adjacent[index] = outgoing_node

                outgoing_node.centerline = np.concatenate([
                    node.centerline,
                    outgoing_node.centerline
                ], axis=0)
                continue

        # If no merges, then mark as checked and keep removed
        checked_nodes.append(node)

    return checked_nodes


class LaneNode:
    def __init__(self, lane_object):
        self.lane_object = lane_object
        self.centerline = states_se2_to_array(lane_object.baseline_path.discrete_path)

        self.heading = lane_object.baseline_path.discrete_path[0].heading
        
        self.incoming = []
        self.outgoing = []
        self.adjacent = []
