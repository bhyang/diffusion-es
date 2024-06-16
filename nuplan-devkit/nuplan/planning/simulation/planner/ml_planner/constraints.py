import numpy as np
import torch
from torch.nn import functional as F

from nuplan.planning.metrics.utils.state_extractors import extract_ego_corners, extract_ego_time_point
from nuplan.planning.metrics.utils.route_extractor import (
    CornersGraphEdgeMapObject,
    extract_corners_route,
    get_common_or_connected_route_objs_of_corners,
    get_outgoing_edges_obj_dict,
    get_route,
    get_timestamps_in_common_or_connected_route_objs,
)

"""
CONSTRAINTS
"""

# NOTE: the trajectories are batched (B, 48)
# The computation should be independent across the batch dimension
# The output should be shape (B,)

def make_go_fast_constraint(weight=1.0):
    """
    Maximizes distance traveled
    """

    def constraint_fn(trajectory):
        return -trajectory.reshape(-1,3)[...,:2].norm(dim=-1).mean() * weight

    return constraint_fn

def make_stop_constraint():
    """
    Minimizes distance traveled
    """

    def constraint_fn(trajectory):
        return trajectory.norm(dim=-1)

    return constraint_fn

def make_goal_constraint(goal, weight=1.0):
    """
    Minimizes distance to a goal point / a set of goal points
    If multiple goals are provided, takes the minimum error over all goals
    We also take the minimum over the trajectory

    goal shape is (N,2) where N is the number of goals
    """
    def constraint_fn(trajectory):
        target = torch.as_tensor(goal, device=trajectory.device, dtype=trajectory.dtype)[...,:2]
        trajectory = trajectory.reshape(trajectory.shape[0], 16, 3)[...,:2]

        target_heading = heading_from_xy(target)
        trajectory_heading = heading_from_xy(trajectory)

        # Unreduced error is shape (batch_size, N, num_timesteps)
        xy_loss = torch.norm(trajectory[:,None] - target[None,:,None], dim=-1)
        # heading_loss = torch.abs(trajectory_heading - target_heading)
        # https://stats.stackexchange.com/questions/425234/loss-function-and-encoding-for-angles
        # heading_loss = torch.atan2(
        #     torch.sin(target_heading - trajectory_heading),
        #     torch.cos(target_heading - trajectory_heading)
        # )
        heading_loss = -torch.cos(target_heading[None,:,None] - trajectory_heading[:,None])
        loss = xy_loss + heading_loss
        
        # Take minimum over goals
        loss = loss.min(dim=1).values

        # Take mean over timesteps
        loss = loss.mean(dim=1) # .min(dim=2).values

        loss = loss * weight
        return loss

    return constraint_fn


def heading_from_xy(xy):
    original_shape = xy.shape
    xy = xy.reshape(-1,2)

    deltas = torch.diff(xy, dim=0)
    heading = torch.arctan2(deltas[:,0], deltas[:,1])

    # Hack
    heading = torch.cat([heading[:1], heading], dim=0)

    heading = heading.reshape(*original_shape[:-1])
    return heading


def make_speed_constraint(max_speed=1.0, weight=1.0):
    def constraint_fn(trajectory):
        # Estimate speed per step
        trajectory = trajectory.reshape(-1,16,3)
        deltas = torch.diff(trajectory, dim=1, prepend=torch.zeros_like(trajectory[:,:1]))
        speeds = torch.norm(deltas, dim=-1)

        loss = -torch.clamp(speeds.mean(dim=-1), 0, max_speed)
        loss = loss * weight
        return loss

    return constraint_fn


def make_lane_following_constraint(features, current_lane, next_lane, weight=0.5):
    def constraint_fn(trajectory):
        # TODO
        lanes = np.concatenate([current_lane, next_lane])[..., :2] # , self._get_left_lane(), self._get_right_lane()
        lanes = torch.as_tensor(lanes, device=trajectory.device, dtype=trajectory.dtype)
        trajectory_xy = trajectory.reshape(trajectory.shape[0], 16, 3)[..., :2]

        # Unreduced error is shape (batch_size, N, num_timesteps)
        loss = torch.norm(trajectory_xy[:,None] - lanes[None,:,None], dim=-1)

        # Take minimum over timesteps
        loss = loss.mean(dim=2)
        # loss = loss.min(dim=2).values
        
        # Take minimum over goals
        loss = loss.min(dim=1).values

        return loss * weight
    return constraint_fn


def make_drivable_area_compliance_constraint(features, map_api, ego_states, weight=0.5):
    def constraint_fn(trajectory):
        # TODO
        lanes = torch.as_tensor(lanes, device=trajectory.device, dtype=trajectory.dtype)
        trajectory_xy = trajectory.reshape(trajectory.shape[0], 16, 3)[..., :2]

        corners_route = extract_corners_route(map_api, ego_states)
        import pdb; pdb.set_trace()
        # Unreduced error is shape (batch_size, N, num_timesteps)
        # loss = torch.norm(trajectory_xy[:,None] - lanes[None,:,None], dim=-1)

        # # Take minimum over timesteps
        # loss = loss.mean(dim=2)
        
        # # Take minimum over goals
        # loss = loss.min(dim=1).values

        return loss * weight
    return constraint_fn
