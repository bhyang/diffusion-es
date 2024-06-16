import io

import numpy as np
import cv2
import torch
from scipy.interpolate import interp1d


def interpolate_trajectory(trajectories, npoints):
    old_xs = np.arange(trajectories.shape[1])
    new_xs = np.linspace(0, trajectories.shape[1]-1, npoints)
    trajectories = interp1d(old_xs, trajectories, axis=1)(new_xs)
    return trajectories


def convert_to_local(ego_state, points):
    points = points.copy()

    ref = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])
    rot = np.array([
        [np.cos(ref[2]), -np.sin(ref[2])],
        [np.sin(ref[2]),  np.cos(ref[2])]
    ])

    original_shape = points.shape

    points_dim = points.shape[-1]
    assert points_dim in (2,3), f'Received invalid feature dimension in convert_to_local, got {points_dim} expected 2 or 3'
    points = points.reshape(-1,points_dim)
    points -= ref[:points_dim]
    points[:,:2] = points[:,:2] @ rot

    points = points.reshape(original_shape)

    return points


def convert_to_global(ego_state, points):
    points = points.copy()

    ref = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])
    rot = np.array([
        [np.cos(ref[2]), -np.sin(ref[2])],
        [np.sin(ref[2]),  np.cos(ref[2])]
    ])

    original_shape = points.shape

    points_dim = points.shape[-1]
    assert points_dim in (2,3), f'Received invalid feature dimension in convert_to_local, got {points_dim} expected 2 or 3'
    points = points.reshape(-1,points_dim)
    points[:,:2] = points[:,:2] @ rot.T
    points += ref[:points_dim]

    points = points.reshape(original_shape)

    return points


def heading_from_xy(xy):
    original_shape = xy.shape
    xy = xy.reshape(-1,2)

    deltas = torch.diff(xy, dim=0)
    heading = torch.arctan2(deltas[:,0], deltas[:,1])

    # Hack
    heading = torch.cat([heading[:1], heading], dim=0)

    heading = heading.reshape(*original_shape[:-1])
    return heading


def fig_to_numpy(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = img[:,:,::-1]
    return img
