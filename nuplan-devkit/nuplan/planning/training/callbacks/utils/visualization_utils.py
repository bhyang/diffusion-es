from enum import Enum
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mpl_colors
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
import torch


from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.agent_history import AgentHistory
from nuplan.planning.training.preprocessing.features.agents_trajectory import AgentTrajectory
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap


class Color(Enum):
    """
    Collection of colors for visualizing map elements, trajectories and agents.
    """

    # BACKGROUND: Tuple[float, float, float] = (0, 0, 0)
    BACKGROUND: Tuple[float, float, float] = (255, 255, 255)
    ROADMAP: Tuple[float, float, float] = (54, 67, 94)
    AGENTS: Tuple[float, float, float] = (113, 100, 222)
    TARGET_AGENT_TRAJECTORY: Tuple[float, float, float] = (193, 180, 232)
    EGO: Tuple[float, float, float] = (82, 86, 92)
    TARGET_TRAJECTORY: Tuple[float, float, float] = (61, 160, 179)
    PREDICTED_TRAJECTORY: Tuple[float, float, float] = (158, 63, 120)
    PREDICTED_AGENT_TRAJECTORY: Tuple[float, float, float] = (180, 120, 180)
    NOISY_TRAJECTORY: Tuple[float, float, float] = (168, 168, 50)
    # BASELINE_PATHS: Tuple[float, float, float] = (210, 220, 220)
    BASELINE_PATHS: Tuple[float, float, float] = (195, 195, 195)
    CURRENT: Tuple[float, float, float] = (150,100,0)
    GOAL: Tuple[float, float, float] = (255, 165, 0)
    SPEED: Tuple[float, float, float] = (255, 255, 255)


class FakeColor:
    value = None


def probability_to_color(prob):
    cmap = cm.get_cmap('winter', 12)
    color = cmap(0.0 + 1.0 * prob)
    fakecolor = FakeColor()
    fakecolor.value = color[0] * 255, color[1] * 255, color[2] * 255
    return fakecolor


def _draw_trajectory(
    image: npt.NDArray[np.uint8],
    trajectory: Trajectory,
    color: Color,
    pixel_size: float,
    radius: int = 5,
    thickness: int = 2,
) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    :param radius: radius of each trajectory pose to be visualized
    :param thickness: thickness of lines connecting trajectory poses to be visualized
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2

    if isinstance(trajectory, Trajectory):
        traj_x, traj_y = trajectory.numpy_position_x, trajectory.numpy_position_y
    else:
        traj_x, traj_y = trajectory[...,0], trajectory[...,1]
    coords_x = (center_x - traj_x / pixel_size).astype(np.int32)
    coords_y = (center_y - traj_y / pixel_size).astype(np.int32)
    idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
    coords_x = coords_x[idxs]
    coords_y = coords_y[idxs]

    for point in zip(coords_y, coords_x):
        cv2.circle(image, point, radius=radius, color=color.value, thickness=-1)

    for point_1, point_2 in zip(zip(coords_y[:-1], coords_x[:-1]), zip(coords_y[1:], coords_x[1:])):
        cv2.line(image, point_1, point_2, color=color.value, thickness=thickness)


def _draw_trajectory_with_grad(
    image: npt.NDArray[np.uint8],
    trajectory: Trajectory,
    color1: Color,
    gradient,
    color2: Color,
    pixel_size: float,
    radius: int = 5,
    thickness: int = 2,
) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    :param radius: radius of each trajectory pose to be visualized
    :param thickness: thickness of lines connecting trajectory poses to be visualized
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2
    gradient *= 100

    if isinstance(trajectory, Trajectory):
        traj_x, traj_y = trajectory.numpy_position_x, trajectory.numpy_position_y
    else:
        traj_x, traj_y = trajectory[..., 0], trajectory[..., 1]
        grad_x, grad_y = gradient[0, ..., 0], gradient[0, ..., 1]

    coords_x = (center_x - traj_x / pixel_size).astype(np.int32)
    coords_y = (center_y - traj_y / pixel_size).astype(np.int32)
    idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
    coords_x = coords_x[idxs]
    coords_y = coords_y[idxs]

    for point in zip(coords_y, coords_x, grad_y, grad_x):
        cv2.circle(image, (point[0], point[1]), radius=radius, color=color1.value, thickness=-1)
        end_point = (int(point[0] + point[2]), int(point[1] + point[3]))
        cv2.arrowedLine(image, (point[0], point[1]), end_point, color=color2.value, thickness=thickness)

    for point_1, point_2 in zip(zip(coords_y[:-1], coords_x[:-1]), zip(coords_y[1:], coords_x[1:])):
        cv2.line(image, point_1, point_2, color=color1.value, thickness=thickness)
        


def _draw_trajectory_gradient(
    image: npt.NDArray[np.uint8],
    trajectory: Trajectory,
    color_1,
    color_2,
    pixel_size: float,
    radius: int = 5,
    thickness: int = 2,
) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    :param radius: radius of each trajectory pose to be visualized
    :param thickness: thickness of lines connecting trajectory poses to be visualized
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2

    if isinstance(trajectory, Trajectory):
        traj_x, traj_y = trajectory.numpy_position_x, trajectory.numpy_position_y
    else:
        traj_x, traj_y = trajectory[...,0], trajectory[...,1]
    coords_x = (center_x - traj_x / pixel_size).astype(np.int32)
    coords_y = (center_y - traj_y / pixel_size).astype(np.int32)
    idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
    coords_x = coords_x[idxs]
    coords_y = coords_y[idxs]

    t = np.linspace(0,1,len(coords_x))[:,None]
    colors = (color_1[None] * t) + (color_2[None] * (1-t))

    for i, point in enumerate(zip(coords_y, coords_x)):
        cv2.circle(image, point, radius=radius, color=colors[i], thickness=-1)

    for i, (point_1, point_2) in enumerate(zip(zip(coords_y[:-1], coords_x[:-1]), zip(coords_y[1:], coords_x[1:]))):
        cv2.line(image, point_1, point_2, color=colors[i], thickness=thickness)


def _draw_trajectories(
    image: npt.NDArray[np.uint8],
    trajectories: AgentTrajectory,
    color: Color,
    pixel_size: float,
    radius: int = 5,
    thickness: int = 2,
) -> None:
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2

    for idx in range(trajectories.data.shape[1]):
        mask = trajectories.mask[:,idx]
        if mask.all():
            trajectory = trajectories.data[:,idx][mask]
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.numpy()
            coords_x = (center_x - trajectory[...,0] / pixel_size).astype(np.int32)
            coords_y = (center_y - trajectory[...,1] / pixel_size).astype(np.int32)
            idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
            coords_x = coords_x[idxs]
            coords_y = coords_y[idxs]

            for point in zip(coords_y, coords_x):
                cv2.circle(image, point, radius=radius, color=color.value, thickness=-1)

            for point_1, point_2 in zip(zip(coords_y[:-1], coords_x[:-1]), zip(coords_y[1:], coords_x[1:])):
                cv2.line(image, point_1, point_2, color=color.value, thickness=thickness)


def _draw_trajectories2(
    image: npt.NDArray[np.uint8],
    trajectories,
    color: Color,
    pixel_size: float,
    radius: int = 5,
    thickness: int = 2,
) -> None:
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2

    for idx in range(trajectories.shape[0]):
        trajectory = trajectories.data[idx]
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.numpy()
        coords_x = (center_x - trajectory[...,0] / pixel_size).astype(np.int32)
        coords_y = (center_y - trajectory[...,1] / pixel_size).astype(np.int32)
        idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
        coords_x = coords_x[idxs]
        coords_y = coords_y[idxs]

        for point in zip(coords_y, coords_x):
            cv2.circle(image, point, radius=radius, color=color.value, thickness=-1)

        for point_1, point_2 in zip(zip(coords_y[:-1], coords_x[:-1]), zip(coords_y[1:], coords_x[1:])):
            cv2.line(image, point_1, point_2, color=color.value, thickness=thickness)


def _draw_point(
    image: npt.NDArray[np.uint8],
    point: Trajectory,
    color: Color,
    pixel_size: float,
    radius: int = 5,
    thickness: int = 2,
) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    :param radius: radius of each trajectory pose to be visualized
    :param thickness: thickness of lines connecting trajectory poses to be visualized
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2

    traj_x, traj_y = point[..., 0], point[..., 1]
    coords_x = (center_x - traj_x / pixel_size).astype(np.int32)
    coords_y = (center_y - traj_y / pixel_size).astype(np.int32)
    idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
    coords_x = coords_x[idxs]
    coords_y = coords_y[idxs]

    for point in zip(coords_y, coords_x):
        cv2.circle(image, point, radius=radius, color=color.value, thickness=-1)


def _draw_text(
    image: npt.NDArray[np.uint8],
    speed: Trajectory,
    color: Color,
    pixel_size: float,
    radius: int = 5,
    thickness: int = 2,
) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    :param radius: radius of each trajectory pose to be visualized
    :param thickness: thickness of lines connecting trajectory poses to be visualized
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2

    traj_x, traj_y = np.array(10), np.array(10)
    coords_x = (center_x - traj_x / pixel_size).astype(np.int32)
    coords_y = (center_y - traj_y / pixel_size).astype(np.int32)
    idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
    coords_x = coords_x[idxs]
    coords_y = coords_y[idxs]

    for point in zip(coords_y, coords_x):
        # cv2.circle(image, point, radius=radius, color=color.value, thickness=-1)
        cv2.putText(image, str(speed), point, cv2.FONT_HERSHEY_SIMPLEX, radius, color, thickness)


def _create_map_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # Extract coordinates from vector map feature
    vector_coords = vector_map.get_lane_coords(0)  # Get first sample in batch

    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    # coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    # coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)

    for i in range(coords.shape[0]):
        mask = vector_map.availabilities['LANE'][0][i]
        # mask = (coords[i] < radius) * (coords[i] > -radius)
        # mask = mask.all(axis=-1)

        masked_coords = shifted_index_coords[i][mask]

        # Paint the grid
        cv2.polylines(
            map_raster,
            masked_coords[None],
            isClosed=False,
            color=color,
            thickness=thickness,
            shift=bit_shift,
            lineType=cv2.LINE_AA,
        )

    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster


def _create_agents_raster(
    agents: Union[Agents, GenericAgents], radius: float, size: int, bit_shift: int, pixel_size: float, color: int = 1
) -> npt.NDArray[np.uint8]:
    """
    Create agents raster layer to be visualized.

    :param agents: agents feature object (either Agents or GenericAgents).
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :return: Instantiated grid.
    """
    # Instantiate grid
    agents_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Extract array data from features
    agents_array: npt.NDArray[np.float32] = np.asarray(
        agents.get_present_agents_in_sample(0)
    )  # Get first sample in batch
    agents_corners: npt.NDArray[np.float32] = np.asarray(
        agents.get_agent_corners_in_sample(0)
    )  # Get first sample in batch

    if len(agents_array) == 0:
        return agents_raster

    # Align coordinates to map, transform them to ego's reference and clip them based on radius
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    transform = Rotation.from_euler('z', agents_array[:, 2], degrees=False).as_matrix().astype(np.float32)
    transform[:, :2, 2] = agents_array[:, :2]
    points = (map_ortho_align @ transform @ agents_corners.transpose([0, 2, 1])).transpose([0, 2, 1])[..., :2]
    points[..., 0] = np.clip(points[..., 0], -radius, radius)
    points[..., 1] = np.clip(points[..., 1], -radius, radius)

    # Convert coordinates to grid indices
    index_points = (radius + points) / pixel_size
    shifted_index_points = (index_points * 2**bit_shift).astype(np.int64)

    # Paint the grid
    for i, box in enumerate(shifted_index_points):
        cv2.fillPoly(agents_raster, box[None], color=color, shift=bit_shift, lineType=cv2.LINE_AA)

    # Flip grid upside down
    agents_raster = np.flipud(agents_raster)

    return agents_raster


def _create_agents_raster_with_weights(
    agents: Union[Agents, GenericAgents], radius: float, size: int, bit_shift: int, pixel_size: float, color: int = 1, agent_weights = None
) -> npt.NDArray[np.uint8]:
    """
    Create agents raster layer to be visualized.

    :param agents: agents feature object (either Agents or GenericAgents).
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :return: Instantiated grid.
    """
    # Instantiate grid
    agents_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Extract array data from features
    agents_array: npt.NDArray[np.float32] = np.asarray(
        agents.get_present_agents_in_sample(0)
    )  # Get first sample in batch
    agents_corners: npt.NDArray[np.float32] = np.asarray(
        agents.get_agent_corners_in_sample(0)
    )  # Get first sample in batch

    if len(agents_array) == 0:
        return agents_raster

    # Align coordinates to map, transform them to ego's reference and clip them based on radius
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    transform = Rotation.from_euler('z', agents_array[:, 2], degrees=False).as_matrix().astype(np.float32)
    transform[:, :2, 2] = agents_array[:, :2]
    points = (map_ortho_align @ transform @ agents_corners.transpose([0, 2, 1])).transpose([0, 2, 1])[..., :2]
    points[..., 0] = np.clip(points[..., 0], -radius, radius)
    points[..., 1] = np.clip(points[..., 1], -radius, radius)

    # Convert coordinates to grid indices
    index_points = (radius + points) / pixel_size
    shifted_index_points = (index_points * 2**bit_shift).astype(np.int64)

    # Paint the grid
    for i, box in enumerate(shifted_index_points):
        if i >= agent_weights.shape[0]:
            break
        color = int(agent_weights[i] * 255)
        cv2.fillPoly(agents_raster, box[None], color=color, shift=bit_shift, lineType=cv2.LINE_AA)

    # Flip grid upside down
    agents_raster = np.flipud(agents_raster)

    return agents_raster


def _create_new_agents_raster(
    agent_history: AgentHistory, radius: float, size: int, bit_shift: int, pixel_size: float, color: int = 1
) -> npt.NDArray[np.uint8]:
    """
    Create agents raster layer to be visualized.

    :param agents: agents feature object (either Agents or GenericAgents).
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :return: Instantiated grid.
    """
    # Instantiate grid
    agents_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Extract array data from features
    # agents_array: npt.NDArray[np.float32] = np.asarray(
    #     agents.get_present_agents_in_sample(0)
    # )  # Get first sample in batch
    # agents_corners: npt.NDArray[np.float32] = np.asarray(
    #     agents.get_agent_corners_in_sample(0)
    # )  # Get first sample in batch

    agent_features = agent_history.data
    agent_masks = agent_history.mask

    if len(agent_features.shape) == 4:
        agent_features = agent_features[0]
        agent_masks = agent_masks[0]

    agents_array = np.asarray(agent_features[-1][agent_masks[-1]])
    widths, lengths = agents_array[:,5], agents_array[:,6]

    half_widths = widths / 2.0
    half_lengths = lengths / 2.0

    agent_corners = np.array(
        [
            [
                [half_length, half_width, 1.0],
                [-half_length, half_width, 1.0],
                [-half_length, -half_width, 1.0],
                [half_length, -half_width, 1.0],
            ]
            for half_width, half_length in zip(half_widths, half_lengths)
        ]
    )

    if len(agents_array) == 0:
        return agents_raster

    # Align coordinates to map, transform them to ego's reference and clip them based on radius
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    transform = Rotation.from_euler('z', agents_array[:, 2], degrees=False).as_matrix().astype(np.float32)
    transform[:, :2, 2] = agents_array[:, :2]
    points = (map_ortho_align @ transform @ agent_corners.transpose([0, 2, 1])).transpose([0, 2, 1])[..., :2]
    points[..., 0] = np.clip(points[..., 0], -radius, radius)
    points[..., 1] = np.clip(points[..., 1], -radius, radius)

    # Convert coordinates to grid indices
    index_points = (radius + points) / pixel_size
    shifted_index_points = (index_points * 2**bit_shift).astype(np.int64)

    # Paint the grid
    for box in shifted_index_points:
        cv2.fillPoly(agents_raster, box[None], color=color, shift=bit_shift, lineType=cv2.LINE_AA)

    # Flip grid upside down
    agents_raster = np.flipud(agents_raster)

    return agents_raster


def _create_ego_raster(
    vehicle_parameters: VehicleParameters, pixel_size: float, size: int, color: int = 1, thickness: int = -1
) -> npt.NDArray[np.uint8]:
    """
    Create ego raster layer to be visualized.

    :param vehicle_parameters: Ego vehicle parameters dataclass object.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Box line thickness (-1 means fill).
    :return: Instantiated grid.
    """
    # Instantiate grid
    ego_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Extract ego vehicle dimensions
    ego_width = vehicle_parameters.width
    ego_front_length = vehicle_parameters.front_length
    ego_rear_length = vehicle_parameters.rear_length

    # Convert coordinates to grid indices
    ego_width_pixels = int(ego_width / pixel_size)
    ego_front_length_pixels = int(ego_front_length / pixel_size)
    ego_rear_length_pixels = int(ego_rear_length / pixel_size)
    map_x_center = int(ego_raster.shape[1] * 0.5)
    map_y_center = int(ego_raster.shape[0] * 0.5)
    ego_top_left = (map_x_center - ego_width_pixels // 2, map_y_center - ego_front_length_pixels)
    ego_bottom_right = (map_x_center + ego_width_pixels // 2, map_y_center + ego_rear_length_pixels)

    # Paint the grid
    cv2.rectangle(ego_raster, ego_top_left, ego_bottom_right, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    return ego_raster


def get_raster_from_vector_map_with_agents(
    vector_map: Union[VectorMap, VectorSetMap],
    agents: Union[Agents, GenericAgents],
    target_trajectory: Optional[Trajectory] = None,
    predicted_trajectory = None,
    probabilities=None,
    pixel_size: float = 0.5,
    bit_shift: int = 12,
    radius: float = 50.0,
    vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    noisy_trajectory: Optional[Trajectory] = None
) -> npt.NDArray[np.uint8]:
    """
    Create rasterized image from vector map and list of agents.

    :param vector_map: Vector map/vector set map feature to visualize.
    :param agents: Agents/GenericAgents feature to visualize.
    :param target_trajectory: Target trajectory to visualize.
    :param predicted_trajectory: Predicted trajectory to visualize.
    :param pixel_size: [m] Size of a pixel.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param radius: [m] Radius of raster.
    :param vehicle_parameters: Parameters of the ego vehicle.
    :return: Composed rasterized image.
    """
    # Raster size
    size = int(2 * radius / pixel_size)

    # Create map layers
    map_raster = _create_map_raster(vector_map, radius, size, bit_shift, pixel_size)
    agents_raster = _create_agents_raster(agents, radius, size, bit_shift, pixel_size)
    ego_raster = _create_ego_raster(vehicle_parameters, pixel_size, size)

    # Compose and paint image
    image: npt.NDArray[np.uint8] = np.full((size, size, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[map_raster.nonzero()] = Color.BASELINE_PATHS.value
    image[agents_raster.nonzero()] = Color.AGENTS.value
    image[ego_raster.nonzero()] = Color.EGO.value

    # Draw predicted and target trajectories
    if predicted_trajectory is not None:
        if probabilities is not None:
            sorted_idxs = probabilities.argsort().tolist()
            for idx in sorted_idxs:
                trajectory = predicted_trajectory[idx]
                prob = probabilities[idx]
                color = probability_to_color(prob.item())
                _draw_trajectory(image, trajectory, color, pixel_size)
        else:
            _draw_trajectory(image, predicted_trajectory, Color.PREDICTED_TRAJECTORY, pixel_size)
    # if noisy_trajectory is not None:
    #     _draw_trajectory(image, noisy_trajectory, Color.NOISY_TRAJECTORY, pixel_size)
    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, Color.TARGET_TRAJECTORY, pixel_size)

    return image


def get_raster_from_vector_map_with_new_agents(
    vector_map: Union[VectorMap, VectorSetMap],
    agent_history: AgentHistory,
    ego_trajectory: Trajectory = None,
    agent_trajectories: AgentTrajectory = None,
    pred_ego_trajectory: Trajectory = None,
    pred_agent_trajectories: AgentTrajectory = None,
    goal = None,
    pixel_size: float = 0.5,
    bit_shift: int = 12,
    radius: float = 50.0,
    vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    noisy_trajectory: Optional[Trajectory] = None,
    multimodal_trajectories = None,
    trajectory = None
) -> npt.NDArray[np.uint8]:
    # Raster size
    size = int(2 * radius / pixel_size)

    # Create map layers
    map_raster = _create_map_raster(vector_map, radius, size, bit_shift, pixel_size)
    agents_raster = _create_new_agents_raster(agent_history, radius, size, bit_shift, pixel_size)
    ego_raster = _create_ego_raster(vehicle_parameters, pixel_size, size)

    # Compose and paint image
    image: npt.NDArray[np.uint8] = np.full((size, size, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[map_raster.nonzero()] = Color.BASELINE_PATHS.value
    image[agents_raster.nonzero()] = Color.AGENTS.value
    image[ego_raster.nonzero()] = Color.EGO.value

    if agent_trajectories is not None:
        _draw_trajectories(image, agent_trajectories, Color.TARGET_AGENT_TRAJECTORY, pixel_size)
    if ego_trajectory is not None:
        _draw_trajectory(image, ego_trajectory, Color.TARGET_TRAJECTORY, pixel_size)
    if pred_agent_trajectories:
        _draw_trajectories(image, pred_agent_trajectories, Color.PREDICTED_AGENT_TRAJECTORY, pixel_size)
    if pred_ego_trajectory:
        _draw_trajectory(image, pred_ego_trajectory, Color.PREDICTED_TRAJECTORY, pixel_size)
    if multimodal_trajectories is not None:
        _draw_trajectories2(image, multimodal_trajectories, Color.PREDICTED_AGENT_TRAJECTORY, pixel_size)
    if goal is not None:
        fakecolor = FakeColor()
        fakecolor.value = Color.GOAL.value[0], Color.GOAL.value[1], Color.GOAL.value[2]
        _draw_point(image, goal, fakecolor, pixel_size)
    if trajectory is not None:
        color_2, color_1 = np.array([0,255,0]), np.array([0,0,255])
        _draw_trajectory_gradient(image, trajectory, color_1, color_2, pixel_size)

    return image


def get_generic_raster_from_vector_map(
    vector_map: Union[VectorMap, VectorSetMap],
    agents: Union[Agents, GenericAgents],
    trajectory = None,
    trajectories = None,
    probabilities = None,
    goal = None,
    current = None,
    connector= None,
    next = None,
    agent_weights = None,
    pixel_size: float = 0.5,
    bit_shift: int = 12,
    radius: float = 50.0,
    vehicle_parameters: VehicleParameters = get_pacifica_parameters()
) -> npt.NDArray[np.uint8]:
    """
    Create rasterized image from vector map and list of agents.

    :param vector_map: Vector map/vector set map feature to visualize.
    :param agents: Agents/GenericAgents feature to visualize.
    :param target_trajectory: Target trajectory to visualize.
    :param predicted_trajectory: Predicted trajectory to visualize.
    :param pixel_size: [m] Size of a pixel.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param radius: [m] Radius of raster.
    :param vehicle_parameters: Parameters of the ego vehicle.
    :return: Composed rasterized image.
    """
    # Raster size
    size = int(2 * radius / pixel_size)

    # Create map layers
    map_raster = _create_map_raster(vector_map, radius, size, bit_shift, pixel_size)
    if isinstance(agents, AgentHistory):
        agents_raster = _create_new_agents_raster(agents, radius, size, bit_shift, pixel_size)
    else:
        if agent_weights is None:
            agents_raster = _create_agents_raster(agents, radius, size, bit_shift, pixel_size)
        else:
            agents_raster = _create_agents_raster_with_weights(agents, radius, size, bit_shift, pixel_size, agent_weights=agent_weights)
    ego_raster = _create_ego_raster(vehicle_parameters, pixel_size, size)

    # Compose and paint image
    image: npt.NDArray[np.uint8] = np.full((size, size, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[map_raster.nonzero()] = Color.BASELINE_PATHS.value
    if agent_weights is None:
        image[agents_raster.nonzero()] = Color.AGENTS.value
    else:
        image[agents_raster.nonzero()] = agents_raster[agents_raster.nonzero()][:,None]
    image[ego_raster.nonzero()] = Color.EGO.value

    if trajectories is not None:
        if probabilities is not None:
            sorted_idx = np.argsort(probabilities)
            for idx in sorted_idx:
                color = probability_to_color(probabilities[idx])
                _draw_trajectory(image, trajectories[idx], color, pixel_size)
        else:
            color_2, color_1 = np.array([0,255,0]), np.array([0,0,255])
            for traj in trajectories:
                _draw_trajectory_gradient(image, traj, color_1, color_2, pixel_size)

    if trajectory is not None:
        color_2, color_1 = np.array([0,255,0]), np.array([0,0,255])
        _draw_trajectory_gradient(image, trajectory, color_1, color_2, pixel_size)

    # if trajectories is not None:
    #     color_2, color_1 = np.array([0,255,0]), np.array([0,0,255])
    #     for traj in trajectories:
    #         _draw_trajectory_gradient(image, traj, color_1, color_2, pixel_size)

    if goal is not None:
        fakecolor = FakeColor()
        fakecolor.value = Color.GOAL.value[0], Color.GOAL.value[1], Color.GOAL.value[2]
        _draw_point(image, goal, fakecolor, pixel_size)
    # _draw_trajectory(image, trajectory, Color.PREDICTED_TRAJECTORY, pixel_size)

    if current is not None:
        fakecolor = FakeColor()
        fakecolor.value = Color.CURRENT.value[0], Color.CURRENT.value[1], Color.CURRENT.value[2]
        _draw_point(image, current, fakecolor, pixel_size)

    # if connector is not None:
    #     fakecolor = FakeColor()
    #     fakecolor.value = Color.CONNECTOR.value[0], Color.CONNECTOR.value[1], Color.CONNECTOR.value[2]
    #     _draw_point(image, connector, fakecolor, pixel_size)

    # if next is not None:
    #     fakecolor = FakeColor()
    #     fakecolor.value = Color.NEXT.value[0], Color.NEXT.value[1], Color.NEXT.value[2]
    #     _draw_point(image, next, fakecolor, pixel_size)

    return image


def get_raster_with_trajectories_as_rgb(
    raster: Raster,
    target_trajectory: Optional[Trajectory] = None,
    predicted_trajectory: Optional[Trajectory] = None,
    pixel_size: float = 0.5,
) -> npt.NDArray[np.uint8]:
    """
    Create an RGB images of the raster layers overlayed with predicted / ground truth trajectories

    :param raster: input raster to visualize
    :param target_trajectory: target (ground truth) trajectory to visualize
    :param predicted_trajectory: predicted trajectory to visualize
    :param background_color: desired color of the image's background
    :param roadmap_color: desired color of the map raster layer
    :param agents_color: desired color of the agents raster layer
    :param ego_color: desired color of the ego raster layer
    :param target_trajectory_color: desired color of the target trajectory
    :param predicted_trajectory_color: desired color of the predicted trajectory
    :param pixel_size: [m] size of pixel in meters
    :return: constructed RGB image
    """
    grid_shape = (raster.height, raster.width)

    # Compose and paint image
    image: npt.NDArray[np.uint8] = np.full((*grid_shape, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[raster.roadmap_layer[0] > 0] = Color.ROADMAP.value
    image[raster.baseline_paths_layer[0] > 0] = Color.BASELINE_PATHS.value
    image[raster.agents_layer.squeeze() > 0] = Color.AGENTS.value  # squeeze to shape of W*H only
    image[raster.ego_layer.squeeze() > 0] = Color.EGO.value

    # Draw predicted and target trajectories
    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, Color.TARGET_TRAJECTORY, pixel_size, 2, 1)
    if predicted_trajectory is not None:
        _draw_trajectory(image, predicted_trajectory, Color.PREDICTED_TRAJECTORY, pixel_size, 2, 1)
        
    return image


def visualize(
    vector_map: Union[VectorMap, VectorSetMap],
    agent_history: AgentHistory,
    pixel_size: float = 0.5,
    bit_shift: int = 12,
    radius: float = 50.0,
    vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    invert_background: bool = False,
    trajectory_only: bool = False,
    **plots,
) -> npt.NDArray[np.uint8]:
    """
    All-in-one visualization function for nuPlan scenes + trajectories
    Renders the map and draws trajectories

    Any of the kwargs for _plot_trajectories can be used by prepending a prefix like so:

        trajectory1 = Trajectory(...)
        trajectory2 = np.asarray(trajectory1.data)
        trajectory3 = np.stack([trajectory2 for _ in range(3)], axis=0)

        visualize(
            vector_map, agent_history,
            
            plot1=trajectory1,
            plot1_color='red',
            
            plot2=trajectory2,
            plot2_color='blue',
            plot2_radius=10,
            plot2_use_lines=False,

            plot3=trajectory3,
            plot3_color='green',
            plot3_use_lines=True,
            plot3_thickness=2,
            plot3_radius=5,
            plot3_grad=...,
        )

    If multiple trajectories are passed in, will render each one
    Supports Trajectory objects, tensors, and numpy arrays
    Can specify colors using matplotlib color names
    """

    # Raster size
    size = int(2 * radius / pixel_size)
    
    image: npt.NDArray[np.uint8] = np.full((size, size, 3), Color.BACKGROUND.value, dtype=np.uint8)

    # Create map layers
    if not trajectory_only:
        map_raster = _create_map_raster(vector_map, radius, size, bit_shift, pixel_size)
        agents_raster = _create_new_agents_raster(agent_history, radius, size, bit_shift, pixel_size)
        ego_raster = _create_ego_raster(vehicle_parameters, pixel_size, size)

        # Compose and paint image
        image[map_raster.nonzero()] = Color.BASELINE_PATHS.value
        image[agents_raster.nonzero()] = Color.AGENTS.value
        image[ego_raster.nonzero()] = Color.EGO.value

    # Plotting lines / points
    # NOTE: plot names cannot have underscores or this will break
    plot_keys = [key for key in plots.keys() if '_' not in key]
    for plot_key in plot_keys:
        data = plots[plot_key]
        data = preprocess_plot_data(data)

        prefix = f'{plot_key}_'
        plot_kwargs = {key.removeprefix(prefix): plots[key] for key in plots.keys() if key.startswith(prefix)}
        _plot_points(image, data, pixel_size=pixel_size, **plot_kwargs)

    return image


def preprocess_plot_data(data):
    """
    Takes one or multiple trajectories and converts to standardized numpy format
    Input: (A, B, *) or (B, *) or Trajectory object
    Output: (A, B, 2) or (1, B, 2)
    """

    if isinstance(data, Trajectory):
        data = data.data
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        raise NotImplementedError

    num_dims = len(data.shape)
    if num_dims == 1:
        data = data[None,None]
    elif num_dims == 2:
        data = data[None]
    assert len(data.shape) == 3, f'Got invalid # dims for plotting, (expected 3, got {num_dims})'

    data_xy = data[...,:2]
    return data_xy


def _plot_points(
    image: npt.NDArray[np.uint8],
    points,
    pixel_size: float,

    color: str = None,        # Color of points/lines. Uses matplotlib naming conventions for colors

    radius: int = 5,            # Radius of points
    thickness: int = 2,         # Thickness of lines

    use_lines: bool = True,     # Whether to draw lines between points
    use_points: bool = True,    # Whether to draw points

    timestep_gradient = False,  # Coloring based on timestep

    c = None,                   # Coloring based on arbitrary scalars
    cmap=None,                  # Colormap for scalars
    cmin = None,                # Manually set min for cmap
    cmax = None,                # Manually set max for cmap
) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    :param radius: radius of each trajectory pose to be visualized
    :param thickness: thickness of lines connecting trajectory poses to be visualized
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2

    # Convert color names to arrays using matplotlib
    # If provided, also process color gradient args
    if color is not None:
        color_arr = mpl_colors.to_rgb(color)
        color_arr = (np.array(color_arr) * 255).astype(int)
        colors = color_arr[None].repeat(points.shape[0],0)
        indices = range(points.shape[0])
    elif c is not None:
        cmin = cmin if cmin is not None else c.min()
        cmax = cmax if cmax is not None else c.max()
        scores = (c - cmin) / (cmax - cmin)
        colormap = plt.cm.get_cmap(cmap)
        colors = [colormap(score)[:3] for score in scores]
        colors = (np.array(colors) * 255).astype(int)
        indices = np.argsort(scores)[::-1]
    elif timestep_gradient:
        scores = np.linspace(0,1,points.shape[1])
        colormap = get_cmap(cmap)
        colors = [colormap(score)[:3] for score in scores]
        colors = (np.array(colors) * 255).astype(int)
        indices = range(points.shape[0])
    else:
        raise Exception('Must provide color when plotting')
    

    for line_idx in indices:    
        traj_x, traj_y = points[line_idx,:,0], points[line_idx,:,1]

        coords_x = (center_x - traj_x / pixel_size).astype(np.int32)
        coords_y = (center_y - traj_y / pixel_size).astype(np.int32)
        idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
        coords_x = coords_x[idxs]
        coords_y = coords_y[idxs]

        if use_points:
            for i, point in enumerate(zip(coords_y, coords_x)):
                color = colors[line_idx].tolist() if not timestep_gradient else colors[i].tolist()
                cv2.circle(image, point, radius=radius, color=color, thickness=-1)

        if use_lines:
            for i, (point_1, point_2) in enumerate(zip(zip(coords_y[:-1], coords_x[:-1]), zip(coords_y[1:], coords_x[1:]))):
                color = colors[line_idx].tolist() if not timestep_gradient else colors[i].tolist()
                cv2.line(image, point_1, point_2, color=color, thickness=thickness)


def get_cmap(cmap):
    try:
        return plt.cm.get_cmap(cmap)
    except:
        from matplotlib.colors import LinearSegmentedColormap
        color1, color2 = cmap.split('->')
        cmap = LinearSegmentedColormap.from_list('test', [color1, color2])
        return cmap