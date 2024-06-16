import logging
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.preprocessing.features.agent_history import AgentHistory
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ConstrainedNonlinearSmoother,
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

logger = logging.getLogger(__name__)


import matplotlib.pyplot as plt
def plot_scene(features, targets, axis):
    axis.set_xlim(-50, 50)
    axis.set_ylim(-50, 50)

    cmap = plt.get_cmap('hsv')
    road_features = features['vector_set_map'].coords['LANE'][0] # .reshape(-1,2)
    for segment_id in range(road_features.shape[0]):
        color = cmap(segment_id / road_features.shape[0])
        axis.scatter(road_features[segment_id,:,0], road_features[segment_id,:,1], color=color)

    # ego_history = features['agent_history'].ego
    # ego_trajectory = targets['trajectory'].data
    # axis.scatter(ego_history[:,0], ego_history[:,1], color='green', marker='x')
    # axis.scatter(ego_trajectory[:,0], ego_trajectory[:,1], color='green', marker='o')

    # agent_history = features['agent_history'].data
    # agent_history_mask = features['agent_history'].mask
    # agent_trajectory = targets['agent_trajectories'].data
    # agent_trajectory_mask = targets['agent_trajectories'].mask
    # axis.scatter(agent_history[agent_history_mask][:,0], agent_history[agent_history_mask][:,1], \
    #     color='blue', marker='x')
    # axis.scatter(agent_trajectory[agent_trajectory_mask][:,0], agent_trajectory[agent_trajectory_mask][:,1], \
    #     color='blue', marker='o')


class AgentHistoryAugmentor(AbstractAugmentor):
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory history that
    satisfies a set of kinematic constraints.

    This involves constrained minimization of the following objective:
    * minimize dist(perturbed_trajectory, ground_truth_trajectory)


    Simple data augmentation that adds Gaussian noise to the ego current position with specified mean and std.
    """

    def __init__(
        self,
        dt: float,
        mean: List[float],
        std: List[float],
        low: List[float],
        high: List[float],
        augment_prob: float,
        use_uniform_noise: bool = False,
        # use_original: bool = False
    ) -> None:
        """
        Initialize the augmentor.
        :param dt: Time interval between trajectory points.
        :param mean: mean of 3-dimensional Gaussian noise to [x, y, yaw]
        :param std: standard deviation of 3-dimenstional Gaussian noise to [x, y, yaw]
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._dt = dt
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)
        self._augment_prob = augment_prob
        # self._use_original = use_original

    def safety_check(self, ego: npt.NDArray[np.float32], agent_history: AgentHistory) -> bool:
        """
        Check if the augmented trajectory violates any safety check (going backwards, collision with other agents).
        :param ego: Perturbed ego feature tensor to be validated.
        :param all_agents: List of agent features to validate against.
        :return: Bool reflecting feature validity.
        """
        # Check if ego goes backward after the perturbation
        if np.diff(ego, axis=0)[-1][0] < 0.0001:
            return False

        # Compute dists from ego to all agent positions
        ego_pose = ego[-1,:2]
        agent_poses = agent_history.data[:,:,:2]
        dists = np.linalg.norm(agent_poses - ego_pose[None,None], axis=2)
        dists[~agent_history.mask] = np.inf
        is_collision = (dists < 2.5).any()
        return not is_collision

    def augment(
        self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets

        # Augment the history to match the distribution shift in close loop rollout
        # for batch_idx in range(len(features['generic_agents'].ego)):
        trajectory_length = len(features['agent_history'].ego) - 1
        _optimizer = ConstrainedNonlinearSmoother(trajectory_length, self._dt)

        ego_trajectory: npt.NDArray[np.float32] = np.copy(features['agent_history'].ego)
        original_ego_state = ego_trajectory.copy()
        ego_trajectory[-1][:3] += self._random_offset_generator.sample()
        ego_x, ego_y, ego_yaw, ego_vx, ego_vy, ego_ax, ego_ay = ego_trajectory.T
        ego_velocity = np.linalg.norm(ego_trajectory[:, 3:5], axis=1)

        # Define the 'earliest history state' as a boundary condition, and reference trajectory
        x_curr = [ego_x[0], ego_y[0], ego_yaw[0], ego_velocity[0]]
        ref_traj = ego_trajectory[:, :3]

        # Set reference and solve
        _optimizer.set_reference_trajectory(x_curr, ref_traj)

        try:
            sol = _optimizer.solve()
        except RuntimeError:
            logger.error("Smoothing failed with status %s! Use G.T. instead" % sol.stats()['return_status'])
            return features, targets

        if not sol.stats()['success']:
            logger.warning("Smoothing failed with status %s! Use G.T. instead" % sol.stats()['return_status'])
            return features, targets

        ego_perturb: npt.NDArray[np.float32] = np.vstack(
            [
                sol.value(_optimizer.position_x),
                sol.value(_optimizer.position_y),
                sol.value(_optimizer.yaw),
                sol.value(_optimizer.speed) * np.cos(sol.value(_optimizer.yaw)),
                sol.value(_optimizer.speed) * np.sin(sol.value(_optimizer.yaw)),
                np.concatenate((sol.value(_optimizer.accel), np.zeros(1))) * np.cos(sol.value(_optimizer.yaw)),
                np.concatenate((sol.value(_optimizer.accel), np.zeros(1))) * np.sin(sol.value(_optimizer.yaw)),
            ]
        )
        ego_perturb = ego_perturb.T

        # fig, axs = plt.subplots(3, figsize=(10,30))
        # plot_scene(features, targets, axs[0])

        # if (np.linalg.norm(targets['trajectory'].data[-1,:2]) > 15) and (np.linalg.norm(features['agent_history'].ego[0,:2]) > 5):
        #     import pdb; pdb.set_trace()

        if self.safety_check(ego_perturb, features['agent_history']):
            features['agent_history'].ego[...] = np.float32(ego_perturb)

            offset = np.float32(ego_perturb[-1,:3]) - original_ego_state[-1,:3]
            # features['agent_history'].ego[:,:3] -= offset

            # Renormalize everything
            tran_offset, head_offset = offset[:2], offset[2]
            rot_matrix = np.array([
                [np.cos(head_offset), -np.sin(head_offset)], 
                [np.sin(head_offset), np.cos(head_offset)]
            ], dtype=np.float32)

            # Agent history
            features['agent_history'].ego[:,:3] -= offset
            features['agent_history'].ego[:,:2] = features['agent_history'].ego[:,:2] @ rot_matrix
            features['agent_history'].ego[:,3:5] = features['agent_history'].ego[:,3:5] @ rot_matrix
            features['agent_history'].ego[:,5:7] = features['agent_history'].ego[:,5:7] @ rot_matrix

            features['agent_history'].data[:,:,:3] -= offset
            features['agent_history'].data[:,:,:2] = features['agent_history'].data[:,:,:2] @ rot_matrix
            features['agent_history'].data[:,:,3:5] = features['agent_history'].data[:,:,3:5] @ rot_matrix

            # plot_scene(features, targets, axs[1])

            # Trajectories
            targets['trajectory'].data[...,:3] -= np.linspace(0,1,targets['trajectory'].data.shape[0])[:,None] * offset[None]
            targets['trajectory'].data[...,:2] = targets['trajectory'].data[...,:2] @ rot_matrix
            targets['agent_trajectories'].data[...,:3] -= offset
            targets['agent_trajectories'].data[...,:2] = targets['agent_trajectories'].data[...,:2] @ rot_matrix
            # Map
            for key in features['vector_set_map'].coords.keys():
                features['vector_set_map'].coords[key][0][:,:,:2] -= tran_offset
                features['vector_set_map'].coords[key][0][:,:,:2] = features['vector_set_map'].coords[key][0][:,:,:2] @ rot_matrix

        # plot_scene(features, targets, axs[2])

        # if (np.linalg.norm(targets['trajectory'].data[-1,:2]) > 15) and (np.linalg.norm(features['agent_history'].ego[0,:2]) > 5):
        #     plt.savefig('/zfsauton2/home/brianyan/nuplan-devkit/test.png')
        #     raise

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agent_history', 'vector_set_map']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ['trajectory', 'agent_trajectories']

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=self._augment_prob,
            param_name=f'{self._augment_prob=}'.partition('=')[0].split('.')[1],
            scaling_direction=ScalingDirection.MAX,
        )

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(List[ParameterToScale], self._random_offset_generator.get_schedulable_attributes())
