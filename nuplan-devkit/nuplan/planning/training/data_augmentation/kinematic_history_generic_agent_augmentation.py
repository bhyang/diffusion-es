import logging
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from tqdm import tqdm

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ConstrainedNonlinearSmoother,
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

logger = logging.getLogger(__name__)


def plot_features_and_targets(features, targets, ax):
    ego_history = features['generic_agents'].ego[0]
    agents_history = features['generic_agents'].agents['VEHICLE'][0]
    map_features = features['vector_set_map'].coords['LANE'][0]
    map_masks = features['vector_set_map'].availabilities['LANE'][0]
    ego_trajectory = targets['trajectory'].data

    # Plot road
    for lane_id in range(map_features.shape[1]):
        masked = map_features[lane_id][map_masks[lane_id]]
        if masked.shape[0] > 0:
            ax.plot(masked[:,0], masked[:,1], color='black')

    # Plot other agents
    for agent_id in range(agents_history.shape[1]):
        ax.scatter(agents_history[:,agent_id,0], agents_history[:,agent_id,1], color='red', marker='o')

    # Plot ego
    colors = np.array([
        np.linspace(0,0,21),
        np.linspace(0,1,21),
        np.linspace(1,0,21)
    ]).T
    for t in range(5):
        if t < 4:
            ax.plot(ego_history[t:t+1,0], ego_history[t:t+1,1], color=colors[t])
        theta = ego_history[t,2]
        ax.arrow(ego_history[t,0], ego_history[t,1], 
                 np.cos(theta), np.sin(theta), 
                 color=colors[t],
                 width=0.5)
    for t in range(16):
        if t < 15:
            ax.plot(ego_trajectory[t:t+1,0], ego_trajectory[t:t+1,1], color=colors[t+5])
        theta = ego_trajectory[t,2]
        ax.arrow(ego_trajectory[t,0], ego_trajectory[t,1], 
                 np.cos(theta), np.sin(theta),
                 color=colors[t+5],
                 width=0.5)
    # ax.plot(ego_history[:,0], ego_history[:,1], color='purple', marker='o')
    # ax.scatter(ego_trajectory[:,0], ego_trajectory[:,1], c=colors[5:])

    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)


def plot_augmented_positions(features, targets, augmented_positions, ax):
    ego_history = features['generic_agents'].ego[0]
    agents_history = features['generic_agents'].agents['VEHICLE'][0]
    map_features = features['vector_set_map'].coords['LANE'][0]
    map_masks = features['vector_set_map'].availabilities['LANE'][0]
    ego_trajectory = targets['trajectory'].data

    # Plot road
    for lane_id in range(map_features.shape[1]):
        masked = map_features[lane_id][map_masks[lane_id]]
        if masked.shape[0] > 0:
            ax.plot(masked[:,0], masked[:,1], color='black')

    # Plot other agents
    for agent_id in range(agents_history.shape[1]):
        ax.scatter(agents_history[:,agent_id,0], agents_history[:,agent_id,1], color='red', marker='o')

    # Plot ego
    for aug_pos in augmented_positions:
        theta = aug_pos[2]
        ax.arrow(aug_pos[0], aug_pos[1], 
                 np.cos(theta), np.sin(theta),
                 color='gray',
                 width=0.5)

    colors = np.array([
        np.linspace(0,0,21),
        np.linspace(0,1,21),
        np.linspace(1,0,21)
    ]).T
    for t in range(5):
        if t < 4:
            ax.plot(ego_history[t:t+1,0], ego_history[t:t+1,1], color=colors[t])
        theta = ego_history[t,2]
        ax.arrow(ego_history[t,0], ego_history[t,1], 
                 np.cos(theta), np.sin(theta), 
                 color=colors[t],
                 width=0.5)
    for t in range(16):
        if t < 15:
            ax.plot(ego_trajectory[t:t+1,0], ego_trajectory[t:t+1,1], color=colors[t+5])
        theta = ego_trajectory[t,2]
        ax.arrow(ego_trajectory[t,0], ego_trajectory[t,1], 
                 np.cos(theta), np.sin(theta),
                 color=colors[t+5],
                 width=0.5)

    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)


def fix_trajectory_yaws(targets):
    trajectory = targets['trajectory']
    deltas = np.diff(trajectory.data, axis=0, prepend=0)
    yaws = np.arctan2(deltas[:,1], deltas[:,0])
    trajectory.data[:,2] = yaws


class KinematicHistoryGenericAgentAugmentor(AbstractAugmentor):
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
        use_original: bool = False
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
        self._use_original = use_original

    def safety_check(self, ego: npt.NDArray[np.float32], all_agents: List[npt.NDArray[np.float32]], target_traj) -> bool:
        """
        Check if the augmented trajectory violates any safety check (going backwards, collision with other agents).
        :param ego: Perturbed ego feature tensor to be validated.
        :param all_agents: List of agent features to validate against.
        :return: Bool reflecting feature validity.
        """
        # Check if ego goes backward after the perturbation
        if np.diff(ego, axis=0)[-1][0] < 0.0001:
            return False

        # Check if there is collision between ego and other agents
        for agents in all_agents:
            dist_to_the_closest_agent = np.min(np.linalg.norm(np.array(agents)[:, :, :2] - ego[-1, :2], axis=1))
            if dist_to_the_closest_agent < 2.5:
                return False

        # Check if the endpoint is too close
        if np.linalg.norm(target_traj[-1,:2]) < 5:
            return False

        return True

    def augment(
        self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        # if np.random.rand() >= self._augment_prob:
        #     return features, targets

        num_augs = 100
        # fig, axs = plt.subplots(num_augs+1, figsize=(10,10*(num_augs+1)))
        # plot_features_and_targets(features, targets, axs[0])

        from copy import deepcopy
        orig_features, orig_targets = deepcopy(features), deepcopy(targets)
        augmented_positions = []

        for i in tqdm(range(num_augs)):
            features, targets = deepcopy(orig_features), deepcopy(orig_targets)

            # Augment the history to match the distribution shift in close loop rollout
            for batch_idx in range(len(features['generic_agents'].ego)):
                trajectory_length = len(features['generic_agents'].ego[batch_idx]) - 1
                _optimizer = ConstrainedNonlinearSmoother(trajectory_length, self._dt)

                ego_trajectory: npt.NDArray[np.float32] = np.copy(features['generic_agents'].ego[batch_idx])
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
                agents: List[npt.NDArray[np.float32]] = [
                    agent_features[batch_idx] for agent_features in features['generic_agents'].agents.values()
                ]

                # fig, axs = plt.subplots(2, figsize=(10,20))
                # plot_features_and_targets(features, targets, axs[0])

                if self.safety_check(ego_perturb, agents, targets['trajectory'].data):
                    if self._use_original:
                        features["generic_agents"].ego[batch_idx] = np.float32(ego_perturb)
                    else:
                        # augmented_positions.append(ego_trajectory[-1][:3])
                        augmented_positions.append(ego_perturb[-1,:3])
                        offset = np.float32(ego_perturb[-1,:3]) - original_ego_state[-1,:3]
                        # features['generic_agents'].ego[batch_idx][:,:3] += offset
                        features["generic_agents"].ego[batch_idx] = np.float32(ego_perturb)
                        targets['trajectory'].data += offset[None] * np.logspace(0,-3,16)[:,None] # np.linspace(1,0,16)[:,None]

                        # Renormalize everything
                        tran_offset, head_offset = offset[:2], offset[2]
                        rot_matrix = np.array([[np.cos(head_offset), -np.sin(head_offset)], [np.sin(head_offset), np.cos(head_offset)]])

                        # Ego
                        # augmented_positions.append(
                        #     deepcopy(features["generic_agents"].ego[batch_idx]))
        
                        features['generic_agents'].ego[batch_idx][:,:3] -= offset
                        features['generic_agents'].ego[batch_idx][:,:2] = features['generic_agents'].ego[batch_idx][:,:2] @ rot_matrix
                        targets['trajectory'].data[...,:3] -= offset
                        targets['trajectory'].data[:,:2] = targets['trajectory'].data[:,:2] @ rot_matrix
                        
                        fix_trajectory_yaws(targets)

                        # Vehicles
                        assert len(features['generic_agents'].agents.keys()) == 1, 'Augmentation only suppports VEHICLE objects currently'
                        features['generic_agents'].agents['VEHICLE'][batch_idx][:,:,:3] -= offset
                        features['generic_agents'].agents['VEHICLE'][batch_idx][:,:,:2] = features['generic_agents'].agents['VEHICLE'][batch_idx][:,:,:2] @ rot_matrix

                        # Map
                        for key in features['vector_set_map'].coords.keys():
                            features['vector_set_map'].coords[key][batch_idx][:,:,:2] -= tran_offset
                            features['vector_set_map'].coords[key][batch_idx][:,:,:2] = features['vector_set_map'].coords[key][batch_idx][:,:,:2] @ rot_matrix

                        # # plot_features_and_targets(features, targets, axs[i+1])
                        # augmented_positions.append(features["generic_agents"].ego[batch_idx])

        # Multiple aug viz
        if len(augmented_positions) > 0:
            augmented_positions = np.stack(augmented_positions)
            # augmented_positions = augmented_positions[:,-1,:3]
            fig, ax = plt.subplots(figsize=(10,10))
            plot_augmented_positions(orig_features, orig_targets, augmented_positions, ax)

        plt.savefig('/zfsauton2/home/brianyan/nuplan-devkit/test.png')
        key = input('waiting... ')
        if key == 'q':
            raise

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['generic_agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

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
