import logging
from concurrent.futures import Future
from typing import List, Optional
import pathlib

import numpy as np

from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

logger = logging.getLogger(__name__)


def run_metric_engine(
    metric_engine: MetricsEngine, 
    scenario: AbstractScenario, 
    planner_name: str, 
    history: SimulationHistory,
    planner: AbstractPlanner = None,
    save_samples_to_disk: bool = False,
    save_samples_path: str = None
) -> None:
    """
    Run the metric engine.
    """
    logger.debug("Starting metrics computation...")
    metric_files, other_files = metric_engine.compute(history, scenario=scenario, planner_name=planner_name)
    logger.debug("Finished metrics computation!")
    logger.debug("Saving metric statistics!")
    metric_engine.write_to_files(metric_files)
    logger.debug("Saved metrics!")

    # if save_samples_to_disk:
    #     logger.debug(f"Saving rollout samples to {save_samples_path}...")

    #     # import time
    #     # start = time.time()

    #     cache = FeatureCachePickle()
    #     save_samples_path = pathlib.Path(save_samples_path)

    #     observations = planner.get_observations()
    #     rewards = compute_rewards(other_files, metric_engine)
    #     dones = rewards == 0

    #     # Downsample
    #     observations = observations[::5]
    #     rewards = rewards[:145].reshape(-1,5).mean(axis=1)
    #     dones = dones[:145].reshape(-1,5).any(axis=1)

    #     # TODO: bruh
    #     for i in range(12):
    #         target = construct_targets(history, i*5)
    #         observation = observations[i]
    #         observation.update({'trajectory': target})

    #         # Store observation features
    #         for feature_name in observation:
    #             file_name = (
    #                 save_samples_path / scenario.log_name / scenario.scenario_type / scenario.token / feature_name
    #             )
    #             file_name.parent.mkdir(parents=True, exist_ok=True)
    #             success = cache.store_computed_feature_to_folder(file_name, observation[feature_name])
    #             assert success, f'Failed to store feature {feature_name} at {file_name}'
    #             # print(f'Saved {feature_name} to {file_name}')

    #         # Store other stuff
    #         pass
        
    #     # end = time.time()
    #     # print('Done saving to disk!')
    #     # print(f'Time elapsed: {end-start}')

    #     logger.debug("Saved rollout samples to disk!")


def construct_targets(history, ref_idx=0):
    ego_states = [history.data[i].ego_state for i in range(len(history.data))]
    ego_poses = [(state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading) for state in ego_states]
    ego_poses = np.array(ego_poses)
    ego_poses = ego_poses[range(ref_idx,ref_idx+85,5)]

    # Transform to ego frame
    ref = ego_poses[0]
    ego_poses[:,:2] -= ref[:2]

    rot = np.array([[np.cos(ref[2]), -np.sin(ref[2])], [np.sin(ref[2]), np.cos(ref[2])]])
    ego_poses[:,:2] = ego_poses[:,:2].dot(rot)

    ego_poses[:,2] -= ref[2]

    return Trajectory(data=ego_poses[1:])


def compute_rewards(metrics, metric_engine):
    """
    Compute rewards given per-step metrics
    Trying to follow the original driving score calculation closely
    TODO: make this configurable, right now copy-pasting magic values
    """
    ego_progress, expert_progress = metrics['ego_progress_along_expert_route']
    ego_is_comfortable = metrics['ego_is_comfortable']
    speed_limit_compliance = metrics['speed_limit_compliance']
    no_ego_at_fault_collisions = metrics['no_ego_at_fault_collisions']
    time_to_collision = metrics['time_to_collision_within_bound']
    drivable_area_compliance = metrics['drivable_area_compliance']
    driving_direction_compliance = metrics['driving_direction_compliance']

    # PROGRESS ALONG ROUTE    
    score_progress_threshold = 2.0

    ego_progress = np.array(ego_progress)
    expert_progress = np.array(expert_progress)
    overall_ego_progress = np.cumsum(ego_progress)
    overall_expert_progress = np.cumsum(expert_progress)[1:]

    overall_ego_progress[overall_ego_progress < score_progress_threshold] = score_progress_threshold
    overall_expert_progress[overall_expert_progress < score_progress_threshold] = score_progress_threshold
    ego_expert_progress_along_route_ratio = overall_ego_progress / overall_expert_progress
    ego_expert_progress_along_route_ratio[ego_expert_progress_along_route_ratio > 1] = 1
    
    # EGO IS MAKING PROGRESS
    min_progress_threshold = 0.2
    ego_is_making_progress = (ego_expert_progress_along_route_ratio > min_progress_threshold).astype(float)

    # EGO IS COMFORTABLE
    ego_is_comfortable = ego_is_comfortable.astype(float)

    # SPEED LIMIT COMPLIANCE
    speed_limit_compliance = (np.array(speed_limit_compliance) == 0).astype(float)

    # NO EGO AT FAULT COLLISIONS
    # If you collide then just set all future steps to 0
    if no_ego_at_fault_collisions.any():
        first_collision_idx = np.min(np.where(no_ego_at_fault_collisions > 0)[0])
        no_ego_at_fault_collisions = np.ones_like(no_ego_at_fault_collisions)
        no_ego_at_fault_collisions[first_collision_idx:] = 0
    else:
        no_ego_at_fault_collisions = 1 - no_ego_at_fault_collisions

    # TIME TO COLLISION
    least_min_ttc = 0.95
    time_to_collision_within_bound = (time_to_collision > least_min_ttc).astype(float)

    # DRIVABLE AREA COMPLIANCE
    drivable_area_compliance = np.array(drivable_area_compliance)

    # DRIVING DIRECTION COMPLIANCE
    driving_direction_compliance_threshold = 2.
    driving_direction_violation_threshold = 6.

    driving_direction_compliance = np.array(driving_direction_compliance)
    max_negative_progress_over_interval = np.abs(np.minimum.accumulate(driving_direction_compliance))
    driving_direction_compliance = np.zeros_like(driving_direction_compliance)
    driving_direction_compliance[max_negative_progress_over_interval < driving_direction_compliance_threshold] = 1.0
    driving_direction_compliance[
        (max_negative_progress_over_interval > driving_direction_compliance_threshold) * \
        (max_negative_progress_over_interval < driving_direction_violation_threshold)
    ] = 0.5

    # COMPUTE WEIGHTED SCORE
    # The scenario score is defined as:
    # - the weighted average score of:
    #   - ego_progress_along_expert_route
    #   - time_to_collision_within_bound
    #   - speed_limit_compliance
    #   - ego_is_comfortable

    ego_progress_along_expert_route_weight = 5.0
    time_to_collision_within_bound_weight = 5.0
    speed_limit_compliance_weight = 4.0
    ego_is_comfortable_weight = 2.0
    total_weight = (
        ego_progress_along_expert_route_weight + time_to_collision_within_bound_weight + \
        speed_limit_compliance_weight + ego_is_comfortable_weight
    )

    weighted_average = (
        ego_expert_progress_along_route_ratio * ego_progress_along_expert_route_weight + \
        time_to_collision_within_bound * time_to_collision_within_bound_weight + \
        speed_limit_compliance * speed_limit_compliance_weight + \
        ego_is_comfortable * ego_is_comfortable_weight
    ) / total_weight

    # - multiplied by the score of:
    #   - no_ego_at_fault_collisions
    #   - drivable_area_compliance
    #   - ego_is_making_progress
    #   - driving_direction_compliance

    driving_score = weighted_average * (
        no_ego_at_fault_collisions * drivable_area_compliance * \
        ego_is_making_progress * driving_direction_compliance
    )

    return driving_score


class MetricCallback(AbstractCallback):
    """Callback for computing metrics at the end of the simulation."""

    def __init__(
        self, 
        metric_engine: MetricsEngine, 
        worker_pool: Optional[WorkerPool] = None,
        planner: AbstractPlanner = None,
        save_samples_to_disk: bool = False, 
        save_samples_path: str = None
    ):
        """
        Build A metric callback.
        :param metric_engine: Metric Engine.
        """
        self._metric_engine = metric_engine
        
        self._planner = planner
        self._save_samples_to_disk = save_samples_to_disk
        self._save_samples_path = save_samples_path

        self._pool = worker_pool
        self._futures: List[Future[None]] = []

    @property
    def metric_engine(self) -> MetricsEngine:
        """
        Returns metric engine.
        :return: metric engine
        """
        return self._metric_engine

    @property
    def futures(self) -> List[Future[None]]:
        """
        Returns a list of futures, eg. for the main process to block on.
        :return: any futures generated by running any part of the callback asynchronously.
        """
        return self._futures

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Inherited, see superclass."""
        if self._pool is not None:
            self._futures = []
            self._futures.append(
                self._pool.submit(
                    Task(run_metric_engine, num_cpus=1, num_gpus=0),
                    metric_engine=self._metric_engine,
                    history=history,
                    scenario=setup.scenario,
                    planner_name=planner.name(),
                    planner=self._planner,
                    save_samples_to_disk=self._save_samples_to_disk,
                    save_samples_path=self._save_samples_path
                )
            )
        else:
            run_metric_engine(
                metric_engine=self._metric_engine, 
                history=history, 
                scenario=setup.scenario, 
                planner_name=planner.name(),
                planner=self._planner,
                save_samples_to_disk=self._save_samples_to_disk,
                save_samples_path=self._save_samples_path
            )
