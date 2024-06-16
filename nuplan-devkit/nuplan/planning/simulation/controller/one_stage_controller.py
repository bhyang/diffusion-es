from typing import Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class OneStageController(AbstractEgoController):
    """
    TwoStageController except it takes controls directly as input to the motion model without a tracker.
    Useful for when we run the tracker ourselves in-the-loop and don't want to repeat computation.
    """

    def __init__(self, scenario: AbstractScenario, motion_model: AbstractMotionModel):
        """
        Constructor for TwoStageController
        :param scenario: Scenario
        :param tracker: The tracker used to compute control actions
        :param motion_model: The motion model to propagate the control actions
        """
        self._scenario = scenario
        self._motion_model = motion_model

        # set to None to allow lazily loading of initial ego state
        self._current_state: Optional[EgoState] = None

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._current_state = None

    def get_state(self) -> EgoState:
        """Inherited, see superclass."""
        if self._current_state is None:
            self._current_state = self._scenario.initial_ego_state

        return self._current_state

    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> None:
        """Inherited, see superclass."""
        sampling_time = next_iteration.time_point - current_iteration.time_point

        # # Compute the dynamic state to propagate the model
        # dynamic_state = self._tracker.track_trajectory(current_iteration, next_iteration, ego_state, trajectory)

        command = trajectory.get_updated_command()

        dynamic_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=ego_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=ego_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(command[0], 0),
            tire_steering_rate=command[1],
        )

        # Propagate ego state using the motion model
        self._current_state = self._motion_model.propagate_state(
            state=ego_state, ideal_dynamic_state=dynamic_state, sampling_time=sampling_time
        )
