from typing import List

import torch

from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class AverageDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'avg_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        return torch.norm(predicted_trajectory.xy - targets_trajectory.xy, dim=-1).mean()


class FinalDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        return torch.norm(predicted_trajectory.terminal_position - targets_trajectory.terminal_position, dim=-1).mean()


class AverageHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'avg_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        errors = torch.abs(predicted_trajectory.heading - targets_trajectory.heading)
        return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()


class FinalHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        errors = torch.abs(predicted_trajectory.terminal_heading - targets_trajectory.terminal_heading)
        return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()


class NearDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from first two seconds of a trajectory.
    """

    def __init__(self, name: str = 'near_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]
        
        return torch.norm(predicted_trajectory.xy - targets_trajectory.xy, dim=-1)[:,:4].mean()
    

from nuplan.planning.training.modeling.models.diffusion_utils import (
    Whitener,
    Standardizer
)


class WhitenedError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from first two seconds of a trajectory.
    """

    def __init__(self, name: str = 'whitened_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

        self.standardizer = Standardizer(50).cuda()
        self.whitener = Whitener('/zfsauton2/home/brianyan/nuplan-devkit/params_abs.th').cuda()

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        pred_std = self.standardizer.transform_features(predicted_trajectory.data)
        target_std = self.standardizer.transform_features(targets_trajectory.data)

        pred_whitened = self.whitener.transform_features(pred_std)
        target_whitened = self.whitener.transform_features(target_std)

        return (pred_whitened - target_whitened).abs().mean()


class WhitenedAccuracy(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from first two seconds of a trajectory.
    """

    def __init__(self, name: str = 'whitened_acc') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

        self.standardizer = Standardizer(50).cuda()
        self.whitener = Whitener('/zfsauton2/home/brianyan/nuplan-devkit/params_abs.th').cuda()
        self._threshold = .02

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        pred_std = self.standardizer.transform_features(predicted_trajectory.data)
        target_std = self.standardizer.transform_features(targets_trajectory.data)

        pred_whitened = self.whitener.transform_features(pred_std)
        target_whitened = self.whitener.transform_features(target_std)

        errors = (pred_whitened - target_whitened).abs().mean(1)
        hits = errors < self._threshold
        return hits.sum() / hits.shape[0]
    

from nuplan.planning.training.preprocessing.features.agents_trajectory import AgentTrajectory
    

class MADisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from first two seconds of a trajectory.
    """

    def __init__(self, name: str = 'ma_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agent_trajectories"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: AgentTrajectory = predictions["agent_trajectories"]
        targets_trajectory: AgentTrajectory = targets["agent_trajectories"]

        loss_unreduced = torch.norm(predicted_trajectory.data - targets_trajectory.data, dim=-1)
        loss_unreduced = loss_unreduced * targets_trajectory.mask
        loss = loss_unreduced.mean()

        return loss
        
        # return torch.norm(predicted_trajectory.xy - targets_trajectory.xy, dim=-1)[:,:4].mean()