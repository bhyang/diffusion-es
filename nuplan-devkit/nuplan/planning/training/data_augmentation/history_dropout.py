from typing import List, Optional, Tuple

import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import ParameterToScale, ScalingDirection
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class HistoryDropoutAugmentor(AbstractAugmentor):
    def __init__(self, augment_prob: float) -> None:
        self._augment_prob = augment_prob

    def augment(
        self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets

        # Zero out the ego history
        for batch_idx in range(len(features['generic_agents'].ego)):
            features['generic_agents'].ego[batch_idx] = np.zeros_like(
                features['generic_agents'].ego[batch_idx])

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
