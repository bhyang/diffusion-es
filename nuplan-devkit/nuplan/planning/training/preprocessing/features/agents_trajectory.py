from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)


@dataclass
class AgentTrajectory(AbstractModelFeature):
    data: List[FeatureDataType]                  # BxAxTxD
    mask: Optional[List[FeatureDataType]] = None # BxAxT

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        if len(self.data) == 0:
            raise AssertionError("Batch size has to be > 0!")

    @property
    def batch_size(self) -> int:
        """
        :return: batch size
        """
        return self.data.shape[0]

    @classmethod
    def collate(cls, batch: List[AgentTrajectory]) -> AgentTrajectory:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return AgentTrajectory(
            data=np.stack([item.data for item in batch], axis=0),
            mask=np.stack([item.mask for item in batch], axis=0)
        )

    def to_feature_tensor(self) -> AgentTrajectory:
        """Implemented. See interface."""
        return self

    def to_device(self, device: torch.device) -> AgentTrajectory:
        """Implemented. See interface."""
        return AgentTrajectory(
            data=to_tensor(self.data).to(device),
            mask=to_tensor(self.mask).to(device)
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AgentTrajectory:
        """Implemented. See interface."""
        return AgentTrajectory(
            data=data["data"], 
            mask=data["mask"]
        )

    def unpack(self) -> List[AgentTrajectory]:
        """Implemented. See interface."""
        return [AgentTrajectory(
            data=self.data[i], 
            mask=self.mask[i]
        ) for i in range(self.batch_size)]