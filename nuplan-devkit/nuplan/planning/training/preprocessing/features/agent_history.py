from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import numpy as np

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)


@dataclass
class AgentHistory(AbstractModelFeature):
    ego: List[FeatureDataType]  # BxTxD
    data: List[FeatureDataType] # BxAxTxD
    mask: List[FeatureDataType] # BxAxT

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

    # @classmethod
    # def collate(cls, batch: List[AgentHistory]) -> AgentTrajectory:
    #     """
    #     Implemented. See interface.
    #     Collates a list of features that each have batch size of 1.
    #     """
    #     return AgentHistory(
    #         ego=np.stack([item.ego for item in batch], axis=0),
    #         data=np.stack([item.data for item in batch], axis=0),
    #         mask=np.stack([item.mask for item in batch], axis=0)
    #     )

    def to_feature_tensor(self) -> AgentHistory:
        """Implemented. See interface."""
        return self

    def to_device(self, device: torch.device) -> AgentHistory:
        """Implemented. See interface."""
        return AgentHistory(
            ego=to_tensor(self.ego).to(device),
            data=to_tensor(self.data).to(device),
            mask=to_tensor(self.mask).to(device)
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AgentHistory:
        """Implemented. See interface."""
        return AgentHistory(
            ego=data["ego"], 
            data=data["data"], 
            mask=data["mask"]
        )

    def unpack(self) -> List[AgentHistory]:
        """Implemented. See interface."""
        return [AgentHistory(
            ego=self.ego[i], 
            data=self.data[i], 
            mask=self.mask[i]
        ) for i in range(self.batch_size)]

    def repeat_interleave(self, n_repeats, dim):
        return AgentHistory(
            ego=self.ego.repeat_interleave(n_repeats, dim),
            data=self.data.repeat_interleave(n_repeats, dim),
            mask=self.mask.repeat_interleave(n_repeats, dim)
        )