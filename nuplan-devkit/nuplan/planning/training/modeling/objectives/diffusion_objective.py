from typing import Dict, List, cast

import torch
import torch.nn.functional as F

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType


class DiffusionObjective(AbstractObjective):
    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0):
        pass

    def name(self) -> str:
        """
        Name of the objective
        """
        return 'diffusion_objective'

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        if 'epsilon' in predictions:
            pred = predictions['epsilon']
            gt = predictions['gt_epsilon']
            loss = F.mse_loss(pred, gt)
        else:
            # this objective is meaningless at test time
            loss = torch.as_tensor(0., device=predictions['trajectory'].data.device)
        return loss
