"""
Balanced/imbalanced learning losses.
Reference: https://github.com/YyzHarry/SubpopBench
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    Paper: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor (float or [num_classes] tensor)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean' or 'none'
    """
    def __init__(self, alpha: Optional[float | torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor([alpha], dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        elif alpha is None:
            self.alpha = None
        else:
            raise ValueError(f"alpha must be float, Tensor, or None, got {type(alpha)}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] unnormalized logits
            targets: [B] class indices
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.dim() == 0 or len(self.alpha) == 1:
                alpha_t = self.alpha.squeeze()
            else:
                alpha_t = self.alpha[targets]  # [B]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"reduction must be 'mean' or 'none', got {self.reduction}")


class BalancedSoftmax(nn.Module):
    """
    Balanced Softmax: adjusted_logits = logits + log(class_counts)
    Paper: https://arxiv.org/abs/2007.10740
    
    Args:
        class_counts: [C] tensor of sample counts per class
        reduction: 'mean' or 'none'
    """
    def __init__(self, class_counts: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        if not isinstance(class_counts, torch.Tensor):
            class_counts = torch.tensor(class_counts, dtype=torch.float32)
        
        class_counts = class_counts.float()
        if (class_counts == 0).any():
            zero_classes = (class_counts == 0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"BalancedSoftmax requires non-zero class counts. Zero counts: {zero_classes}")
        
        self.register_buffer("log_class_counts", torch.log(class_counts))
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] unnormalized logits
            targets: [B] class indices
        """
        adjusted_logits = logits + self.log_class_counts.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)
