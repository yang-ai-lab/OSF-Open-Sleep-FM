"""
Data augmentations for SSL pretraining (SimCLR, DINO).
"""
import torch
import torch.nn.functional as F
from typing import Tuple


@torch.no_grad()
def random_time_crop(
    x: torch.Tensor, 
    ratio: Tuple[float, float] | float = (0.6, 0.9),
    *, 
    resize_back: bool = True,
    align_to: int | None = 40
) -> torch.Tensor:
    """
    Randomly crop a contiguous sub-sequence per sample, optionally resize back to original T.
    
    Args:
        x: (B, C, T)
        ratio: crop length ratio in [low, high] or a float
        resize_back: if True, linearly interpolate the cropped view back to length T
        align_to: if not None, crop length is rounded to a multiple of align_to (>= align_to)
    """
    assert x.dim() == 3, f"expected (B,C,T), got {tuple(x.shape)}"
    B, C, T = x.shape
    dev = x.device

    def _sample_L() -> int:
        if isinstance(ratio, (tuple, list)):
            a, b = float(ratio[0]), float(ratio[1])
            r = torch.empty((), device=dev).uniform_(a, b).item()
        else:
            r = float(ratio)
        L = max(2, int(round(T * r)))
        if align_to and align_to > 1:
            L = max(align_to, int(round(L / align_to)) * align_to)
        return min(L, T)

    Ls = [_sample_L() for _ in range(B)]
    outs = []
    for b in range(B):
        L = Ls[b]
        max_start = max(0, T - L)
        s = int(torch.randint(0, max_start + 1, (1,), device=dev).item())
        v = x[b, :, s:s+L]  # (C, L)
        if resize_back and v.shape[-1] != T:
            v = F.interpolate(v[None], size=T, mode="linear", align_corners=False)[0]
        outs.append(v)
    return torch.stack(outs, dim=0)


@torch.no_grad()
def channel_dropout(
    x: torch.Tensor, 
    drop_prob: float = 0.2, 
    min_keep: int = 1
) -> torch.Tensor:
    """
    Drop entire channels to zero with probability drop_prob (per sample, per channel).
    Ensures at least `min_keep` channels remain active in each sample.
    
    Args:
        x: (B, C, T)
        drop_prob: probability to drop each channel
        min_keep: minimum number of channels to keep per sample
    """
    assert x.dim() == 3
    B, C, T = x.shape
    mask = (torch.rand(B, C, 1, device=x.device, dtype=x.dtype) > drop_prob).to(x.dtype)
    
    # Ensure at least min_keep channels kept
    keep = mask.sum(dim=1, keepdim=True)  # (B, 1, 1)
    need = (keep < min_keep).squeeze(-1).squeeze(-1)  # (B,)
    if need.any():
        for b in torch.where(need)[0]:
            idx = torch.randperm(C, device=x.device)[:min_keep]
            mask[b, idx, 0] = 1.0
    
    return x * mask
