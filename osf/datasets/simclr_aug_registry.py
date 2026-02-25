"""
Two-view augmentation registry for SSL pretraining (SimCLR, DINO).
Provides multi-view generation pipelines for contrastive and self-distillation methods.
"""
from __future__ import annotations
from typing import Callable, Dict
import torch
from osf.datasets import augmentations as A


def _two_view(pipe1: Callable, pipe2: Callable | None = None) -> Callable:
    """Wrap one/two single-view pipelines into a two-view augmentation maker."""
    if pipe2 is None:
        pipe2 = pipe1
    def make(x: torch.Tensor):
        return pipe1(x), pipe2(x)
    return make


SIMCLR_AUG_REGISTRY: Dict[str, Callable] = {
    "none": _two_view(lambda x: x),
    
    "channel_dropout": _two_view(lambda x: A.channel_dropout(x, drop_prob=0.2, min_keep=1)),
    "channel_dropout_light": _two_view(lambda x: A.channel_dropout(x, drop_prob=0.25, min_keep=1)),
    "channel_dropout_aligned": _two_view(lambda x: A.channel_dropout(x, drop_prob=0.5, min_keep=1)),
}

SIMCLR_AUG_FACTORIES: Dict[str, Callable[..., Callable]] = {}


def build_simclr_augmentor(name: str, **kwargs) -> Callable:
    key = (name or "none").lower()
    if key in SIMCLR_AUG_REGISTRY:
        return SIMCLR_AUG_REGISTRY[key]
    if key in SIMCLR_AUG_FACTORIES:
        return SIMCLR_AUG_FACTORIES[key](**kwargs)
    raise ValueError(
        f"Unknown simclr_augmentation '{name}'. "
        f"Available presets: {list(SIMCLR_AUG_REGISTRY.keys())} | "
        f"factories: {list(SIMCLR_AUG_FACTORIES.keys())}"
    )


def _per_channel_span_mask_factory(
    ratio: tuple[float, float] = (0.10, 0.30),
    n_spans: int = 1,
    fill: str | torch.Tensor = "zero",
    noise_scale: float = 0.05,
    same_mask_for_batch: bool = False,
):
    assert 0.0 <= ratio[0] <= ratio[1] <= 1.0

    def _single_view(x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        device, dtype = x.device, x.dtype

        min_len = max(1, int(round(ratio[0] * T)))
        max_len = max(min_len, int(round(ratio[1] * T)))
        arange_T = torch.arange(T, device=device)
        mask = torch.zeros((B, C, T), device=device, dtype=torch.bool)
        shape_bc = (1, C) if same_mask_for_batch else (B, C)

        for _ in range(max(1, int(n_spans))):
            if max_len == min_len:
                lengths = torch.full(shape_bc, max_len, device=device, dtype=torch.long)
            else:
                lengths = torch.randint(min_len, max_len + 1, shape_bc, device=device)
            max_start = (T - lengths).clamp_min(0)
            if (max_start > 0).any():
                rnd = torch.rand_like(max_start, dtype=torch.float32)
                starts = torch.floor(rnd * (max_start.to(torch.float32) + 1)).to(torch.long)
            else:
                starts = torch.zeros_like(max_start)
            if same_mask_for_batch and B > 1:
                starts = starts.expand(B, C)
                lengths = lengths.expand(B, C)
            span_mask = (arange_T.view(1, 1, T) >= starts.unsqueeze(-1)) & \
                        (arange_T.view(1, 1, T) < (starts + lengths).unsqueeze(-1))
            mask |= span_mask

        y = x.clone()
        if isinstance(fill, torch.Tensor):
            fill_t = fill.to(device=device, dtype=dtype)
            if fill_t.dim() == 0:
                fill_t = fill_t.view(1, 1, 1)
            if fill_t.shape[-1] == 1 and fill_t.dim() == 3 and fill_t.shape[0] in (1, B):
                fill_t = fill_t if fill_t.shape[0] == B else fill_t.expand(B, -1, -1)
            elif fill_t.dim() == 3 and fill_t.shape == (B, C, T):
                pass
            elif fill_t.dim() == 3 and fill_t.shape == (1, C, 1):
                fill_t = fill_t.expand(B, -1, T)
            y[mask] = fill_t[mask.expand_as(fill_t)]
        elif fill == "zero":
            y[mask] = 0.0
        elif fill == "mean":
            m = x.mean(dim=-1, keepdim=True)
            y = torch.where(mask, m.expand_as(x), y)
        elif fill == "noise":
            m = x.mean(dim=-1, keepdim=True)
            s = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
            noise = torch.randn_like(x) * (s * noise_scale) + m
            y = torch.where(mask, noise, y)
        else:
            raise ValueError(f"Unknown fill mode: {fill!r}")
        return y

    return _two_view(_single_view)


SIMCLR_AUG_FACTORIES["pc_span_mask"] = _per_channel_span_mask_factory

SIMCLR_AUG_REGISTRY.update({
    "pc_span_mask_light": _per_channel_span_mask_factory(
        ratio=(0.1, 0.3), n_spans=1, fill="zero", noise_scale=0.05, same_mask_for_batch=False
    ),
    "pc_span_mask_heavy": _per_channel_span_mask_factory(
        ratio=(0.20, 0.6), n_spans=2, fill="zero", noise_scale=0.05, same_mask_for_batch=False
    ),
    "pc_span_mask_aligned": _per_channel_span_mask_factory(
        ratio=(0.3, 0.6), n_spans=1, fill="zero", noise_scale=0, same_mask_for_batch=False
    ),
})


def _channel_then_pcspan_factory(
    drop_prob: float = 0.3,
    min_keep: int = 1,
    ratio: tuple[float, float] = (0.10, 0.30),
    n_spans: int = 1,
    fill: str = "zero",
    noise_scale: float = 0.05,
    same_mask_for_batch: bool = False,
):
    def single_view(x: torch.Tensor) -> torch.Tensor:
        y = A.channel_dropout(x, drop_prob=drop_prob, min_keep=min_keep)
        B, C, T = y.shape
        device = y.device

        min_len = max(1, int(round(ratio[0] * T)))
        max_len = max(min_len, int(round(ratio[1] * T)))
        arange_T = torch.arange(T, device=device)
        mask = torch.zeros((B, C, T), device=device, dtype=torch.bool)
        shape_bc = (1, C) if same_mask_for_batch else (B, C)

        for _ in range(max(1, int(n_spans))):
            lengths = torch.full(shape_bc, max_len, device=device, dtype=torch.long) \
                      if max_len == min_len else torch.randint(min_len, max_len + 1, shape_bc, device=device)
            max_start = (T - lengths).clamp_min(0)
            if (max_start > 0).any():
                rnd = torch.rand_like(max_start, dtype=torch.float32)
                starts = torch.floor(rnd * (max_start.to(torch.float32) + 1)).to(torch.long)
            else:
                starts = torch.zeros_like(max_start)
            if same_mask_for_batch and B > 1:
                starts = starts.expand(B, C)
                lengths = lengths.expand(B, C)
            span_mask = (arange_T.view(1, 1, T) >= starts.unsqueeze(-1)) & \
                        (arange_T.view(1, 1, T) < (starts + lengths).unsqueeze(-1))
            mask |= span_mask

        out = y.clone()
        if fill == "zero":
            out[mask] = 0.0
        elif fill == "mean":
            m = y.mean(dim=-1, keepdim=True)
            out = torch.where(mask, m.expand_as(y), out)
        elif fill == "noise":
            m = y.mean(dim=-1, keepdim=True)
            s = y.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
            noise = torch.randn_like(y) * (s * noise_scale) + m
            out = torch.where(mask, noise, out)
        else:
            raise ValueError(f"Unknown fill: {fill!r}")
        return out

    return _two_view(single_view)


SIMCLR_AUG_FACTORIES["chan_then_pcspan"] = _channel_then_pcspan_factory


def _crop_then_chan_pcspan_factory(
    crop_ratio: tuple[float, float] = (0.25, 0.75),
    align_to: int = 40,
    drop_prob: float = 0.5,
    min_keep: int = 1,
    span_ratio: tuple[float, float] = (0.3, 0.6),
    n_spans: int = 1,
    fill: str = "zero",
    noise_scale: float = 0.0,
    same_mask_for_batch: bool = False,
):
    def single_view(x: torch.Tensor) -> torch.Tensor:
        y = A.random_time_crop(x, ratio=crop_ratio, resize_back=True, align_to=align_to)
        y = A.channel_dropout(y, drop_prob=drop_prob, min_keep=min_keep)

        B, C, T = y.shape
        device = y.device
        min_len = max(1, int(round(span_ratio[0] * T)))
        max_len = max(min_len, int(round(span_ratio[1] * T)))
        arange_T = torch.arange(T, device=device)
        mask = torch.zeros((B, C, T), device=device, dtype=torch.bool)
        shape_bc = (1, C) if same_mask_for_batch else (B, C)

        for _ in range(max(1, int(n_spans))):
            lengths = torch.full(shape_bc, max_len, device=device, dtype=torch.long) \
                      if max_len == min_len else torch.randint(min_len, max_len + 1, shape_bc, device=device)
            max_start = (T - lengths).clamp_min(0)
            if (max_start > 0).any():
                rnd = torch.rand_like(max_start, dtype=torch.float32)
                starts = torch.floor(rnd * (max_start.to(torch.float32) + 1)).to(torch.long)
            else:
                starts = torch.zeros_like(max_start)
            if same_mask_for_batch and B > 1:
                starts = starts.expand(B, C)
                lengths = lengths.expand(B, C)
            span_mask = (arange_T.view(1, 1, T) >= starts.unsqueeze(-1)) & \
                        (arange_T.view(1, 1, T) < (starts + lengths).unsqueeze(-1))
            mask |= span_mask

        out = y.clone()
        if fill == "zero":
            out[mask] = 0.0
        elif fill == "mean":
            m = y.mean(dim=-1, keepdim=True)
            out = torch.where(mask, m.expand_as(y), out)
        elif fill == "noise":
            m = y.mean(dim=-1, keepdim=True)
            s = y.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
            noise = torch.randn_like(y) * (s * noise_scale) + m
            out = torch.where(mask, noise, out)
        else:
            raise ValueError(f"Unknown fill: {fill!r}")
        return out

    return _two_view(single_view)


SIMCLR_AUG_FACTORIES["crop_then_chan_pcspan"] = _crop_then_chan_pcspan_factory

SIMCLR_AUG_REGISTRY.update({
    "chan_then_pcspan": _channel_then_pcspan_factory(
        drop_prob=0.5, min_keep=1, ratio=(0.3, 0.6), n_spans=1, fill="zero",
        noise_scale=0, same_mask_for_batch=False
    ),
    "chan_then_pcspan_light": _channel_then_pcspan_factory(
        drop_prob=0.25, min_keep=1, ratio=(0.3, 0.6), n_spans=1, fill="zero",
        noise_scale=0, same_mask_for_batch=False
    ),
    "crop_then_chan_pcspan": _crop_then_chan_pcspan_factory(
        crop_ratio=(0.25, 0.75), align_to=40, drop_prob=0.5, min_keep=1,
        span_ratio=(0.3, 0.6), n_spans=1, fill="zero", noise_scale=0, same_mask_for_batch=False
    ),
    "crop_then_chan_pcspan_light": _crop_then_chan_pcspan_factory(
        crop_ratio=(0.25, 0.75), align_to=40, drop_prob=0.25, min_keep=1,
        span_ratio=(0.3, 0.6), n_spans=1, fill="zero", noise_scale=0, same_mask_for_batch=False
    ),
})
