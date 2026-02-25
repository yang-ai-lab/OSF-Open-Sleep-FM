"""
1D Vision Transformer for time-series signals.

Patchify modes:
- lead_wise=0: 1D patchify (all channels in one patch), no lead embedding
- lead_wise=1: 2D patchify (channel groups), with lead embedding by default
"""

import torch
import torch.nn as nn
from einops import rearrange


class DropPath(nn.Module):
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, drop_out_rate=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(drop_out_rate)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int = 8, dim_head: int = 64,
                 qkv_bias: bool = True, drop_out_rate: float = 0., attn_drop_out_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, output_dim), nn.Dropout(drop_out_rate))
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, heads: int = 8,
                 dim_head: int = 32, qkv_bias: bool = True, drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0., drop_path_rate: float = 0.):
        super().__init__()
        attn = Attention(input_dim, output_dim, heads, dim_head, qkv_bias, drop_out_rate, attn_drop_out_rate)
        self.attn = PreNorm(input_dim, attn)
        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        ff = FeedForward(output_dim, output_dim, hidden_dim, drop_out_rate)
        self.ff = PreNorm(output_dim, ff)
        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.droppath1(self.attn(x)) + x
        x = self.droppath2(self.ff(x)) + x
        return x


class ViT(nn.Module):
    def __init__(self,
                 num_leads: int,
                 seq_len: int,
                 patch_size: int,
                 lead_wise=0,
                 patch_size_ch=4,
                 use_lead_embedding: bool = True,
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 **kwargs):
        super().__init__()
        assert seq_len % patch_size == 0
        num_patches = seq_len // patch_size
        self.lead_wise = lead_wise
        self.use_lead_embedding = use_lead_embedding

        if lead_wise == 0:
            self.to_patch_embedding = nn.Conv1d(num_leads, width, kernel_size=patch_size, stride=patch_size, bias=False)
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, width))
        else:
            self.to_patch_embedding = nn.Conv2d(1, width, kernel_size=(patch_size_ch, patch_size),
                                                stride=(patch_size_ch, patch_size), bias=False)
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches * num_leads // patch_size_ch, width))
            if use_lead_embedding:
                self.lead_emb = nn.Embedding(num_leads // patch_size_ch, width)
            else:
                self.lead_emb = None

        self.dropout = nn.Dropout(drop_out_rate)
        self.depth = depth
        self.width = width

        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(width, width, mlp_dim, heads, dim_head, qkv_bias,
                                     drop_out_rate, attn_drop_out_rate, drop_path_rate_list[i])
            self.add_module(f'block{i}', block)

        self.norm = nn.LayerNorm(width)
        self.head = nn.Identity()

    def _patchify_and_embed(self, series: torch.Tensor) -> torch.Tensor:
        """Patchify input and add positional/lead embeddings. [B,C,T] -> [B,N,D]"""
        if self.lead_wise == 0:
            x = self.to_patch_embedding(series)  # [B, D, N]
            x = rearrange(x, 'b c n -> b n c')   # [B, N, D]
            x = x + self.pos_embedding[:, :x.size(1), :].to(x.device)
        else:
            x = self.to_patch_embedding(series.unsqueeze(1))  # [B, D, Lr, Nt]
            Lr, Nt = x.shape[-2], x.shape[-1]
            x = rearrange(x, 'b c lr nt -> b (lr nt) c')      # [B, N, D]
            x = x + self.pos_embedding[:, :x.size(1), :].to(x.device)
            if self.use_lead_embedding and self.lead_emb is not None:
                row_ids = torch.arange(Lr, device=x.device).repeat_interleave(Nt)
                x = x + self.lead_emb(row_ids)[None, :, :]
        return x

    def forward_encoding(self, series: torch.Tensor) -> torch.Tensor:
        """Encode series. Returns [B,D] (mean pooled)."""
        x = self._patchify_and_embed(series)
        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)
        x = x.mean(dim=1)
        return self.norm(x)

    def forward(self, series):
        x = self.forward_encoding(series)
        return self.head(x)

    def reset_head(self, num_classes=1):
        del self.head
        self.head = nn.Linear(self.width, num_classes)


def vit_nano(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    return ViT(num_leads=num_leads, num_classes=num_classes, seq_len=seq_len, patch_size=patch_size,
               width=128, depth=6, heads=4, mlp_dim=512, **kwargs)


def vit_tiny(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    return ViT(num_leads=num_leads, num_classes=num_classes, seq_len=seq_len, patch_size=patch_size,
               width=192, depth=12, heads=3, mlp_dim=768, **kwargs)


def vit_small(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    return ViT(num_leads=num_leads, num_classes=num_classes, seq_len=seq_len, patch_size=patch_size,
               width=384, depth=12, heads=6, mlp_dim=1536, **kwargs)


def vit_middle(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    return ViT(num_leads=num_leads, num_classes=num_classes, seq_len=seq_len, patch_size=patch_size,
               width=512, depth=12, heads=8, mlp_dim=2048, **kwargs)


def vit_base(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    return ViT(num_leads=num_leads, num_classes=num_classes, seq_len=seq_len, patch_size=patch_size,
               width=768, depth=12, heads=12, mlp_dim=3072, **kwargs)
