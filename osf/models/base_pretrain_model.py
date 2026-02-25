import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from osf.backbone.vit1d import vit_nano, vit_tiny, vit_small, vit_middle, vit_base

VIT_FACTORIES = {
    "vit_nano": vit_nano,
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_middle": vit_middle,
    "vit_base": vit_base,
}


class PSGModalityEncoder(nn.Module):
    """ViT encoder for PSG signals: backbone -> optional projection -> L2-norm"""

    def __init__(self, *,
                 encoder_name: str,
                 proj_out: int = 256,
                 proj_hidden: int = 512,
                 freq: int = 64,
                 win_sec: int = 30,
                 channel: int = 11,
                 lead_wise=0,
                 patch_size=40,
                 patch_size_ch=4,
                 use_lead_embedding: bool = True,
                 is_proj_head=1):
        super().__init__()
        token_len = freq * win_sec
        self.token_len = token_len
        self.patch_size = patch_size

        if encoder_name not in VIT_FACTORIES:
            raise ValueError(f"Unknown encoder_name: {encoder_name}. Choose from {list(VIT_FACTORIES.keys())}")

        self.backbone = VIT_FACTORIES[encoder_name](
            num_leads=channel, seq_len=token_len, patch_size=patch_size,
            lead_wise=lead_wise, patch_size_ch=patch_size_ch,
            use_lead_embedding=use_lead_embedding,
        )

        d_model = self.backbone.width
        if is_proj_head == 1:
            self.proj_head = nn.Sequential(
                nn.Linear(d_model, proj_hidden),
                nn.LayerNorm(proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden, proj_out),
                nn.LayerNorm(proj_out),
            )
        else:
            self.proj_head = None

    def forward(self, x, normalize=True):
        # x: [B, C, T]
        h = self.backbone(x)  # [B, D]
        if self.proj_head is not None:
            h = self.proj_head(h)  # [B, proj_out]
        if normalize:
            return F.normalize(h, dim=-1)
        return h


class BasePretrainModel(LightningModule):
    def __init__(self,
                 psg_encoder_name: str = "vit_base",
                 text_encoder_name: str = "google/flan-t5-base",
                 fusion_decoder_name: str = 'cross-attn',
                 shared_emb_dim: int = 256,
                 lr: float = 2e-4,
                 weight_decay: float = 0.2,
                 training_steps_per_epoch: int = 7000,
                 max_epochs: int = 100,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.psg_encoder_name = psg_encoder_name
        self.text_encoder_name = text_encoder_name
        self.fusion_decoder_name = fusion_decoder_name
        self.shared_emb_dim = shared_emb_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_steps_per_epoch = training_steps_per_epoch
        self.max_epochs = max_epochs
        self.warmup_epochs = 0.1 * self.max_epochs
        self.proj_out = shared_emb_dim
        self.proj_hidden = 256

        assert self.training_steps_per_epoch > 1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        total_steps = int(self.training_steps_per_epoch * self.max_epochs)
        warmup_steps = int(round(self.training_steps_per_epoch * self.warmup_epochs))
        warmup_steps = max(0, warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)

        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=decay_steps, eta_min=1e-8)
            sched = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=decay_steps, eta_min=1e-8)

        return [optimizer], [{"scheduler": sched, "interval": "step", "frequency": 1}]

    def training_step(self, batch, batch_idx):
        loss_dict, metrics_dict = self.shared_step(batch, batch_idx)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in metrics_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss_dict, metrics_dict = self.shared_step(batch, batch_idx)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in metrics_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, metrics_dict = self.shared_step(batch, batch_idx)
        for k, v in loss_dict.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in metrics_dict.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict
