import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from osf.models.dino_utils.dino_clstoken_loss import DINOLoss
from osf.models.dino_utils.ibot_patch_loss import iBOTPatchLoss
from osf.models.dino_utils.koleo_loss import KoLeoLoss
from osf.models.base_pretrain_model import BasePretrainModel
from osf.models.base_pretrain_model_cls import PSGModalityEncoderCLS
from osf.datasets.simclr_aug_registry import build_simclr_augmentor


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        num_layers = max(nlayers, 1)
        if num_layers == 1:
            self.mlp = nn.Sequential(nn.Linear(in_dim, bottleneck_dim))
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.prototypes = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.prototypes.weight_g.data.fill_(1.0)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.prototypes(x)


class DINOCLSModel(BasePretrainModel):
    def __init__(
        self,
        psg_encoder_name: str = "vit_base",
        text_encoder_name: Optional[str] = None,
        shared_emb_dim: int = 768,
        out_dim: int = 2048,
        patch_out_dim: int = 2048,
        dino_out_dim: int = None,
        dino_patch_out_dim: int = None,
        dino_hidden_dim: int = 2048,
        dino_bottleneck_dim: int = 256,
        student_temp: float = 0.1,
        teacher_temp_warmup: float = 0.04,
        teacher_temp_final: float = 0.07,
        teacher_temp_warmup_iters: int = 10000,
        base_momentum: float = 0.996,
        use_koleo: bool = True,
        koleo_lambda: float = 0.0,
        ibot_lambda: float = 0.0,
        lr: float = 2e-4,
        weight_decay: float = 0.2,
        num_freeze_layers: int = 6,
        simclr_augmentation: dict | None = None,
        n_local_crops: int = 2,
        *args, **kwargs
    ):
        super().__init__(
            psg_encoder_name=psg_encoder_name,
            text_encoder_name=None,
            shared_emb_dim=shared_emb_dim,
            lr=lr,
            weight_decay=weight_decay,
            *args, **kwargs
        )
        self.save_hyperparameters()

        self.proj_out = shared_emb_dim
        self.proj_hidden = 256
        self.num_freeze_layers = num_freeze_layers

        num_leads = kwargs.get('num_leads', 12)
        self.num_leads = num_leads

        self.cfg = [dict(name="all", freq=64, win_sec=30, in_ch=num_leads)]
        self.encoders = nn.ModuleDict()
        for mod in self.cfg:
            self.encoders[mod["name"]] = PSGModalityEncoderCLS(
                encoder_name=psg_encoder_name,
                proj_out=shared_emb_dim,
                proj_hidden=256,
                freq=mod["freq"],
                win_sec=mod["win_sec"],
                channel=mod["in_ch"],
                patch_size=kwargs['patch_size_time'],
                lead_wise=kwargs['lead_wise'],
                patch_size_ch=(num_leads if kwargs['lead_wise'] == 0 else kwargs['patch_size_ch']),
                is_proj_head=0,
            )
        self.lead_wise = kwargs['lead_wise']
        self.patch_size_time = kwargs['patch_size_time']
        self.patch_size_ch = (num_leads if self.lead_wise == 0 else kwargs['patch_size_ch'])
        trunk_dim = self.encoders['all'].backbone.width
        out_dim = dino_out_dim if dino_out_dim is not None else out_dim
        patch_out_dim = dino_patch_out_dim if dino_patch_out_dim is not None else patch_out_dim
        self.out_dim = out_dim
        self.patch_out_dim = patch_out_dim

        self.student_global_head = DINOHead(trunk_dim, out_dim, dino_hidden_dim, dino_bottleneck_dim, 3)
        self.student_patch_head = DINOHead(trunk_dim, patch_out_dim, dino_hidden_dim, dino_bottleneck_dim, 3)
        self.teacher_encoder = copy.deepcopy(self.encoders["all"])
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        self.teacher_encoder.eval()

        self.teacher_global_head = DINOHead(trunk_dim, out_dim, dino_hidden_dim, dino_bottleneck_dim, 3)
        self.teacher_patch_head = DINOHead(trunk_dim, patch_out_dim, dino_hidden_dim, dino_bottleneck_dim, 3)
        self.teacher_global_head.load_state_dict(self.student_global_head.state_dict(), strict=True)
        self.teacher_patch_head.load_state_dict(self.student_patch_head.state_dict(), strict=True)
        for p in self.teacher_global_head.parameters():
            p.requires_grad = False
        for p in self.teacher_patch_head.parameters():
            p.requires_grad = False
        self.teacher_global_head.eval()
        self.teacher_patch_head.eval()
        self.dino_loss = DINOLoss(out_dim=out_dim, student_temp=student_temp, center_momentum=0.9)
        self.ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim, student_temp=student_temp, center_momentum=0.9)
        self.koleo = KoLeoLoss() if use_koleo else None
        self.koleo_lambda = float(koleo_lambda)
        self.ibot_lambda = float(ibot_lambda)
        self.teacher_temp_warmup = float(teacher_temp_warmup)
        self.teacher_temp_final = float(teacher_temp_final)
        self.teacher_temp_warmup_iters = int(teacher_temp_warmup_iters)
        self.base_momentum = float(base_momentum)

        self.register_buffer("seen_steps", torch.tensor(0, dtype=torch.long))

        if simclr_augmentation is None:
            simclr_augmentation = {}
        self.simclr_augmentation = simclr_augmentation
        self.augmentor = build_simclr_augmentor(self.simclr_augmentation)
        self.n_local_crops = int(n_local_crops)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, trunk_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _teacher_temp(self, step: int) -> float:
        if step < self.teacher_temp_warmup_iters:
            alpha = step / float(max(1, self.teacher_temp_warmup_iters))
            return self.teacher_temp_warmup * (1 - alpha) + self.teacher_temp_final * alpha
        return self.teacher_temp_final

    def _momentum(self, step: int, max_steps: int) -> float:
        return 1.0 - (1.0 - self.base_momentum) * (math.cos(math.pi * step / max_steps) + 1) / 2

    @torch.no_grad()
    def _ema_update(self, m: float):
        for param_q, param_k in zip(self.encoders['all'].parameters(), self.teacher_encoder.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
        for param_q, param_k in zip(self.student_global_head.parameters(), self.teacher_global_head.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
        for param_q, param_k in zip(self.student_patch_head.parameters(), self.teacher_patch_head.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
        self.teacher_encoder.eval()
        self.teacher_global_head.eval()
        self.teacher_patch_head.eval()

    def _forward_encoder(self, encoder, x, return_tokens=True):
        # x: [B, C, T]
        if return_tokens:
            cls, patches = encoder.backbone.forward_encoding(x, return_sequence=False)
            return cls, patches  # [B, D], [B, N, D]
        else:
            cls = encoder.backbone(x)
            return cls, None  # [B, D], None

    def _make_views_aug(self, x: torch.Tensor):
        v1, v2 = self.augmentor(x)
        globals_x = [v1, v2]
        locals_x = []
        for _ in range(self.n_local_crops):
            lv1, _ = self.augmentor(x)
            locals_x.append(lv1)
        return globals_x, locals_x

    def shared_step(self, batch, batch_idx):
        x = batch["psg"]
        globals_x, locals_x = self._make_views_aug(x)
        tt = self._teacher_temp(int(self.global_step))

        with torch.no_grad():
            teacher_out_soft_list = []
            teacher_global_logits_cache = []
            teacher_patch_logits_cache = []

            if len(globals_x) > 0:
                g_sizes = [gx.size(0) for gx in globals_x]
                g_cat = torch.cat(globals_x, dim=0)
                cls_t_cat, _ = self._forward_encoder(self.teacher_encoder, g_cat, return_tokens=True)
                g_logits_cat = self.teacher_global_head(cls_t_cat)
                g_logits_split = list(torch.split(g_logits_cat, g_sizes, dim=0))
                teacher_out_soft_list = [self.dino_loss.softmax_center_teacher(gl, tt) for gl in g_logits_split]
                teacher_global_logits_cache = g_logits_split

        student_global_logits = []
        student_cls_tokens = []
        all_student_views = globals_x + locals_x
        if len(all_student_views) > 0:
            s_sizes = [sx.size(0) for sx in all_student_views]
            s_cat = torch.cat(all_student_views, dim=0)
            cls_s_cat, _ = self._forward_encoder(self.encoders["all"], s_cat, return_tokens=False)
            sg_logits_cat = self.student_global_head(cls_s_cat)
            student_global_logits = list(torch.split(sg_logits_cat, s_sizes, dim=0))
            student_cls_tokens = list(torch.split(cls_s_cat, s_sizes, dim=0))

        ibot_loss_val = torch.tensor(0.0, device=x.device)
        if len(globals_x) > 0:
            with torch.no_grad():
                t_tokens, _ = self.teacher_encoder.backbone.to_tokens_2d(
                    globals_x[0], patch_size_ch=self.patch_size_ch, patch_size_time=self.patch_size_time)
                B2 = t_tokens.size(0)
                cls_tok = self.teacher_encoder.backbone.cls_token.expand(B2, -1, -1)
                t_full = torch.cat([cls_tok, t_tokens], dim=1)
                pe_full = self.teacher_encoder.backbone.pos_embedding[:, :t_full.size(1), :].to(t_full.device)
                t_full = t_full + pe_full
                t_full = self.teacher_encoder.backbone._run_blocks(t_full)
                _, t_patches = t_full[:, 0], t_full[:, 1:]
                t_logits_all = self.teacher_patch_head(t_patches)
                t_soft = self.ibot_loss.softmax_center_teacher(t_logits_all, tt)

            s_tokens, _ = self.encoders["all"].backbone.to_tokens_2d(
                globals_x[0], patch_size_ch=self.patch_size_ch, patch_size_time=self.patch_size_time)
            B2, N, Dtok = s_tokens.shape

            mask_ratio = float(getattr(self, "ibot_mask_ratio", 0.3))
            n_mask = max(1, int(round(N * mask_ratio)))
            rand = torch.rand(B2, N, device=x.device)
            topk_idx = rand.topk(k=n_mask, dim=1, largest=True).indices
            masks = torch.zeros(B2, N, dtype=torch.bool, device=x.device)
            masks.scatter_(1, topk_idx, True)

            s_tokens_masked = torch.where(
                masks.unsqueeze(-1),
                self.mask_token.expand_as(s_tokens),
                s_tokens
            )

            cls_tok_s = self.encoders["all"].backbone.cls_token.expand(B2, -1, -1)
            s_full = torch.cat([cls_tok_s, s_tokens_masked], dim=1)
            pe_full_s = self.encoders["all"].backbone.pos_embedding[:, :s_full.size(1), :].to(s_full.device)
            s_full = s_full + pe_full_s
            s_full = self.encoders["all"].backbone._run_blocks(s_full)
            _, s_patches = s_full[:, 0], s_full[:, 1:]
            s_logits_all = self.student_patch_head(s_patches)

            ibot_loss_val = self.ibot_loss.forward_masked(
                student_patch_tokens_masked=s_logits_all[masks],
                teacher_patch_tokens_masked=t_soft[masks],
                student_masks_flat=masks,
            )

            with torch.no_grad():
                teacher_patch_logits_cache.append(t_logits_all)

        dino_loss_val = self.dino_loss(student_global_logits, teacher_out_soft_list)
        pair_norm = max(1, len(student_global_logits) * len(teacher_out_soft_list))
        dino_loss_val = dino_loss_val / pair_norm
        koleo_val = torch.tensor(0.0, device=x.device)
        if self.koleo is not None and len(student_cls_tokens) > 0:
            koleo_val = self.koleo(F.normalize(student_cls_tokens[0], dim=-1))

        total_loss = dino_loss_val + self.ibot_lambda * ibot_loss_val + self.koleo_lambda * koleo_val

        with torch.no_grad():
            if self.training:
                if len(teacher_global_logits_cache) > 0:
                    self.dino_loss.update_center(torch.cat(teacher_global_logits_cache, dim=0))
                if len(teacher_patch_logits_cache) > 0:
                    self.ibot_loss.update_center(torch.cat(teacher_patch_logits_cache, dim=0))

        metrics = {
            "loss": total_loss,
            "loss/dino": dino_loss_val,
            "loss/ibot": ibot_loss_val,
            "loss/koleo": koleo_val,
            "sched/teacher_temp": torch.tensor(tt, device=x.device),
        }
        return {"loss": total_loss}, metrics

    def training_step(self, batch, batch_idx):
        loss_dict, metrics = self.shared_step(batch, batch_idx)
        for k, v in metrics.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=(k == "loss"), sync_dist=True)
        return loss_dict["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        max_steps = max(1, getattr(self.trainer, "max_steps", getattr(self.trainer, "estimated_stepping_batches", 100000)))
        m = self._momentum(int(self.global_step), max_steps)
        self._ema_update(m)
        self.log("sched/momentum", torch.tensor(m, device=self.device), on_step=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        loss_dict, metrics = self.shared_step(batch, batch_idx)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, on_step=True, on_epoch=True, prog_bar=(k == "loss"), sync_dist=True)
        return loss_dict["loss"]
