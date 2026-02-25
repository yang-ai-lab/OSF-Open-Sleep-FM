from typing import Tuple, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from einops import rearrange
from itertools import chain
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix, CohenKappa, AveragePrecision, MetricCollection
from osf.models.balanced_losses import FocalLoss, BalancedSoftmax


def _create_pred_metrics(num_classes: int) -> MetricCollection:
    """Create metrics that take preds (class indices) as input."""
    metrics = {
        "acc": Accuracy(task="multiclass", num_classes=num_classes, average="micro"),
        "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        "f1_w": F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
        "rec_m": Recall(task="multiclass", num_classes=num_classes, average="macro"),
        "kappa": CohenKappa(task="multiclass", num_classes=num_classes, weights="quadratic"),
    }
    return MetricCollection(metrics)


def _create_prob_metrics(num_classes: int) -> MetricCollection:
    """Create metrics that take probs (probabilities) as input."""
    metrics = {
        "auc": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
        "auprc": AveragePrecision(task="multiclass", num_classes=num_classes, average="macro"),
    }
    return MetricCollection(metrics)


def _create_perclass_pred_metrics(num_classes: int) -> MetricCollection:
    """Create per-class metrics that take preds as input."""
    metrics = {
        "acc_c": Accuracy(task="multiclass", num_classes=num_classes, average=None),
        "prec_c": Precision(task="multiclass", num_classes=num_classes, average=None),
        "rec_c": Recall(task="multiclass", num_classes=num_classes, average=None),
        "f1_c": F1Score(task="multiclass", num_classes=num_classes, average=None),
        "cm": ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize=None),
    }
    return MetricCollection(metrics)


def _create_perclass_prob_metrics(num_classes: int) -> MetricCollection:
    """Create per-class metrics that take probs as input."""
    metrics = {
        "auc_c": AUROC(task="multiclass", num_classes=num_classes, average=None),
        "auprc_c": AveragePrecision(task="multiclass", num_classes=num_classes, average=None),
    }
    return MetricCollection(metrics)



class SSLFineTuner(LightningModule):
    def __init__(self,
        backbones,
        use_which_backbone,
        config = None,
        in_features: int = 256,
        num_classes: int = 2,
        epochs: int = 10,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        final_lr: float = 1e-5,
        use_channel_bank: bool = True,
        loss_type: str = "ce",
        class_distribution: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float | torch.Tensor] = None,
        use_mean_pool: bool = False,
        total_training_steps: int = None,
        finetune_backbone: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.final_lr = final_lr
        self.use_channel_bank = use_channel_bank
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.use_mean_pool = use_mean_pool
        self.total_training_steps = total_training_steps
        self.finetune_backbone = finetune_backbone
        
        if loss_type == "ce":
            self.criterion = None 
        elif loss_type == "focal":
            alpha = focal_alpha
            if alpha is None and class_distribution is not None:
                class_dist = class_distribution.float()
                total_samples = class_dist.sum()
                alpha = total_samples / (num_classes * class_dist)
                alpha = alpha / alpha.mean()
            self.criterion = FocalLoss(alpha=alpha, gamma=focal_gamma, reduction="mean")
        elif loss_type == "balanced_softmax":
            self.criterion = BalancedSoftmax(class_distribution, reduction="mean")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be one of ['ce', 'focal', 'balanced_softmax']")

        if isinstance(backbones, nn.ModuleDict):
            self.backbones = backbones
        else:
            self.backbones = nn.ModuleDict(backbones)
        self.config = config
        self.use_which_backbone = use_which_backbone
        self.backbone = self.backbones[self.use_which_backbone] if self.use_which_backbone != "fusion" else None


        if self.use_which_backbone == "fusion":
            for k in ("ecg", "resp", "elect"):
                if k in self.backbones:
                    for p in self.backbones[k].parameters():
                        p.requires_grad = self.finetune_backbone
                    if not self.finetune_backbone:
                        self.backbones[k].eval()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = self.finetune_backbone
            if not self.finetune_backbone:
                self.backbone.eval()
        
        if self.finetune_backbone:
            print(f"[INFO] Full finetuning mode: backbone parameters are TRAINABLE")

        if self.use_which_backbone == "fusion":
            dims = [getattr(self.backbones[k], "out_dim", in_features)
                    for k in ("ecg", "resp", "elect") if k in self.backbones]
            if len(dims) == 0:
                raise ValueError("fusion requires at least one of {'ecg','resp','elect'} in backbones.")
            if len(set(dims)) != 1:
                raise ValueError(f"Mean fusion requires equal output dims, got {dims}")
            final_in_features = dims[0]
        else:
            final_in_features = getattr(self.backbone, "out_dim", in_features)

        self.linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_in_features, num_classes)
        )

        self.train_pred_metrics = _create_pred_metrics(num_classes)
        self.val_pred_metrics = _create_pred_metrics(num_classes)
        self.test_pred_metrics = _create_pred_metrics(num_classes)

        self.train_prob_metrics = _create_prob_metrics(num_classes)
        self.val_prob_metrics = _create_prob_metrics(num_classes)
        self.test_prob_metrics = _create_prob_metrics(num_classes)

        self.train_pred_metrics_c = _create_perclass_pred_metrics(num_classes)
        self.val_pred_metrics_c = _create_perclass_pred_metrics(num_classes)
        self.test_pred_metrics_c = _create_perclass_pred_metrics(num_classes)

        self.train_prob_metrics_c = _create_perclass_prob_metrics(num_classes)
        self.val_prob_metrics_c = _create_perclass_prob_metrics(num_classes)
        self.test_prob_metrics_c = _create_perclass_prob_metrics(num_classes)


        self.class_names = getattr(self.config, "class_names", [str(i) for i in range(num_classes)])

    def on_train_epoch_start(self) -> None:
        if not self.finetune_backbone:
            if self.use_which_backbone == "fusion":
                for k in ("ecg", "resp", "elect"):
                    if k in self.backbones:
                        self.backbones[k].eval()
            else:
                self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        probs = logits.softmax(-1)
        preds = logits.argmax(-1)

        self.train_pred_metrics.update(preds, y)
        self.train_prob_metrics.update(probs, y)
        self.train_pred_metrics_c.update(preds, y)
        self.train_prob_metrics_c.update(probs, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        pred_agg = self.train_pred_metrics.compute()
        prob_agg = self.train_prob_metrics.compute()
        
        self.log("train_acc", pred_agg["acc"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_f1", pred_agg["f1"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_auc", prob_agg["auc"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_auprc", prob_agg["auprc"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        pred_c = self.train_pred_metrics_c.compute()
        prob_c = self.train_prob_metrics_c.compute()
        cm = pred_c["cm"]
        support = cm.sum(dim=1) if cm is not None else None

        for i in range(len(pred_c["acc_c"])):
            name = self.class_names[i] if i < len(self.class_names) else str(i)
            self.log(f"train/acc_{name}", pred_c["acc_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"train/prec_{name}", pred_c["prec_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"train/rec_{name}", pred_c["rec_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"train/f1_{name}", pred_c["f1_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"train/auc_{name}", prob_c["auc_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"train/auprc_{name}", prob_c["auprc_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if support is not None:
                self.log(f"train/support_{name}", support[i].to(pred_c["acc_c"][i].dtype), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.train_pred_metrics.reset()
        self.train_prob_metrics.reset()
        self.train_pred_metrics_c.reset()
        self.train_prob_metrics_c.reset()

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        probs = logits.softmax(-1)
        preds = logits.argmax(-1)

        self.val_pred_metrics.update(preds, y)
        self.val_prob_metrics.update(probs, y)
        self.val_pred_metrics_c.update(preds, y)
        self.val_prob_metrics_c.update(probs, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        pred_agg = self.val_pred_metrics.compute()
        prob_agg = self.val_prob_metrics.compute()
        
        self.log("val_acc", pred_agg["acc"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_f1", pred_agg["f1"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_f1_w", pred_agg["f1_w"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_rec_m", pred_agg["rec_m"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_auc", prob_agg["auc"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_auprc", prob_agg["auprc"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_kappa", pred_agg["kappa"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        pred_c = self.val_pred_metrics_c.compute()
        prob_c = self.val_prob_metrics_c.compute()
        cm = pred_c["cm"]
        support = cm.sum(dim=1)

        for i in range(len(pred_c["acc_c"])):
            name = self.class_names[i] if i < len(self.class_names) else str(i)
            self.log(f"val/acc_{name}", pred_c["acc_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"val/prec_{name}", pred_c["prec_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"val/rec_{name}", pred_c["rec_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"val/f1_{name}", pred_c["f1_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"val/auc_{name}", prob_c["auc_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"val/auprc_{name}", prob_c["auprc_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"val/support_{name}", support[i].to(pred_c["acc_c"][i].dtype), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.val_pred_metrics.reset()
        self.val_prob_metrics.reset()
        self.val_pred_metrics_c.reset()
        self.val_prob_metrics_c.reset()

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        probs = logits.softmax(-1)
        preds = logits.argmax(-1)

        self.test_pred_metrics.update(preds, y)
        self.test_prob_metrics.update(probs, y)
        self.test_pred_metrics_c.update(preds, y)
        self.test_prob_metrics_c.update(probs, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        pred_agg = self.test_pred_metrics.compute()
        prob_agg = self.test_prob_metrics.compute()
        
        self.log("test_acc", pred_agg["acc"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_f1", pred_agg["f1"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_f1_w", pred_agg["f1_w"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_rec_m", pred_agg["rec_m"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_auc", prob_agg["auc"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_auprc", prob_agg["auprc"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_kappa", pred_agg["kappa"], on_step=False, on_epoch=True, sync_dist=True)

        pred_c = self.test_pred_metrics_c.compute()
        prob_c = self.test_prob_metrics_c.compute()
        cm = pred_c["cm"]
        support = cm.sum(dim=1) if cm is not None else None

        for i in range(len(pred_c["acc_c"])):
            name = self.class_names[i] if i < len(self.class_names) else str(i)
            self.log(f"test/acc_{name}", pred_c["acc_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"test/prec_{name}", pred_c["prec_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"test/rec_{name}", pred_c["rec_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"test/f1_{name}", pred_c["f1_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"test/auc_{name}", prob_c["auc_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"test/auprc_{name}", prob_c["auprc_c"][i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if support is not None:
                self.log(f"test/support_{name}", support[i].to(pred_c["acc_c"][i].dtype),
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.test_pred_metrics.reset()
        self.test_prob_metrics.reset()
        self.test_pred_metrics_c.reset()
        self.test_prob_metrics_c.reset()
    def shared_step(self, batch):
        context = torch.no_grad() if not self.finetune_backbone else torch.enable_grad()
        
        with context:
            psg = batch['psg']
            if self.use_which_backbone == 'ecg':
                x = psg[:, 0:1, :]
                feats = self._get_features(self.backbone, x)
                
            elif self.use_which_backbone == 'resp':
                x = psg[:, 1:5, :]
                feats = self._get_features(self.backbone, x)
            elif self.use_which_backbone == 'elect':
                x = psg[:, 5:, :]
                feats = self._get_features(self.backbone, x)
            elif self.use_which_backbone == 'all':

                x = psg
                feats = self._get_features(self.backbone, x)
                
            elif self.use_which_backbone == 'fusion':
                feats_list = []
                if 'ecg' in self.backbones:
                    x_ecg = psg[:, 0:1, :]
                    f_ecg = self._get_features(self.backbones['ecg'], x_ecg)
                    feats_list.append(f_ecg)
                if 'resp' in self.backbones:
                    x_resp = psg[:, 1:5, :]
                    f_resp = self._get_features(self.backbones['resp'], x_resp)
                    feats_list.append(f_resp)
                if 'elect' in self.backbones:
                    x_elect = psg[:, 5:, :]
                    f_elect = self._get_features(self.backbones['elect'], x_elect)
                    feats_list.append(f_elect)


                feats = torch.stack(feats_list, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown use_which_backbone: {self.use_which_backbone}")

        y = batch["label"]
        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        y = y.squeeze(1).long()

        if self.criterion is None:
            loss = F.cross_entropy(logits, y)
        else:
            loss = self.criterion(logits, y)
        
        return loss, logits, y
    
    def _get_features(self, backbone, x):
        """Get features from backbone. Uses mean pooling if use_mean_pool=True."""
        if self.use_mean_pool:
            if hasattr(backbone, 'forward_encoding_mean_pool'):
                return backbone.forward_encoding_mean_pool(x)
            elif hasattr(backbone, 'forward_avg_pool'):
                return backbone.forward_avg_pool(x)
        return backbone(x)


    def configure_optimizers(self):
        if self.finetune_backbone:
            if self.use_which_backbone == "fusion":
                backbone_params = chain(*[self.backbones[k].parameters() 
                                          for k in ("ecg", "resp", "elect") if k in self.backbones])
            else:
                backbone_params = self.backbone.parameters()
            params = chain(backbone_params, self.linear_layer.parameters())
        else:
            params = self.linear_layer.parameters()
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.total_training_steps is not None and self.total_training_steps > 0:
            warmup_steps = int(0.1 * self.total_training_steps)
            cosine_steps = self.total_training_steps - warmup_steps
            
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=cosine_steps, 
                eta_min=self.final_lr
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return [optimizer]



class SSLVitalSignsRegressor(SSLFineTuner):
    """SSL Finetuner for vital signs regression (HR, SPO2). Uses MSE loss."""
    def __init__(self,
        backbones,
        use_which_backbone,
        config = None,
        in_features: int = 256,
        num_classes: int = 1,
        target_names: list = None,
        dropout: float = 0.0,
        **kwargs
    ) -> None:
        kwargs['loss_type'] = 'ce'

        super().__init__(
            backbones=backbones,
            use_which_backbone=use_which_backbone,
            config=config,
            in_features=in_features,
            num_classes=2,
            dropout=dropout,
            **kwargs
        )

        self.num_targets = num_classes
        self.target_names = target_names or [f"target_{i}" for i in range(num_classes)]
        self.criterion = nn.MSELoss()

        in_feat = self.linear_layer[1].in_features
        self.linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, num_classes)
        )

        del self.train_pred_metrics, self.val_pred_metrics, self.test_pred_metrics
        del self.train_prob_metrics, self.val_prob_metrics, self.test_prob_metrics
        del self.train_pred_metrics_c, self.val_pred_metrics_c, self.test_pred_metrics_c
        del self.train_prob_metrics_c, self.val_prob_metrics_c, self.test_prob_metrics_c

    def shared_step(self, batch):
        """Override: regression loss instead of classification."""
        context = torch.no_grad() if not self.finetune_backbone else torch.enable_grad()
        
        with context:
            psg = batch['psg']
            if self.use_which_backbone == 'ecg':
                x = psg[:, 0:1, :]
                feats = self._get_features(self.backbone, x)
            elif self.use_which_backbone == 'resp':
                x = psg[:, 1:5, :]
                feats = self._get_features(self.backbone, x)
            elif self.use_which_backbone == 'elect':
                x = psg[:, 5:, :]
                feats = self._get_features(self.backbone, x)
            elif self.use_which_backbone == 'all':
                x = psg
                feats = self._get_features(self.backbone, x)
            elif self.use_which_backbone == 'fusion':
                feats_list = []
                if 'ecg' in self.backbones:
                    f_ecg = self._get_features(self.backbones['ecg'], psg[:, 0:1, :])
                    feats_list.append(f_ecg)
                if 'resp' in self.backbones:
                    f_resp = self._get_features(self.backbones['resp'], psg[:, 1:5, :])
                    feats_list.append(f_resp)
                if 'elect' in self.backbones:
                    f_elect = self._get_features(self.backbones['elect'], psg[:, 5:, :])
                    feats_list.append(f_elect)
                
                feats = torch.stack(feats_list, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown use_which_backbone: {self.use_which_backbone}")
        
        y = batch["label"].float()  # [B, num_targets]
        feats = feats.view(feats.size(0), -1)
        preds = self.linear_layer(feats)  # [B, num_targets]
        
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        """Override: regression metrics."""
        loss, preds, y = self.shared_step(batch)
        
        with torch.no_grad():
            for i, name in enumerate(self.target_names):
                mae = F.l1_loss(preds[:, i], y[:, i])
                self.log(f"train_{name}_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        """Override: no classification metrics to compute."""
        pass

    def validation_step(self, batch, batch_idx):
        """Override: regression metrics."""
        loss, preds, y = self.shared_step(batch)
        
        for i, name in enumerate(self.target_names):
            mae = F.l1_loss(preds[:, i], y[:, i])
            self.log(f"val_{name}_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        
        overall_mae = F.l1_loss(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_mae", overall_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        """Override: no classification metrics to compute."""
        pass

    def test_step(self, batch, batch_idx):
        """Override: regression metrics."""
        loss, preds, y = self.shared_step(batch)
        
        for i, name in enumerate(self.target_names):
            p, t = preds[:, i], y[:, i]
            mae = F.l1_loss(p, t)
            mse = F.mse_loss(p, t)
            rmse = torch.sqrt(mse)
            
            self.log(f"test_{name}_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test_{name}_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test_{name}_rmse", rmse, on_step=False, on_epoch=True, sync_dist=True)
        
        overall_mae = F.l1_loss(preds, y)
        overall_mse = F.mse_loss(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_mae", overall_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_mse", overall_mse, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        """Override: no classification metrics to compute."""
        pass


class SupervisedVitalSignsRegressor(SSLVitalSignsRegressor):
    """Supervised from-scratch regression. Equivalent to SSLVitalSignsRegressor with finetune_backbone=True."""
    def __init__(self, 
        backbones,
        use_which_backbone,
        epochs: int = 100,
        **kwargs
    ):
        kwargs['finetune_backbone'] = True
        super().__init__(
            backbones=backbones,
            use_which_backbone=use_which_backbone,
            epochs=epochs,
            **kwargs
        )
