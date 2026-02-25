from pprint import pprint
import os
from argparse import ArgumentParser, Namespace
import datetime
from dateutil import tz
import random
import numpy as np
import torch
import warnings
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from osf.datasets.pretrain_datamodule import SleepDataModule
from osf.models.dino_model_cls import DINOCLSModel
from config import *
from train_config import *
from osf.models.ssl_finetuner import SSLFineTuner, SSLVitalSignsRegressor
from osf.utils.results_utils import save_results_to_json

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def main(hparams: Namespace):
    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S") + f"_{now.microsecond // 1000:03d}"

    if hparams.monitor_type == "main":
        exp_name = "finetune_12ch"
    else:
        exp_name = f"finetune_{hparams.monitor_type}"

    if hparams.finetune_backbone:
        exp_name = f"{exp_name}_full"

    if hasattr(hparams, 'n_train_samples') and hparams.n_train_samples is not None and hparams.n_train_samples > 0:
        pct_str = f"k{hparams.n_train_samples}"
    elif hparams.train_data_pct < 1:
        pct_str = f"{int(hparams.train_data_pct * 100)}pct"
    else:
        pct_str = "full"
    if hparams.task_type == "classification":
        task_label = hparams.eval_label
    elif hparams.task_type == "regression":
        task_label = "_".join(hparams.regression_targets)
    else:
        raise NotImplementedError(f"Unknown task_type: {hparams.task_type}")
    run_name = f"{task_label}_{hparams.downstream_dataset_name}_{hparams.model_name}_{pct_str}_{timestamp}"
    
    ckpt_dir = os.path.join(
        CKPT_PATH, f"logs/{exp_name}/ckpts/{run_name}")
    os.makedirs(ckpt_dir, exist_ok=True)

    if hparams.task_type == "regression":
        ckpt_monitor = "val_mae"
        ckpt_mode = "min"
    else:
        ckpt_monitor = "val_auc"
        ckpt_mode = "max"
    
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor=ckpt_monitor, dirpath=ckpt_dir,
                        save_last=False, mode=ckpt_mode, save_top_k=1,
                        auto_insert_metric_name=True),
    ]
    if getattr(hparams, 'early_stopping', False):
        early_stop_callback = EarlyStopping(
            monitor=ckpt_monitor,
            patience=getattr(hparams, 'early_stopping_patience', 10),
            mode=ckpt_mode,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        print(f"[INFO] Early stopping enabled: monitor={ckpt_monitor}, patience={hparams.early_stopping_patience}")
    logger_dir = os.path.join(CKPT_PATH, f"logs/{exp_name}")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project=f"{exp_name}_sleepuni", save_dir=logger_dir, name=run_name)
    trainer = Trainer(
        max_steps=hparams.max_steps,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        deterministic=True,
        devices=hparams.num_devices,
        strategy="ddp_find_unused_parameters_true",
        precision=hparams.precision,
        callbacks=callbacks,
        logger=wandb_logger
    )

    hparams.exp_log_dir = os.path.join(
        CKPT_PATH, f"data/{run_name}/exp_logs")
    train_edf_cols = MONITOR_TYPE_MAP.get(hparams.monitor_type, TRAIN_EDF_COLS_UNI_ENC)
    
    if hparams.task_type == "regression":
        event_cols = None
        regression_targets = hparams.regression_targets
        print(f"[INFO] Regression task with targets: {regression_targets}")
    else:  # classification
        event_cols = hparams.eval_label
        regression_targets = None

    regression_filter_config = None
    if hparams.task_type == "regression" and "SPO2" in hparams.regression_targets:
        if hparams.filter_spo2_min is not None or hparams.filter_spo2_max is not None:
            spo2_filter = {}
            if hparams.filter_spo2_min is not None:
                spo2_filter["min"] = hparams.filter_spo2_min
            if hparams.filter_spo2_max is not None:
                spo2_filter["max"] = hparams.filter_spo2_max
            regression_filter_config = {"SPO2_mean": spo2_filter}
            print(f"[INFO] Will filter SPO2_mean with: {spo2_filter}")
    
    datamodule = SleepDataModule(
            is_pretrain    = 0,
            data_pct       = hparams.train_data_pct,
            downstream_dataset_name  = hparams.downstream_dataset_name,
            csv_dir        = SPLIT_DATA_FOLDER,
            train_edf_cols = train_edf_cols,   
            event_cols     = event_cols,
            batch_size     = hparams.batch_size,
            num_workers    = hparams.num_workers,
            sample_rate = hparams.sample_rate,
            window_size = 30,
            data_source = hparams.data_source,
            include_datasets = hparams.include_datasets,
            regression_targets = regression_targets,
            regression_filter_config = regression_filter_config,
            n_train_samples = getattr(hparams, 'n_train_samples', None),
            val_batch_size = getattr(hparams, 'val_batch_size', None),
            val_data_pct = getattr(hparams, 'val_data_pct', None),
            random_seed = hparams.seed,
        )
    if hparams.task_type == "regression":
        hparams.num_classes = len(hparams.regression_targets)  # output dim
        hparams.target_names = hparams.regression_targets
        print(f"[INFO] Regression targets: {hparams.target_names}, num_classes={hparams.num_classes}")
    else:  # classification
        train_dataset = datamodule.train_dataloader().dataset
        if hasattr(train_dataset, 'dataset'):  # It's a Subset
            hparams.num_classes = train_dataset.dataset.num_classes
        else:
            hparams.num_classes = train_dataset.num_classes
        print(f"[INFO] Classification num_classes: {hparams.num_classes}")
    hparams.training_steps_per_epoch = len(datamodule.train_dataloader()) // hparams.accumulate_grad_batches // hparams.num_devices

    if hparams.max_steps > 0:
        hparams.total_training_steps = hparams.max_steps
    else:
        hparams.total_training_steps = hparams.training_steps_per_epoch * hparams.max_epochs
    
    print(f"Total training steps: {hparams.total_training_steps}")
    print(f"Steps per epoch: {hparams.training_steps_per_epoch}")

    class_distribution = datamodule.get_class_distribution()
    if class_distribution is not None:
        print(f"Class distribution: {class_distribution}")
    hparams.class_distribution = class_distribution
    
    # Load pretrained DINO model
    pretrain_model = DINOCLSModel.load_from_checkpoint(hparams.ckpt_path)
    pprint(vars(hparams))

    hparams.epochs = hparams.max_epochs
    
    def create_finetuner(backbones, hparams, train_edf_cols=None):
        exclude_keys = {'train_edf_cols', 'regression_targets'}
        hparams_dict = {k: v for k, v in vars(hparams).items() if k not in exclude_keys}
        
        if hparams.task_type == "regression":
            return SSLVitalSignsRegressor(backbones=backbones, **hparams_dict)
        else:
            return SSLFineTuner(backbones=backbones, **hparams_dict)

    # Extract ViT backbone from DINO model
    vit = pretrain_model.encoders["all"].backbone
    hparams.in_features = vit.width
    print(f"[INFO] Extracted ViT backbone for dino_ours, in_features={hparams.in_features}")
    model = create_finetuner(backbones={"all": vit}, hparams=hparams, train_edf_cols=train_edf_cols)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="last")


if __name__ == '__main__':
    parser = ArgumentParser(description="Fine-tune pretrained model for downstream tasks.")
    parser.add_argument("--model_name", type=str, default="dino_ours")
    parser.add_argument("--eval_label", type=str, default="Stage",
                        )
    parser.add_argument("--downstream_dataset_name", type=str, default="mros",
                        )
    parser.add_argument("--use_which_backbone", type=str, default="all",
                        )
    parser.add_argument("--monitor_type", type=str, default="main",
                        choices=["main", "type3", "type4"],
                        help="Channel configuration: main (12ch), type3 (5ch), type4 (3ch)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--n_train_samples", type=int, default=None,
                        help="If set, use exactly this many training samples (overrides train_data_pct for few-shot)")
    parser.add_argument("--data_source", type=str, default="auto",
                        choices=["auto", "pretrain", "downstream", "both"],
                        help="Which CSV source to use: auto (default), pretrain, downstream, or both")
    parser.add_argument("--include_datasets", type=str, nargs="*", default=None,
                        help="Filter by dataset names, e.g., --include_datasets shhs mros")
    parser.add_argument("--batch_size", type=int, default=800)
    parser.add_argument("--val_batch_size", type=int, default=None,
                        help="Batch size for val/test (defaults to batch_size if not set, useful for few-shot)")
    parser.add_argument("--val_data_pct", type=float, default=None,
                        help="Percentage of val data to use (0-1, useful for few-shot to speed up validation)")
    parser.add_argument("--patch_size_time", type=int, default=64)
    parser.add_argument("--patch_size_ch", type=int, default=4,
                        help="Channel patch size for 2D patchify (default: 4)")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2500)
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping based on val metric (useful for few-shot)")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Patience for early stopping (number of val checks without improvement)")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_features", type=int, default=256)
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal", "balanced_softmax"],
                        help="Loss type: 'ce' (cross-entropy), 'focal' (Focal Loss), or 'balanced_softmax' (Balanced Softmax)")
    parser.add_argument("--focal_gamma", type=float, default=1.0,
                        help="Gamma parameter for Focal Loss (focusing parameter)")
    parser.add_argument("--focal_alpha", type=float, default=None,
                        help="Alpha parameter for Focal Loss (class weighting). If None, computed from class distribution.")
    parser.add_argument("--final_lr", type=float, default=0,
                        help="Final learning rate for cosine annealing scheduler")
    parser.add_argument("--use_mean_pool", action="store_true",
                        help="Use mean pooling of all patches instead of CLS token for feature extraction")
    parser.add_argument("--task_type", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Task type: classification or regression")
    parser.add_argument("--regression_targets", type=str, nargs="*", default=["HR", "SPO2"],
                        help="Regression targets, e.g., --regression_targets HR SPO2")
    parser.add_argument("--filter_spo2_min", type=float, default=None,
                        help="Filter out SPO2 values below this threshold (e.g., 70). Only applies when SPO2 is a regression target.")
    parser.add_argument("--filter_spo2_max", type=float, default=None,
                        help="Filter out SPO2 values above this threshold (e.g., 100). Only applies when SPO2 is a regression target.")
    parser.add_argument("--finetune_backbone", action="store_true",
                        help="If set, finetune the entire backbone (full finetuning); otherwise linear probing only")
    parser.add_argument("--precision", type=str, default="32-true",
                        choices=["32-true", "16-mixed", "bf16-mixed"],
                        help="Training precision: 32-true (full), 16-mixed (FP16), bf16-mixed (BF16)")
    parser.add_argument("--sample_rate", type=int, default=64,
                        help="Input sample rate in Hz (default: 64). Use 32 for half resolution.")
    hparams = parser.parse_args()

    seed_everything(hparams.seed)
    main(hparams)