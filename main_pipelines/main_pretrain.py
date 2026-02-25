from pprint import pprint
import os
from argparse import ArgumentParser, Namespace
import datetime
from dateutil import tz
import random
import numpy as np
import torch
import warnings
from datetime import timedelta
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy


class DenseStepCheckpoint(Callback):
    """Save checkpoints at specific training steps."""
    
    def __init__(self, dirpath: str, save_steps: list = None):
        super().__init__()
        self.dirpath = dirpath
        self.save_steps = set(save_steps) if save_steps else {1, 10, 100, 1000, 10000, 100000}
        self.saved_steps = set()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step in self.save_steps and global_step not in self.saved_steps:
            ckpt_path = os.path.join(self.dirpath, f"step={global_step}.ckpt")
            trainer.save_checkpoint(ckpt_path)
            self.saved_steps.add(global_step)
            if trainer.is_global_zero:
                print(f"[DenseStepCheckpoint] Saved checkpoint at step {global_step}: {ckpt_path}")

from osf.datasets.pretrain_datamodule import SleepDataModule
from osf.models.dino_model_cls import DINOCLSModel
from config import *
from train_config import *

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.optimize_ddp = False



def param_stats(model: torch.nn.Module, verbose: bool = False):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"{'Name':40s} {'Shape':20s} {'#Params':>10s} {'Train?':>6s}")
        print("-" * 80)
        for name, p in model.named_parameters():
            print(f"{name:40s} {str(list(p.shape)):20s} {p.numel():10d} {str(p.requires_grad):>6s}")
        print("-" * 80)
    print(f"Total parameters:     {total / 1e6:.3f} M ({total})")
    print(f"  Trainable params:   {trainable / 1e6:.3f} M ({trainable})")
    print(f"  Frozen params:      {(total-trainable) / 1e6:.3f} M ({total-trainable})")
def main(hparams: Namespace):

    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = f"final_sleep_unimodal_{hparams.model_name}_{hparams.psg_encoder_name}_bz{hparams.batch_size}_{extension}"
    ckpt_dir = os.path.join(
        CKPT_PATH, f"logs/sleepuni/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    if hparams.model_name in MODEL_LIST:
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="val/loss", dirpath=ckpt_dir,
                            save_last=True, every_n_epochs=2, mode="min", save_top_k=-1,
                            save_on_train_epoch_end=False, auto_insert_metric_name=True),
        ]
        if hparams.dense_ckpt:
            dense_ckpt_dir = os.path.join(ckpt_dir, "dense_steps")
            os.makedirs(dense_ckpt_dir, exist_ok=True)
            callbacks.append(DenseStepCheckpoint(
                dirpath=dense_ckpt_dir,
                save_steps=hparams.dense_ckpt_steps
            ))
    else:
        raise NotImplementedError
    logger_dir = os.path.join(CKPT_PATH, "logs/sleepuni")
    os.makedirs(logger_dir, exist_ok=True)
    print("wandb logger dir: ", logger_dir)
    wandb_logger = WandbLogger(
        project=hparams.wandb_proj_name + f'final_{hparams.model_name}_{hparams.psg_encoder_name}_bz{hparams.batch_size}', save_dir=logger_dir, name=extension)

    strategy = DDPStrategy(
        find_unused_parameters=True,
        static_graph=False,
        timeout=timedelta(minutes=15),
    )

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        devices=hparams.num_devices,
        num_nodes=hparams.num_nodes,
        precision=hparams.precision,
        gradient_clip_val=3.0,
        gradient_clip_algorithm="norm",
        strategy=strategy,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    hparams.exp_log_dir = os.path.join(
        CKPT_PATH, f"data/{extension}/exp_logs")
    train_edf_cols = MONITOR_TYPE_MAP.get(hparams.monitor_type, TRAIN_EDF_COLS_UNI_ENC)
    hparams.num_leads = len(train_edf_cols)
    
    dm = SleepDataModule(
            is_pretrain    = 1,
            csv_dir        = SPLIT_DATA_FOLDER,
            train_edf_cols = train_edf_cols,
            batch_size     = hparams.batch_size,
            num_workers    = hparams.num_workers,
            data_pct       = hparams.train_data_pct,
            window_size = 30,
            sample_rate = 64,
            val_dataset_list = hparams.val_dataset_list,
            data_source = hparams.data_source,
            include_datasets = hparams.include_datasets,
        )

    hparams.simclr_augmentation = AUGMENTATION_MAP.get(hparams.model_name, "none")

    # Create DINO model
    model = DINOCLSModel(**vars(hparams))
    model.training_steps_per_epoch = len(dm.train_dataloader()) // hparams.accumulate_grad_batches // hparams.num_devices
    model.teacher_temp_warmup_iters = model.training_steps_per_epoch * 0.1 * hparams.max_epochs
    print(f"[INFO] DINO teacher warmup steps: {model.teacher_temp_warmup_iters}")
    pprint(vars(hparams))

    if hparams.ckpt_path:
        trainer.fit(model, datamodule = dm, ckpt_path=hparams.ckpt_path)
    else:
        trainer.fit(model, datamodule = dm)


if __name__ == '__main__':
    parser = ArgumentParser(description="Pretraining DINO model for sleep PSG data.")
    parser.add_argument("--model_name", type=str, default="dino_ours",
                        choices=MODEL_LIST)
    
    parser.add_argument("--psg_encoder_name", type=str, default="vit_base")
    parser.add_argument("--val_dataset_list", default=PRETRAIN_VAL_DATASET_LIST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--data_source", type=str, default="auto",
                        choices=["auto", "pretrain", "downstream", "both"])
    parser.add_argument("--include_datasets", type=str, nargs="*", default=None)
    parser.add_argument("--monitor_type", type=str, default="main",
                        choices=["main", "type3", "type4"],
                        help="Channel configuration: main (12ch), type3 (5ch), type4 (3ch)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size_time", type=int, default=4)
    parser.add_argument("--patch_size_ch", type=int, default=4)
    parser.add_argument("--use_2d_pos_embed", type=bool, default=True)
    parser.add_argument("--sample_rate", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--num_devices", type=int, default=4)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--text_encoder_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--lead_wise", type=int, default=0)
    parser.add_argument("--use_lead_embedding", type=int, default=1)
    # DINO-specific args
    parser.add_argument("--koleo_lambda", type=float, default=0.0)
    parser.add_argument("--ibot_lambda", type=float, default=0.0)
    parser.add_argument("--dino_out_dim", type=int, default=2048)
    parser.add_argument("--dino_patch_out_dim", type=int, default=2048)
    parser.add_argument("--dino_hidden_dim", type=int, default=2048)
    parser.add_argument("--dino_bottleneck_dim", type=int, default=256)
    parser.add_argument("--wandb_proj_name", type=str, default="sleepuni")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--dense_ckpt", action="store_true")
    parser.add_argument("--dense_ckpt_steps", type=int, nargs="+", default=[10, 100, 200, 400, 500, 800, 1000, 1600, 2500, 3200, 6400, 10000, 12500, 12800, 25600, 51200, 62500, 100000])
    


    hparams = parser.parse_args()
    
    seed_everything(hparams.seed)
    main(hparams)