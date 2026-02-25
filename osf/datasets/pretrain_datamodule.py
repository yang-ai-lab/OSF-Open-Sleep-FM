import os
from typing import List, Sequence, Optional, Dict, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Subset

from osf.datasets.pretrain_dataset import SleepEpochDataset



class SleepDataModule(LightningDataModule):

    def __init__(
        self,
        csv_dir: str | Path,
        *,
        is_pretrain,
        data_pct = 1,
        val_dataset_list: Optional[List[str]] = None,
        downstream_dataset_name  = None,
        batch_size: int = 128,
        num_workers: int = 4,
        patient_cols: Optional[Union[str, Sequence[str]]] = None,
        event_cols: Optional[Union[str, Sequence[str]]] = None,
        train_edf_cols: Sequence[str] | None,
        transforms=None,
        n_views: int = 1,
        cache_size: int = 8,
        sample_rate: int = 128,
        window_size: int = 30,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        data_source: str = "auto",
        include_datasets: Optional[List[str]] = None,
        regression_targets: Optional[List[str]] = None,
        regression_filter_config: Optional[Dict] = None,
        n_train_samples: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        val_data_pct: Optional[float] = None,
        return_all_event_cols: bool = False,
        return_nsrrid: bool = False,
        random_seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transforms"])  
        self.downstream_dataset_name  = downstream_dataset_name
        self.csv_dir   = csv_dir
        self.transforms = transforms
        self.n_views    = n_views
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.is_pretrain = is_pretrain
        self.patient_cols = patient_cols
        self.event_cols = event_cols
        self.data_pct = data_pct
        self.data_source = data_source
        self.include_datasets = include_datasets
        self.regression_targets = regression_targets
        self.regression_filter_config = regression_filter_config
        self.n_train_samples = n_train_samples
        self.val_batch_size = val_batch_size
        self.val_data_pct = val_data_pct
        self.return_all_event_cols = return_all_event_cols
        self.return_nsrrid = return_nsrrid
        self.random_seed = random_seed

    def train_dataloader(self):
        if self.is_pretrain == 1:
            train_set = SleepEpochDataset(
                    csv_dir       = self.csv_dir,
                    split         = "pretrain",
                    data_pct      = self.data_pct,
                    train_edf_cols= self.hparams.train_edf_cols,
                    transform     = self.transforms,
                    sample_rate   = self.hparams.sample_rate,
                    window_size   = self.hparams.window_size,
                    cache_size    = self.hparams.cache_size,
                    data_source   = self.data_source,
                    include_datasets = self.include_datasets,
                )
            persistent_workers = self.persistent_workers
        else:
            train_set = SleepEpochDataset(
                    csv_dir       = self.csv_dir,
                    split         = "train",
                    data_pct      = self.data_pct,
                    patient_cols  = self.patient_cols,
                    event_cols    = self.event_cols,
                    train_edf_cols= self.hparams.train_edf_cols,
                    transform     = self.transforms,
                    sample_rate   = self.hparams.sample_rate,
                    window_size   = self.hparams.window_size,
                    cache_size    = self.hparams.cache_size,
                    downstream_dataset_name  = self.downstream_dataset_name,
                    data_source   = self.data_source,
                    include_datasets = self.include_datasets,
                    regression_targets = self.regression_targets,
                    regression_filter_config = self.regression_filter_config,
                    return_all_event_cols = self.return_all_event_cols,
                    return_nsrrid = self.return_nsrrid,
                )
            self._train_dataset = train_set
            persistent_workers = True
        
        if self.n_train_samples is not None and self.n_train_samples > 0:
            n_total = len(train_set)
            rng = np.random.default_rng(seed=self.random_seed)
            
            if hasattr(train_set, 'event_cols') and train_set.event_cols and hasattr(train_set, 'all_epoch_df'):
                label_col = train_set.event_cols[0]
                if label_col in train_set.all_epoch_df.columns:
                    labels = train_set.all_epoch_df[label_col].values
                    num_classes = getattr(train_set, 'num_classes', None)
                    
                    if num_classes is not None:
                        all_indices = []
                        for c in range(num_classes):
                            class_indices = np.where(labels == c)[0]
                            n_per_class = min(self.n_train_samples, len(class_indices))
                            if n_per_class > 0:
                                sampled = rng.choice(class_indices, size=n_per_class, replace=False)
                                all_indices.extend(sampled.tolist())
                                print(f"[Few-shot] Class {c}: sampled {n_per_class}/{len(class_indices)} samples")
                        
                        indices = all_indices
                        train_set = Subset(train_set, indices)
                        print(f"[Few-shot] Total: {len(indices)}/{n_total} samples ({self.n_train_samples}-shot per class)")
                    else:
                        n_keep = min(self.n_train_samples, n_total)
                        indices = rng.choice(n_total, size=n_keep, replace=False).tolist()
                        train_set = Subset(train_set, indices)
                        print(f"[Few-shot] Using {n_keep}/{n_total} training samples (random, n_train_samples={self.n_train_samples})")
                else:
                    n_keep = min(self.n_train_samples, n_total)
                    indices = rng.choice(n_total, size=n_keep, replace=False).tolist()
                    train_set = Subset(train_set, indices)
                    print(f"[Few-shot] Using {n_keep}/{n_total} training samples (random, n_train_samples={self.n_train_samples})")
            else:
                n_keep = min(self.n_train_samples, n_total)
                indices = rng.choice(n_total, size=n_keep, replace=False).tolist()
                train_set = Subset(train_set, indices)
                print(f"[Few-shot] Using {n_keep}/{n_total} training samples (random, n_train_samples={self.n_train_samples})")
        
        return DataLoader(
            train_set,
            batch_size     = self.hparams.batch_size,
            shuffle        = True,
            num_workers    = self.hparams.num_workers,
            pin_memory     = self.pin_memory,
            persistent_workers = persistent_workers,
            drop_last      = True,
        )
    
    def get_class_distribution(self) -> Optional[torch.Tensor]:
        """
        Get class distribution from training dataset.
        Returns [num_classes] tensor of class counts, or None if not available.
        """
        if hasattr(self, '_train_dataset'):
            counts = self._train_dataset.get_class_counts()
            if counts is not None:
                return torch.from_numpy(counts).float()
        return None

    def val_dataloader(self):
        if self.hparams.val_dataset_list:       
            if self.is_pretrain == 1:
                val_sets = [
                        SleepEpochDataset(
                            csv_dir       = self.csv_dir,
                            split         = "pretrain-val",
                            data_pct      = self.data_pct,
                            patient_cols   = self.patient_cols,
                            event_cols   = self.event_cols,
                            train_edf_cols= self.hparams.train_edf_cols,
                            transform     = None,        
                            sample_rate   = self.hparams.sample_rate,
                            window_size   = self.hparams.window_size,
                            cache_size    = self.hparams.cache_size,
                            downstream_dataset_name  = ds_name,
                            data_source   = self.data_source,
                            include_datasets = self.include_datasets,
                        )
                        for ds_name in self.hparams.val_dataset_list
                    ]
                persistent_workers = self.persistent_workers
        else:
            if self.is_pretrain == 1:
                val_sets = [
                    SleepEpochDataset(
                        csv_dir       = self.csv_dir,
                        split         = "pretrain-val",
                        data_pct      = self.data_pct,
                        patient_cols   = self.patient_cols,
                        event_cols   = self.event_cols,
                        train_edf_cols= self.hparams.train_edf_cols,
                        transform     = None,
                        sample_rate   = self.hparams.sample_rate,
                        window_size   = self.hparams.window_size,
                        cache_size    = self.hparams.cache_size,
                        data_source   = self.data_source,
                        include_datasets = self.include_datasets,
                    )
                    ]
                persistent_workers = self.persistent_workers
            else:
                val_sets = [
                    SleepEpochDataset(
                        csv_dir       = self.csv_dir,
                        split         = "val",
                        data_pct      = self.data_pct,
                        patient_cols   = self.patient_cols,
                        event_cols   = self.event_cols,
                        train_edf_cols= self.hparams.train_edf_cols,
                        transform     = None,
                        sample_rate   = self.hparams.sample_rate,
                        window_size   = self.hparams.window_size,
                        cache_size    = self.hparams.cache_size,
                        downstream_dataset_name  = self.downstream_dataset_name,
                        data_source   = self.data_source,
                        include_datasets = self.include_datasets,
                        regression_targets = self.regression_targets,
                        regression_filter_config = self.regression_filter_config,
                    )
                    ]
                persistent_workers = True
        
        if self.val_data_pct is not None and 0 < self.val_data_pct < 1.0:
            subsampled_val_sets = []
            for ds in val_sets:
                n_total = len(ds)
                n_keep = max(1, int(n_total * self.val_data_pct))
                rng = np.random.default_rng(seed=self.random_seed)
                indices = rng.choice(n_total, size=n_keep, replace=False).tolist()
                subsampled_val_sets.append(Subset(ds, indices))
                print(f"[Val subsample] Using {n_keep}/{n_total} val samples ({self.val_data_pct*100:.1f}%)")
            val_sets = subsampled_val_sets
        
        val_bs = self.val_batch_size if self.val_batch_size is not None else self.hparams.batch_size
        return [
            DataLoader(
                ds,
                batch_size     = val_bs,
                shuffle        = False,
                num_workers    = self.hparams.num_workers,
                pin_memory     = self.pin_memory,
                persistent_workers = persistent_workers,
                drop_last      = True,
            )
            for ds in val_sets
        ]

    def test_dataloader(self):
        if self.is_pretrain == 1:
            test_set = SleepEpochDataset(
                csv_dir       = self.csv_dir,
                split         = "pretrain-test",
                patient_cols   = self.patient_cols,
                event_cols   = self.event_cols,
                train_edf_cols= self.hparams.train_edf_cols,
                transform     = None,
                sample_rate   = self.hparams.sample_rate,
                window_size   = self.hparams.window_size,
                cache_size    = self.hparams.cache_size,
                data_source   = self.data_source,
                include_datasets = self.include_datasets,
            )
            persistent_workers = self.persistent_workers
        else:
            test_set = SleepEpochDataset(
                    csv_dir       = self.csv_dir,
                    split         = "test",
                    patient_cols   = self.patient_cols,
                    event_cols   = self.event_cols,
                    train_edf_cols= self.hparams.train_edf_cols,
                    transform     = None,
                    sample_rate   = self.hparams.sample_rate,
                    window_size   = self.hparams.window_size,
                    cache_size    = self.hparams.cache_size,
                    downstream_dataset_name  = self.downstream_dataset_name,
                    data_source   = self.data_source,
                    include_datasets = self.include_datasets,
                    regression_targets = self.regression_targets,
                    regression_filter_config = self.regression_filter_config,
                )
            persistent_workers = True
        test_bs = self.val_batch_size if self.val_batch_size is not None else self.hparams.batch_size
        return DataLoader(
            test_set,
            batch_size     = test_bs,
            shuffle        = False,
            num_workers    = self.hparams.num_workers,
            pin_memory     = self.pin_memory,
            drop_last      = True,
            persistent_workers = persistent_workers,
        )

