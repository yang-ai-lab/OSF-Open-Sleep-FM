# Sleep Epoch Dataset for pretraining and downstream tasks

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from contextlib import suppress
from typing import Sequence, Optional, Dict, Union, List
from torch.utils.data import Dataset
from train_config import NEED_NORM_COL


def to_pm1(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    vmin, vmax = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax <= vmin:
        return pd.Series(0.0, index=s.index)
    return (2 * (s - vmin) / (vmax - vmin) - 1).fillna(0.0)


class SleepEpochDataset(Dataset):
    def __init__(
        self,
        csv_dir='/Yourpath',
        split: str = "train",
        *,
        data_pct=1,
        patient_cols: Optional[Union[str, Sequence[str]]] = None,
        event_cols: Optional[Union[str, Sequence[str]]] = None,
        train_edf_cols=None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 1337,
        sample_rate: int = 128,
        window_size: int = 300,
        epoch_length: int = 30,
        cache_size: int = 8,
        transform=None,
        downstream_dataset_name=None,
        data_source: str = "auto",
        include_datasets: Optional[List[str]] = None,
        regression_targets: Optional[List[str]] = None,
        regression_filter_config: Optional[Dict] = None,
        return_all_event_cols: bool = False,
        return_nsrrid: bool = False,
    ):
        assert split in {"pretrain", "pretrain-val", "pretrain-test", "train", "val", "test"}
        assert data_source in {"auto", "pretrain", "downstream", "both"}

        self.transform = transform
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.epoch_length = epoch_length
        self.patient_cols = [patient_cols] if isinstance(patient_cols, str) else patient_cols
        self.event_cols = [event_cols] if isinstance(event_cols, str) else event_cols
        self.train_edf_cols = train_edf_cols
        self.split = split
        self.data_pct = float(data_pct)
        self.data_source = data_source
        self.regression_targets = regression_targets
        self.regression_filter_config = regression_filter_config
        self.return_all_event_cols = return_all_event_cols
        self.return_nsrrid = return_nsrrid

        patient_df, epoch_df = self._load_csvs(
            csv_dir, split, data_source, include_datasets, self.event_cols,
            regression_targets=self.regression_targets,
            regression_filter_config=self.regression_filter_config,
            return_all_event_cols=self.return_all_event_cols,
        )

        if downstream_dataset_name and include_datasets is None:
            if downstream_dataset_name != "all":
                mask = epoch_df['dataset_name'].astype(str).str.lower().str.startswith(downstream_dataset_name)
                epoch_df = epoch_df.loc[mask].copy()
                ids = epoch_df["nsrrid"].astype(str).unique()
                patient_df = patient_df[patient_df["nsrrid"].astype(str).isin(ids)].copy()

        # Determine num_classes
        if self.event_cols:
            if self.event_cols[0] in ['Hypopnea', 'Arousal', 'Oxygen Desaturation']:
                self.num_classes = 2
            elif self.event_cols[0] == 'Stage':
                self.num_classes = 4
                mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
                epoch_df['Stage'] = epoch_df['Stage'].replace(mapping)
            else:
                self.num_classes = 2
        else:
            self.num_classes = 2

        # Drop Stage == -1
        if self.event_cols and ('Stage' in self.event_cols) and ('Stage' in epoch_df.columns):
            epoch_df = epoch_df.loc[epoch_df['Stage'] != -1].copy()

        # Build tables
        if split in ("pretrain", "pretrain-val"):
            sort_cols = [c for c in ['nsrrid', 'seg_id', 'epoch_id'] if c in epoch_df.columns]
            self.all_epoch_df = epoch_df.sort_values(sort_cols).reset_index(drop=True)

            idx_keep_cols = [c for c in ['nsrrid', 'seg_id', 'path_head'] if c in self.all_epoch_df.columns]
            if self.regression_targets:
                for t in self.regression_targets:
                    col = f"{t}_mean"
                    if col in self.all_epoch_df.columns:
                        idx_keep_cols.append(col)
            self.epoch_df = (
                self.all_epoch_df[idx_keep_cols]
                .drop_duplicates(['nsrrid', 'seg_id'], keep='first')
                .reset_index(drop=True)
            )
        else:
            expected_len = self.window_size // self.epoch_length
            grp = epoch_df.groupby(['nsrrid', 'seg_id']).size().rename('n').reset_index()
            valid_keys = grp.loc[grp['n'] == expected_len, ['nsrrid', 'seg_id']]
            epoch_df_valid = epoch_df.merge(valid_keys, on=['nsrrid', 'seg_id'], how='inner')

            sort_cols = [c for c in ['nsrrid', 'seg_id', 'epoch_id'] if c in epoch_df_valid.columns]
            self.all_epoch_df = epoch_df_valid.sort_values(sort_cols).reset_index(drop=True)

            idx_keep_cols = [c for c in ['nsrrid', 'seg_id', 'path_head'] if c in self.all_epoch_df.columns]
            if self.regression_targets:
                for t in self.regression_targets:
                    col = f"{t}_mean"
                    if col in self.all_epoch_df.columns:
                        idx_keep_cols.append(col)
            self.epoch_df = (
                self.all_epoch_df[idx_keep_cols]
                .drop_duplicates(['nsrrid', 'seg_id'], keep='first')
                .reset_index(drop=True)
            )

        # Patient-level sampling
        if not (0 < self.data_pct <= 1.0):
            raise ValueError(f"data_pct must be in (0,1], got {self.data_pct}")

        if self.data_pct < 1.0:
            eligible_patients = pd.Index(self.epoch_df['nsrrid'].unique())
            n_keep = max(1, int(len(eligible_patients) * self.data_pct))
            sampled_nsrrids = pd.Series(eligible_patients).sample(n=n_keep, random_state=random_state).to_list()
            self.epoch_df = self.epoch_df.loc[self.epoch_df['nsrrid'].isin(sampled_nsrrids)].reset_index(drop=True)
            self.all_epoch_df = self.all_epoch_df.loc[self.all_epoch_df['nsrrid'].isin(sampled_nsrrids)].reset_index(drop=True)
            patient_df = patient_df.loc[patient_df['nsrrid'].isin(sampled_nsrrids)].copy()

        self.patient_df = patient_df.set_index("nsrrid")

        # Build segment indices
        self._seg_indices = None
        if hasattr(self, "all_epoch_df") and {'nsrrid', 'seg_id'}.issubset(self.all_epoch_df.columns):
            grp_indices = self.all_epoch_df.groupby(['nsrrid', 'seg_id'], sort=False).indices
            self._seg_indices = {}
            has_epoch_id = 'epoch_id' in self.all_epoch_df.columns
            epoch_id_values = self.all_epoch_df['epoch_id'].to_numpy() if has_epoch_id else None
            for key, idx_list in grp_indices.items():
                idx_arr = np.fromiter(idx_list, dtype=np.int64)
                if has_epoch_id:
                    order = np.argsort(epoch_id_values[idx_arr])
                    idx_arr = idx_arr[order]
                self._seg_indices[key] = idx_arr

        # Compute class distribution
        self._class_counts = None
        if self.event_cols and self.event_cols[0] in self.all_epoch_df.columns:
            label_col = self.event_cols[0]
            value_counts = self.all_epoch_df[label_col].value_counts().sort_index()
            class_counts = np.zeros(self.num_classes, dtype=np.int64)
            for cls_idx, count in value_counts.items():
                if 0 <= int(cls_idx) < self.num_classes:
                    class_counts[int(cls_idx)] = int(count)
            self._class_counts = class_counts

    def _load_csvs(self, csv_dir, split, data_source, include_datasets, event_cols,
                   regression_targets=None, regression_filter_config=None, return_all_event_cols=False):
        split_suffix_map = {
            "pretrain": "train", "pretrain-val": "valid", "pretrain-test": "test",
            "train": "train", "val": "valid", "test": "test"
        }
        split_suffix = split_suffix_map[split]

        if data_source == "auto":
            sources = ["pretrain"] if split.startswith("pretrain") else ["downstream"]
        elif data_source == "both":
            sources = ["pretrain", "downstream"]
        else:
            sources = [data_source]

        patient_dfs = []
        epoch_dfs = []
        csv_prefix = "epoch_regression" if regression_targets else "epoch"

        for source in sources:
            patient_csv = f"{csv_dir}/patient_{source}_{split_suffix}.csv"
            epoch_csv = f"{csv_dir}/{csv_prefix}_{source}_{split_suffix}.csv"

            if Path(patient_csv).is_file() and Path(epoch_csv).is_file():
                patient_dfs.append(pd.read_csv(patient_csv))
                epoch_dfs.append(pd.read_csv(epoch_csv))

        patient_df = pd.concat(patient_dfs, ignore_index=True).drop_duplicates(subset=['nsrrid'])
        epoch_df = pd.concat(epoch_dfs, ignore_index=True)

        base_cols = ['nsrrid', 'seg_id', 'dataset_name', 'epoch_id', 'path_head']
        if event_cols:
            if return_all_event_cols:
                for col in event_cols:
                    if col and col not in base_cols:
                        base_cols.append(col)
            elif event_cols[0]:
                base_cols.append(event_cols[0])

        if regression_targets:
            for t in regression_targets:
                col_name = f"{t}_mean"
                if col_name in epoch_df.columns:
                    base_cols.append(col_name)

        keep_cols = [c for c in base_cols if c in epoch_df.columns]
        epoch_df = epoch_df[keep_cols].copy()

        if regression_targets:
            label_cols = [f"{t}_mean" for t in regression_targets]
            existing = [c for c in label_cols if c in epoch_df.columns]
            if existing:
                epoch_df = epoch_df.dropna(subset=existing).reset_index(drop=True)

        if regression_filter_config:
            for col_name, filter_rules in regression_filter_config.items():
                if col_name in epoch_df.columns:
                    mask = pd.Series([True] * len(epoch_df))
                    if "min" in filter_rules:
                        mask = mask & (epoch_df[col_name] >= filter_rules["min"])
                    if "max" in filter_rules:
                        mask = mask & (epoch_df[col_name] <= filter_rules["max"])
                    epoch_df = epoch_df[mask].reset_index(drop=True)

        if include_datasets is not None and 'dataset_name' in epoch_df.columns:
            include_lower = [d.lower() for d in include_datasets]
            mask = epoch_df['dataset_name'].astype(str).str.lower().isin(include_lower)
            epoch_df = epoch_df[mask].copy()
            patient_df = patient_df[patient_df['nsrrid'].isin(epoch_df['nsrrid'].unique())].copy()

        return patient_df, epoch_df

    def __len__(self) -> int:
        return len(self.epoch_df)

    def get_class_counts(self) -> Optional[np.ndarray]:
        return self._class_counts

    def _resample_df(self, df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
        if not np.issubdtype(df.index.dtype, np.number):
            t = np.arange(len(df)) / float(target_hz)
            df = df.copy()
            df.index = t

        t0 = float(df.index.min())
        t1 = float(df.index.max())
        t_target = np.arange(t0, t0 + self.window_size, 1.0 / target_hz)
        if t_target[-1] > t1:
            t_target = t_target[t_target <= t1 + 1e-9]
        out = df.reindex(t_target).interpolate(method="linear", limit_direction="both")
        return out.fillna(0.0)

    def __getitem__(self, idx: int):
        row = self.epoch_df.iloc[idx]
        nsrrid = row["nsrrid"]
        seg_id = int(row["seg_id"])
        cols = list(self.train_edf_cols) if self.train_edf_cols is not None else None

        if self.split == "pretrain":
            df_epoch = self._load_epoch_all_df(row["path_head"], seg_id, columns=cols)
            df_epoch = self._resample_df(df_epoch, self.sample_rate)

            if cols is not None:
                for ch in cols:
                    if ch not in df_epoch.columns:
                        df_epoch[ch] = 0.0
                    elif ch in NEED_NORM_COL:
                        df_epoch[ch] = to_pm1(df_epoch[ch])
                df_epoch = df_epoch[cols]

            samples_per_epoch = int(self.window_size * self.sample_rate)
            if len(df_epoch) < samples_per_epoch:
                pad = samples_per_epoch - len(df_epoch)
                tail = pd.DataFrame({c: 0.0 for c in df_epoch.columns},
                                    index=df_epoch.index[-1] + (np.arange(1, pad + 1) / self.sample_rate))
                df_epoch = pd.concat([df_epoch, tail], axis=0)
            elif len(df_epoch) > samples_per_epoch:
                df_epoch = df_epoch.iloc[:samples_per_epoch]

            x = torch.tensor(df_epoch.to_numpy(copy=False), dtype=torch.float32).t().contiguous()
            x = torch.clamp(x, min=-6, max=6)

            output = {"psg": x}
            if self.return_nsrrid:
                output["nsrrid"] = nsrrid
                output["seg_id"] = seg_id

            if self.patient_cols:
                y = torch.tensor(self.patient_df.loc[nsrrid, self.patient_cols].values.astype(float), dtype=torch.float32)
                output["label"] = y.long() if not self.return_nsrrid else y
            elif self.event_cols:
                if self.return_all_event_cols:
                    available_cols = [c for c in self.event_cols if c in row.index]
                    y = torch.tensor([row[c] for c in available_cols], dtype=torch.float32)
                else:
                    y = torch.tensor([row[self.event_cols[0]]], dtype=torch.float32)
                output["label"] = y

            return output
        else:
            # Downstream split
            if self._seg_indices is None:
                seg_df = self.all_epoch_df[
                    (self.all_epoch_df['nsrrid'] == nsrrid) & (self.all_epoch_df['seg_id'] == seg_id)
                ].sort_values('epoch_id')
            else:
                idx_arr = self._seg_indices.get((nsrrid, seg_id))
                seg_df = self.all_epoch_df.iloc[idx_arr] if idx_arr is not None else \
                    self.all_epoch_df[(self.all_epoch_df['nsrrid'] == nsrrid) & (self.all_epoch_df['seg_id'] == seg_id)].sort_values('epoch_id')

            df_epoch = self._load_epoch_all_df(row["path_head"], seg_id, columns=cols)
            df_epoch = self._resample_df(df_epoch, self.sample_rate)

            if cols is not None:
                for ch in cols:
                    if ch not in df_epoch.columns:
                        df_epoch[ch] = 0.0
                    elif ch in NEED_NORM_COL:
                        df_epoch[ch] = to_pm1(df_epoch[ch])
                df_epoch = df_epoch[cols]

            samples_per_epoch = int(self.window_size * self.sample_rate)
            if len(df_epoch) < samples_per_epoch:
                pad = samples_per_epoch - len(df_epoch)
                tail = pd.DataFrame({c: 0.0 for c in df_epoch.columns},
                                    index=df_epoch.index[-1] + (np.arange(1, pad + 1) / self.sample_rate))
                df_epoch = pd.concat([df_epoch, tail], axis=0)
            elif len(df_epoch) > samples_per_epoch:
                df_epoch = df_epoch.iloc[:samples_per_epoch]

            x = torch.tensor(df_epoch.to_numpy(copy=False), dtype=torch.float32).t().contiguous()
            x = torch.clamp(x, min=-6, max=6)

            output = {"psg": x}
            if self.return_nsrrid:
                output["nsrrid"] = nsrrid
                output["seg_id"] = seg_id

            if self.patient_cols:
                y = torch.tensor(self.patient_df.loc[nsrrid, self.patient_cols].values.astype(float), dtype=torch.float32)
                y = y.repeat(self.window_size // self.epoch_length)
                output["label"] = y
            elif self.event_cols:
                if self.return_all_event_cols:
                    available_cols = [c for c in self.event_cols if c in seg_df.columns]
                    y = torch.tensor(seg_df[available_cols].values.astype(float), dtype=torch.float32).squeeze(0)
                else:
                    y = torch.tensor(seg_df[self.event_cols].values.astype(float), dtype=torch.float32).squeeze(1)
                output["label"] = y
            elif self.regression_targets:
                label_cols = [f"{t}_mean" for t in self.regression_targets]
                y = torch.tensor([row[c] for c in label_cols], dtype=torch.float32)
                output["label"] = y

            return output

    def _build_epoch_all_path(self, path_head: str, epoch_id: int) -> Path:
        return Path(f"{path_head}/epoch-{epoch_id:05d}_all.parquet")

    def _load_epoch_all_df(self, path_head: str, epoch_id: int, columns=None) -> pd.DataFrame:
        fp = self._build_epoch_all_path(path_head, epoch_id)
        if not fp.is_file():
            raise FileNotFoundError(f"Parquet missing: {fp}")
        df = pd.read_parquet(fp)
        for c in df.columns:
            if not np.issubdtype(df[c].dtype, np.floating):
                with suppress(Exception):
                    df[c] = df[c].astype(np.float32)
        return df
