"""
Utility functions for saving experiment results to JSON/CSV.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


def convert_to_serializable(value):
    """Convert tensor/numpy values to Python native types for JSON serialization."""
    if hasattr(value, 'item'):  # torch.Tensor
        return float(value.item())
    elif isinstance(value, (np.ndarray, np.generic)):
        return float(value)
    return value


def extract_embedding_type(root_dir: str) -> str:
    """
    Extract embedding type identifier from root_dir path.
    
    Examples:
        "/work/.../dino_stage1_emb_no_norm" -> "dino_no_norm"
        "/work/.../dino_stage1_emb" -> "dino"
        "/work/.../mae_emb_normalized" -> "mae_normalized"
    
    Args:
        root_dir: Path to embedding directory
        
    Returns:
        Short embedding type identifier
    """
    if not root_dir:
        return "unknown"
    
    basename = os.path.basename(root_dir.rstrip('/'))
    
    # Remove common suffixes/patterns
    emb_type = basename
    emb_type = emb_type.replace("_stage1_emb", "")
    emb_type = emb_type.replace("_stage1", "")
    emb_type = emb_type.replace("_emb", "")
    emb_type = emb_type.replace("final_", "")
    
    # Keep it concise
    if len(emb_type) > 30:
        emb_type = emb_type[:30]
    
    return emb_type if emb_type else "emb"


def format_lr(lr: float) -> str:
    """Format learning rate for filenames (e.g., 0.001 -> 1e-3)."""
    if lr >= 1:
        return f"{lr:.0f}"
    elif lr >= 0.1:
        return f"{lr:.1f}"
    else:
        # Convert to scientific notation
        exp = 0
        val = lr
        while val < 1:
            val *= 10
            exp += 1
        return f"{val:.0f}e-{exp}"


def save_results_to_json(
    test_metrics: Dict[str, Any],
    hparams: Any,
    extension: str,
    ckpt_dir: str,
    timestamp: str,
    results_dir: str = "./results",
    extra_fields: Optional[Dict[str, Any]] = None,
    filename_prefix: str = "",
) -> str:
    """
    Save test results to a JSON file.
    
    Args:
        test_metrics: Dictionary of test metrics from trainer.test()
        hparams: Hyperparameters namespace/object
        extension: Experiment extension string (now used as run_name)
        ckpt_dir: Checkpoint directory path
        timestamp: Timestamp string
        results_dir: Directory to save results (default: ./results)
        extra_fields: Additional fields to include in the result record
            - Should include: exp_type, task, dataset, model, etc.
        filename_prefix: Prefix for the filename
    
    Returns:
        Path to the saved JSON file
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Base result record
    result_record = {
        "run_name": extension,  # Renamed from "extension" for clarity
        "ckpt_dir": ckpt_dir,
        "timestamp": timestamp,
    }
    
    common_fields = [
        "model_name", "downstream_dataset_name", "ckpt_path", "stage2_ckpt_path",
        "eval_label", "patient_cols", "use_which_backbone", "variant",
        "in_features", "train_data_pct", "lr", "batch_size", 
        "max_epochs", "max_steps", "loss_type", "use_mean_pool",
        "root_dir", "is_pretrain", "pooling", "use_transformer", "use_mil",
        "encoder_name", "encoder", "mask_channels", "encoder_size",
        "num_classes", "seed",
    ]
    for field in common_fields:
        if hasattr(hparams, field):
            result_record[field] = getattr(hparams, field)
    
    standard_metrics = [
        "test_acc", "test_f1", "test_f1_w", "test_auc", "test_auprc",
        "test_kappa", "test_rec_m", "test_loss",
        "test/acc", "test/f1_macro", "test/auc_macro", "test/auprc_macro",
    ]
    for metric in standard_metrics:
        if metric in test_metrics:
            key = metric.replace("/", "_")
            result_record[key] = test_metrics[metric]
    
    for key, value in test_metrics.items():
        if key.startswith("test/") or key.startswith("test_"):
            normalized_key = key.replace("/", "_")
            if normalized_key not in result_record:
                result_record[normalized_key] = value
    
    if extra_fields:
        result_record.update(extra_fields)
    
    for key, value in result_record.items():
        result_record[key] = convert_to_serializable(value)
    
    if filename_prefix:
        result_filename = f"{filename_prefix}_{timestamp}.json"
    else:
        model_name = getattr(hparams, 'model_name', 'model')
        dataset_name = getattr(hparams, 'downstream_dataset_name', 'dataset')
        label = getattr(hparams, 'eval_label', None) or getattr(hparams, 'patient_cols', 'task')
        result_filename = f"{model_name}_{dataset_name}_{label}_{timestamp}.json"
    
    result_path = os.path.join(results_dir, result_filename)
    
    # Save to JSON
    with open(result_path, 'w') as f:
        json.dump(result_record, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {result_path}")
    print(f"{'='*80}\n")
    
    return result_path


def aggregate_results_to_csv(
    results_dirs: List[str],
    output_path: str = "./results/aggregated_results.csv",
    key_columns: Optional[List[str]] = None,
    metric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate all JSON result files from multiple directories into a single CSV.
    
    Args:
        results_dirs: List of directories containing JSON result files
        output_path: Path to save the aggregated CSV
        key_columns: Columns to use as identifiers (default: common experiment params)
        metric_columns: Metric columns to include (default: all test metrics)
    
    Returns:
        DataFrame with aggregated results
    """
    if key_columns is None:
        key_columns = [
            "exp_type", "task", "dataset", "model", "encoder",
            "train_data_pct", "lr", "embedding_type",
            "pretrain_ckpt_path", "finetuned_ckpt_dir", "trained_ckpt_dir",
            "stage2_pretrain_ckpt", "embedding_root_dir",
            "model_name", "downstream_dataset_name", "eval_label", "patient_cols",
            "use_which_backbone", "variant", "loss_type",
            "use_mean_pool", "pooling", "use_transformer", "use_mil",
            "mask_channels", "mask_channels_str",
            "ckpt_path", "stage2_ckpt_path", "root_dir",
        ]
    
    if metric_columns is None:
        metric_columns = [
            "test_acc", "test_f1", "test_f1_w", "test_auc", "test_auprc",
            "test_kappa", "test_rec_m", "test_loss",
        ]
    
    all_records = []
    
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            print(f"[WARN] Directory not found: {results_dir}")
            continue
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(results_dir, "*.json"))
        print(f"[INFO] Found {len(json_files)} JSON files in {results_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    record = json.load(f)
                record['_source_file'] = os.path.basename(json_file)
                record['_source_dir'] = results_dir
                all_records.append(record)
            except Exception as e:
                print(f"[WARN] Failed to load {json_file}: {e}")
    
    if not all_records:
        print("[WARN] No records found!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    existing_key_cols = [c for c in key_columns if c in df.columns]
    existing_metric_cols = [c for c in metric_columns if c in df.columns]
    
    per_class_cols = [c for c in df.columns if c.startswith("test_") and c not in existing_metric_cols]
    per_class_cols = sorted(per_class_cols)
    
    other_cols = [c for c in df.columns if c not in existing_key_cols + existing_metric_cols + per_class_cols]
    
    ordered_cols = existing_key_cols + existing_metric_cols + per_class_cols + other_cols
    df = df[[c for c in ordered_cols if c in df.columns]]
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"Aggregated {len(all_records)} results to: {output_path}")
    print(f"Columns: {list(df.columns[:10])}... ({len(df.columns)} total)")
    print(f"{'='*80}\n")
    
    return df


def load_results_from_json(json_path: str) -> Dict[str, Any]:
    """Load a single JSON result file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def filter_results(
    df: pd.DataFrame,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    eval_label: Optional[str] = None,
    patient_cols: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter aggregated results DataFrame by common fields.
    
    Args:
        df: DataFrame from aggregate_results_to_csv()
        model_name: Filter by model name
        dataset_name: Filter by downstream dataset name
        eval_label: Filter by eval label (stage 1)
        patient_cols: Filter by patient columns (stage 2)
    
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    if model_name is not None and 'model_name' in filtered.columns:
        filtered = filtered[filtered['model_name'] == model_name]
    if dataset_name is not None and 'downstream_dataset_name' in filtered.columns:
        filtered = filtered[filtered['downstream_dataset_name'] == dataset_name]
    if eval_label is not None and 'eval_label' in filtered.columns:
        filtered = filtered[filtered['eval_label'] == eval_label]
    if patient_cols is not None and 'patient_cols' in filtered.columns:
        filtered = filtered[filtered['patient_cols'] == patient_cols]
    
    return filtered

