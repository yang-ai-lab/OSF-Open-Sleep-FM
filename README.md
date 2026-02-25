# OSF: On Pre-training and Scaling of Sleep Foundation Models

## 🔥 News

- Our codebase and checkpoint is released. Full codebase for benchmarking will be public available after acceptance.
- Our paper is out.

## 📖 Introduction

Polysomnography (PSG) provides the gold standard for sleep assessment but suffers from substantial heterogeneity across recording devices and cohorts.
There have been growing efforts to build general-purpose foundation models (FMs) for sleep physiology, but lack an in-depth understanding of the pre-training process and scaling patterns that lead to more generalizable sleep FMs.
To fill this gap, we curate a massive corpus of 166,500 hours of sleep recordings from nine public sources and establish SleepBench, a comprehensive, fully open-source benchmark.
Leveraging SleepBench, we systematically evaluate four families of self-supervised pre-training objectives and uncover three critical findings:
(1) existing FMs fail to generalize to missing channels at inference;
(2) channel-invariant feature learning is essential for pre-training;
and (3) scaling sample size, model capacity, and multi-source data mixture consistently improves downstream performance.
With an enhanced pre-training and scaling recipe, we introduce OSF, a family of sleep FMs that achieves state-of-the-art performance across nine datasets on diverse sleep and disease prediction tasks.
Further analysis of OSF also reveals intriguing properties in sample efficiency, hierarchical aggregation, and cross-dataset scaling.


## 📖 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Pretrained Weights](#-pretrained-weights)
4. [Usage](#-usage)
5. [Benchmark Evaluations](#-benchmark-evaluations)
6. [Supported Datasets](#-supported-datasets)
7. [Citation](#-citation)

## 💿 Installation

```bash
git clone https://github.com/tennis-rabbit/OSF-Open-Sleep-Foundation-Model.git
cd OSF-Open-Sleep-Foundation-Model
conda env create -f environment.yml
conda activate myenv
```


### Dependencies

- Python >= 3.10
- PyTorch >= 2.9.0
- PyTorch Lightning >= 2.5.5


## 🚀 Quick Start

We provide a demo notebook (`demo.ipynb`) demonstrating how to extract embeddings from PSG signals using the pretrained model.

```python
import torch
from osf.backbone.vit1d_cls import vit_base

# Load pretrained weights
payload = torch.load("pretrained_weights/osf_backbone.pth", map_location="cpu")
meta = payload["metadata"]

# Initialize model
backbone = vit_base(
    num_leads=meta["num_leads"],        # 12 channels
    seq_len=meta["seq_len"],            # 1920 (64 Hz × 30 s)
    patch_size=meta["patch_size_time"],
    lead_wise=meta["lead_wise"],
    patch_size_ch=meta["patch_size_ch"],
)
backbone.load_state_dict(payload["state_dict"])
backbone.eval()

# Extract embeddings
# x: [B, 12, 1920] - 12-channel PSG, 64 Hz × 30 seconds
with torch.no_grad():
    cls_embs, patch_embs = backbone.forward_encoding(x, return_sequence=False)
# cls_embs: [B, 768] - Global epoch-level representation
# patch_embs: [B, 90, 768] - Local patch representations
```

## 📦 Pretrained Weights

| Model | Backbone | Channels | Download |
|-------|----------|----------|----------|
| OSF | ViT-Base | 12-ch | [OSF Open Download Link](https://drive.google.com/drive/u/1/folders/1GmXbBRU05NQZaU37fiEWe4PhUmjD-clt) |

After downloading, place the weight files in the `pretrained_weights/` directory.

## 👩‍💻 Usage

### Input Format

Expected input format:
- **12 PSG Channels**: ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN, EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1
- **Sample Rate**: 64 Hz
- **Epoch Length**: 30 seconds
- **Input Shape**: `[B, 12, 1920]`

### Pretraining

We support multiple self-supervised pretraining methods, for example, to launch pre-training of our OSF method, run pretraining:

```bash
python main_pretrain.py \
    --model_name "dino_ours" \
    --psg_encoder_name "vit_base" \
    --batch_size 256 \
    --lr 5e-5 \
    --max_epochs 30 \
    --num_devices 4 \
    --patch_size_time 64 \
    --patch_size_ch 4 \
    --precision "bf16-mixed"
```

See `main_pipleines/main_pretrain.py` for more detailed settings.

### Fine-tuning

Fine-tune the pretrained model on downstream tasks:

```bash
python main_finetune.py \
    --model_name "dino_ours" \
    --ckpt_path "/path/to/pretrained/checkpoint.ckpt" \
    --downstream_dataset_name "shhs" \
    --eval_label "Stage" \
    --train_data_pct 1.0 \
    --max_steps 500 \
    --lr 0.1 \
    --num_devices 4
```


## 📊 Benchmark Evaluations

### Benchmarked SSL Methods

| Method | Type | Original Paper |
|--------|------|-------------|
| SleepFM | Contrastive | [Leave-one-out multi-modal contrastive learning](https://www.nature.com/articles/s41591-025-04133-4.pdf) |
| SimCLR | Contrastive | [Simple Constrastive Learning](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf) |
| DINO | Self-distillation | [DINO](https://arxiv.org/pdf/2304.07193) |
| VQ-VAE | Reconstruction | [Vector-quantized variational autoencoder](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf) |
| MAE | Reconstruction | [Masked Autoencoding](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) |
| AR | Autoregressive | [Autoregressive Next-Token prediction](https://storage.prod.researchhub.com/uploads/papers/2020/06/01/language-models.pdf) |
| OSF | Self-distillation | ours |

### Downstream Tasks

**Epoch-level Classification Tasks:**

| Task | Classes | Description |
|------|---------|-------------|
| Sleep Stage | 4 | Awake, Light Sleep, Deep Sleep, REM classification |
| Arousal | 2 | Arousal event detection |
| Hypopnea | 2 | Hypopnea event detection |
| Oxygen Desaturation | 2 | Oxygen desaturation detection |


### Evaluation Settings

| Setting | Description |
|---------|-------------|
| Linear Probing | Freeze backbone, train linear classifier |
| Full Fine-tuning | Fine-tune entire model end-to-end |
| Few-shot (k-shot) | Train with limited labeled samples |

For example scripts, see `main_pipelines` and `bash_scripts` folders.

## 📊 Supported Datasets

We aggregated nine large-scale datasets from the National Sleep Research Resource platform.

| Dataset | Full Name | Source |
|---------|-----------|--------|
| SHHS | Sleep Heart Health Study | NSRR |
| CHAT | Childhood Adenotonsillectomy Trial | NSRR |
| MROS | MrOS Sleep Study | NSRR |
| CCSHS | Cleveland Children's Sleep and Health Study | NSRR |
| CFS | Cleveland Family Study | NSRR |
| MESA | Multi-Ethnic Study of Atherosclerosis | NSRR |
| SOF | Study of Osteoporotic Fractures | NSRR |
| WSC | Wisconsin Sleep Cohort | NSRR |
| STAGES | Stanford Technology Analytics and Genomics in Sleep | NSRR |
| NCHSDB | NCH Sleep DataBank | NSRR |

For new users, please apply for an account and access to each of these datasets following instructions here [NSRR Registration](https://sleepdata.org/join)

## 📁 Project Structure

```
OSF-Open-Sleep-Foundation-Model/
├── osf/
│   ├── backbone/          # ViT backbone implementations
│   │   └── vit1d_cls.py
│   ├── models/            # SSL model implementations
│   │   └── dino_model_cls.py
│   │   
│   ├── datasets/          # Data loading utilities
│   └── utils/             # Helper functions
├── main_pipelines/        # Training scripts
│   ├── main_pretrain.py
│   └── ...
├── bash_scripts/          # Example bash scripts
├── pretrained_weights/    # Pretrained model weights
├── demo.ipynb             # Quick start demo
├── config.py              # Dataset and channel configurations
└── train_config.py        # Training configurations
```


## 📝 Citation

If you use this code or models in your research, please cite our paper:

```bibtex
[TODO]
```

