# Monocular Depth Estimation via Fine-Tuning Depth Anything V2

**CS 4391 — Computer Vision | University of Texas at Dallas | Spring 2026**
*Joshua Oliveira-Martin · Suraj Rongali · Oghenetejiri Etaghene · Kenton Le* ·
*Advisor: Yu Xiang*

---

## Overview

This repository contains the full pipeline for **domain-adaptive monocular depth estimation** built on top of [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — a Vision Transformer (ViT) encoder paired with a Dense Prediction Transformer (DPT) decoder.

Monocular depth estimation recovers per-pixel depth from a single RGB image, with no specialized hardware required. While Depth Anything V2 generalizes well across many scenes, it exhibits a measurable domain gap on **controlled indoor university classroom environments**, where lighting conditions, object arrangements, and scale differ significantly from its pre-training distribution.

This project addresses that gap through:

1. **Custom data collection** — 114 RGB images captured across UTD classrooms using three consumer mobile devices (Samsung Galaxy, iPhone, Google Pixel), filtered to 100 clean frames via an automated quality audit pipeline
2. **Targeted fine-tuning** — decoder-only and full fine-tuning strategies on the ViT-S (25M parameter) variant, with ablation over loss weight configurations and training duration
3. **Systematic evaluation** — comparison across eight standard depth metrics against the pretrained baseline
4. **Edge deployment** — TensorRT/ONNX optimization for real-time inference on the **NVIDIA Jetson Orin Nano**

Our best fine-tuned model achieves a **37.5% reduction in RMSE-log**, a **13.9% reduction in AbsRel**, and a **+2.1% improvement in δ₁ accuracy** over the pretrained baseline.

---

## Architecture

```
RGB Input (518×518)
        │
        ▼
┌───────────────────┐
│   ViT-S Encoder   │  DINOv2 backbone (25M params)
│  16×16 patches    │  Pretrained on 62M images + synthetic data
│  ImageNet norm    │  Frozen (decoder_only) or 10× lower LR (full)
└────────┬──────────┘
         │  Feature maps from 4 ViT stages
         ▼
┌───────────────────┐
│   DPT Decoder     │  Dense Prediction Transformer head
│  Progressive      │  Skip connections from all encoder stages
│  upsampling       │  Fine-grained edge and detail preservation
└────────┬──────────┘
         │
         ▼
  Dense Depth Map (518×518)
  warm = near  ·  cool = far
```

**Supported encoder variants:**

| Encoder | Parameters | Use Case |
|---------|-----------|----------|
| ViT-S   | 25M       | Development, low-VRAM training |
| ViT-B   | 97M       | Balanced quality/speed |
| ViT-L   | 307M      | Maximum accuracy |
| ViT-G   | 1.3B      | Research / full-scale evaluation |

---

## Key Results

### Quantitative Metrics (Best Fine-Tuned vs. Baseline)

| Model | AbsRel ↓ | RMSE-log ↓ | δ₁ ↑ | Loss Config |
|-------|----------|------------|------|-------------|
| Pretrained baseline (ViT-S) | — | — | — | — |
| FT · L(1.0, 0.5, 0.25) · 400ep | −12.1% | −35.2% | +1.8% | SILog + Edge + SSIM |
| **FT · L(1.0, 0.3, 0.1) · 400ep** | **−13.9%** | **−37.5%** | **+2.1%** | SILog + Edge + SSIM |
| FT · L(1.0, 0.3, 0.1) · 150ep | −10.4% | −29.8% | +1.4% | SILog + Edge + SSIM |

> All improvements are relative to the pretrained ViT-S checkpoint on the UTD custom test split.
> ↓ lower is better · ↑ higher is better

### Loss Ablation Summary

| λ_SILog | λ_Edge | λ_SSIM | Epochs | Batch | Best Val Loss |
|---------|--------|--------|--------|-------|---------------|
| 1.0 | 0.5 | 0.25 | 400 | 8 | — |
| **1.0** | **0.3** | **0.1** | **400** | **8** | **best** |
| 1.0 | 0.3 | 0.1 | 150 | 8 | — |
| 1.0 | 1.0 | 1.0 | 400 | 8 | — |
| 1.0 | 0.2 | 0.05 | 400 | 8 | — |

Higher edge/SSIM weights improve structural fidelity (δ thresholds), while lower weights allow more aggressive fitting of the target domain's depth distribution. For fixed-domain applications, 400-epoch training maximizes accuracy; for out-of-distribution robustness, 150 epochs provides a better generalization tradeoff.

---

## Dataset

### UTD Custom Classroom Dataset

**Full dataset and pretrained checkpoints:** [Google Drive](https://drive.google.com/drive/folders/17aWaookJQaYZaPKRYY1DmzQfMN4aCsjd?usp=sharing)

The Drive folder contains:
- `UTD-CV-Data/` — 114 raw captured images (pre-audit)
- `UTD_cust/processed/` — 100 clean 518×518 processed images, pseudo-depth maps (`.npy`), validity masks, and train/val/test splits
- `checkpoints/` — pretrained ViT-S baseline and all fine-tuned `.pt` checkpoints
- Example inference results and qualitative comparisons

### Collection Protocol

| Device | Resolution | Count |
|--------|-----------|-------|
| Samsung Galaxy | 3000×4000 | ~24 |
| Apple iPhone | 4032×3024 | ~56 |
| Google Pixel | 4080×3072 | ~34 |

Images were captured across **multiple UTD classrooms** under varied lighting conditions — overhead fluorescents, mixed natural/artificial, and window-adjacent scenes. The multi-device strategy introduces natural variation in color response, noise characteristics, and field of view.

### Automated Quality Audit

Of **114 captured images**, the audit pipeline retained **100 clean frames**:

| Outcome | Count | Reason |
|---------|-------|--------|
| ✅ Passed (clean) | 98 | All checks passed |
| ⚠️ Warning | 2 | Near-duplicate detected (manual review) |
| ❌ Failed | 14 | Motion blur (Laplacian variance < 100) |

Audit thresholds:

```
Blur threshold:           Laplacian variance < 100  → reject
Near-duplicate:           Perceptual hash distance ≤ 8  → flag for review
Brightness range:         Mean pixel value 20–235
Minimum dimension:        392 px short edge
Min file density:         10 KB / megapixel
```

### Preprocessing

All retained images are center-cropped and resized to **518×518** (matching DAv2 input resolution). Mean crop content retention is **75.1%**. The processed directory follows this structure:

```
UTD_cust/
└── processed/
    ├── images/         # 000001.jpg … 000100.jpg  (518×518 JPEG)
    ├── depths/         # 000001.npy … 000100.npy  (float32 depth maps)
    ├── masks/          # 000001.png … 000100.png  (validity masks)
    ├── metadata.json
    └── splits/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

### Example Data

A small set of example images and their corresponding pseudo-depth maps is included in `data/examples/` for quick testing without downloading the full dataset. See the [Google Drive link](https://drive.google.com/drive/folders/17aWaookJQaYZaPKRYY1DmzQfMN4aCsjd?usp=sharing) for the complete dataset.

---


## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (training tested on NVIDIA RTX 3060 12 GB)
- CUDA 11.8+ and cuDNN

### 1. Clone this repository

```bash
git clone https://github.com/josh-oliveira/DA_V2_UTD.git
cd DA_V2_UTD
```

### 2. Clone Depth Anything V2 alongside this repo

The training script expects Depth Anything V2 as a sibling directory:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2 && pip install -e . && cd ..
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download pretrained weights

Place the pretrained ViT-S checkpoint in `checkpoints/`:

All fine-tuned checkpoints are also available on [Google Drive](https://drive.google.com/drive/folders/17aWaookJQaYZaPKRYY1DmzQfMN4aCsjd?usp=sharing).

---

## Pipeline Walkthrough

### Step 1 — Image Quality Audit

Scans the raw image directory and flags motion-blurred and near-duplicate frames. Generates `audit_report.json` and `audit_summary.txt`.

```bash
python scripts/audit.py --input_dir ./UTD-CV-Data/
```

Key outputs:
- `audit_report.json` — per-image checks (blur, brightness, resolution, duplicates)
- `audit_summary.txt` — human-readable pass/fail/warning table

### Step 2 — Resize and Crop

Center-crops and resizes all passing images to 518×518 and generates pseudo-depth maps from the pretrained model.

```bash
python scripts/resize.py \
    --input_dir  ./UTD-CV-Data/ \
    --output_dir ./UTD_cust/processed/
```

Then generate pseudo-depth labels using the pretrained ViT-L encoder for highest-quality targets:

```bash
python gen.py \
    --image_dir  ./UTD_cust/processed/images \
    --encoder    vitl \
    --visualize \
    --weights    checkpoints/depth_anything_v2_vitl.pth
```

### Step 3 — Dataset Analysis

Produces the 12-panel `dataset_analysis.png` covering orientation split, aspect ratio distribution, resolution tiers, scale factors, crop content retention, brightness, and sharpness. See [`info.md`](info.md) for a full panel-by-panel reference.

```bash
python scripts/analyze.py \
    --metadata     ./UTD_cust/processed/metadata.json \
    --audit_report ./UTD_cust/processed/audit_report.json
```

### Step 4 — Train/Val/Test Splits

Creates `splits/train.txt`, `splits/val.txt`, and `splits/test.txt` inside the processed directory.

```bash
python splits.py --processed_dir ./UTD_cust/processed/
```

Verify the dataloader loads correctly before training:

```bash
python scripts/UTD_dataloader.py --processed_dir ./UTD_cust/
```

### Step 5 — Fine-Tuning

The main training script supports two strategies and a composable set of loss functions.

**Recommended starting point (decoder-only, all three losses):**

```bash
python scripts/train.py \
    --processed_dir ./UTD_cust/ \
    --strategy      decoder_only \
    --losses        silog edge ssim \
    --loss_weights  1.0 0.3 0.1 \
    --epochs        400 \
    --batch_size    8 \
    --lr            1e-5
```

**Full fine-tune (encoder + decoder, differential LR):**

```bash
python scripts/train.py \
    --processed_dir ./UTD_cust/ \
    --strategy      full \
    --losses        silog edge ssim \
    --loss_weights  1.0 0.3 0.1 \
    --epochs        400 \
    --batch_size    8 \
    --lr            1e-5
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--processed_dir` | required | Path to processed/ directory |
| `--strategy` | `decoder_only` | `decoder_only` or `full` |
| `--losses` | `silog edge ssim` | Space-separated list of loss terms |
| `--loss_weights` | `1.0 0.3 0.1` | Weights for each loss term |
| `--epochs` | `400` | Number of training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-5` | Base learning rate |
| `--encoder` | `vits` | `vits`, `vitb`, `vitl`, `vitg` |

Checkpoints are saved to `<processed_dir>/checkpoints/` with dynamic names encoding all key hyperparameters, e.g.:
```
best_model_batch8_stratdecoder_only_encvits_epoch400_L_1_03_01.pt
```

### Step 6 — Plot Training Curves

```bash
python scripts/plot_training.py \
    --log logs/training_log_batch8_stratdecoder_only_encvits_epoch400_L_1_03_01.json
```

### Step 7 — Ablation Study

Compare multiple checkpoints side-by-side across all depth metrics. The experiment string format is `<label>::<checkpoint_path>::<processed_dir>`.

```bash
python scripts/ablation.py \
    --experiments \
        baseline::checkpoints/depth_anything_v2_vits.pth::./UTD_cust \
        ft_L105025::checkpoints/best_model_batch8_stratdecoder_only_encvits_epoch400_L_1_05_025.pt::./UTD_cust \
        ft_L10301::checkpoints/best_model_batch8_stratdecoder_only_encvits_epoch400_L_1_03_01.pt::./UTD_cust \
        ft_L10301_e150::checkpoints/best_model_batch8_stratdecoder_only_encvits_epoch150_L_1_03_01.pt::./UTD_cust
```

Outputs:
- `ablation_results.json` — machine-readable per-experiment metrics
- `ablation_report.txt` — human-readable comparison table

---

## Training Details

### Loss Functions

The total loss is a weighted sum of three complementary terms:

```
L_total = λ_SILog · L_SILog + λ_Edge · L_Edge + λ_SSIM · L_SSIM
```

| Term | Formula | Role |
|------|---------|------|
| **SILog** | `(1/n)Σdᵢ² − (λ/n²)(Σdᵢ)²` where `dᵢ = log(ŷᵢ) − log(yᵢ)`, `λ=0.85` | Primary depth regression; scale-invariant |
| **Edge-aware** | Depth gradient error weighted by RGB image edges | Preserves sharp object boundaries |
| **SSIM** | Structural similarity on local depth patches | Maintains local depth patterns and textures |

SILog operates pixel-independently; SSIM complements it by capturing local structural consistency. The edge term bridges both by anchoring gradients to scene geometry.

### Optimizer and Schedule

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Encoder LR (`decoder_only` frozen; `full` strategy) | 5×10⁻⁵ |
| Decoder LR | 5×10⁻⁴ (10× encoder LR) |
| Schedule | Cosine annealing with 3-epoch linear warmup |
| Batch sizes ablated | 4, 8, 16, 32 |

### Fine-Tuning Strategies

**`decoder_only`** — The DINOv2 ViT encoder is fully frozen; only the DPT head is updated. Fast, stable, and preferred for small datasets (< 200 images). Recommended for this dataset.

**`full`** — All weights are updated with differential learning rates (encoder at 10× lower LR to preserve pretrained representations). Higher accuracy ceiling but requires more data and longer training.

---

## Deployment

### Phase 1 — GPU Training

Training was conducted on an **NVIDIA RTX 5070TI (16 GB VRAM GDDR7)** using the ViT-S variant. All training logs are committed to `logs/` and can be replayed with `plot_training.py`.

Checkpoint naming convention:

```
best_model_batch{B}_strat{strategy}_enc{encoder}_epoch{E}_L_{λ1}_{λ2}_{λ3}.pt
```

Example: `best_model_batch8_stratdecoder_only_encvits_epoch400_L_1_03_01.pt` corresponds to batch size 8, decoder-only strategy, ViT-S encoder, 400 epochs, λ = (1.0, 0.3, 0.1).

### Phase 2 — Edge Deployment (NVIDIA Jetson Orin Nano)

The fine-tuned model is exported to ONNX and then compiled to TensorRT for real-time inference on the **NVIDIA Jetson Orin Nano** embedded platform.

#### Export to ONNX

```bash
python deployment/export.py \
    --checkpoint checkpoints/best_model_batch8_stratdecoder_only_encvits_epoch400_L_1_03_01.pt \
    --encoder    vits \
    --output     deployment/depth_utd_ft.onnx
```


#### Inference on Jetson

```bash
python depth_cam.py --engine models/best_e400_L_1_03_01.engine
```



---

## References

```
[1] L. Yang et al. Depth Anything V2. arXiv:2406.09414, 2024.
[2] L. Yang et al. Depth Anything: Unleashing the power of large-scale unlabeled data. CVPR, 2024.
[3] N. Silberman et al. Indoor segmentation and support inference from RGB-D images. ECCV, 2012.
[4] J. Uhrig et al. Sparsity invariant CNNs. 3DV, 2017.
[5] R. Ranftl et al. Vision Transformers for Dense Prediction (DPT). ICCV, 2021.
[6] D. Eigen et al. Depth map prediction from a single image using a multi-scale deep network. NeurIPS, 2014.
[7] S.F. Bhat et al. ZoeDepth: Zero-shot transfer for combining relative and metric depth. arXiv:2302.12288, 2023.
[8] A. Bhoi. Monocular depth estimation: A survey. arXiv:1901.09402, 2019.
```

---

## Acknowledgements

This project was completed as the CS 4391 Computer Vision final project at the **University of Texas at Dallas**, Spring 2026. We thank **Professor Yu Xiang** for advising this work and the Depth Anything team for their excellent open-source release.

---

## License

This repository is released under the MIT License. The Depth Anything V2 model weights are subject to their own [license terms](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE).