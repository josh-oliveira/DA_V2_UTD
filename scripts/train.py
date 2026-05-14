"""
train.py
--------


Fine-tunes Depth-Anything-V2 ViT-S (Small) on your custom dataset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Decoder-only, all three losses (recommended starting point):
python train.py \\
    --processed_dir /path/to/processed \\
    --strategy      decoder_only

# Full fine-tune with SILog only:
python train.py \\
    --processed_dir /path/to/processed \\
    --strategy      full \\
    --losses        silog

# Full fine-tune, custom loss weights, longer run:
python train.py \\
    --processed_dir /path/to/processed \\
    --strategy      full \\
    --losses        silog edge ssim \\
    --loss_weights  1.0 0.5 0.15 \\
    --epochs        50 \\
    --batch_size    4 \\
    --lr            1e-5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FINE-TUNING STRATEGIES
  decoder_only  Freeze the DINOv2 encoder, train the DPT head only.
                Fast, stable, works well with small datasets (<200 images).
                LR: 1e-4 on decoder.

  full          Update all weights.
                Better ceiling quality but needs more data and longer training.
                LR: 1e-5 on encoder, 1e-4 on decoder (differential LR).

LOSS FUNCTIONS (mix and match via --losses)
  silog         Scale-invariant log loss. Primary depth metric loss.
                Tolerates the scale ambiguity of relative depth.
  edge          Edge-aware gradient loss. Preserves sharp object boundaries.
                Weights depth gradient error by RGB image edges.
  ssim          Structural similarity loss. Preserves local depth patterns.
                Complements SILog which operates pixel-independently.

CHECKPOINTS
  Saved to:  <processed_dir>/checkpoints/
    best_model.pth        Best validation loss across all epochs
    latest_model.pth      Most recent epoch (for resuming)
    training_log.json     Full per-epoch metrics

Requirements:
    torch, torchvision, opencv-python, pillow, numpy
    Depth-Anything-V2 cloned and installed next to this script
    dataloader.py in the same folder as this script
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# ── DAv2 import ──
sys.path.insert(0, str(Path(__file__).parent / "Depth-Anything-V2"))
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print(
        "\n  Error: could not import DepthAnythingV2.\n"
        "  Clone and install Depth-Anything-V2 next to this script:\n"
        "    git clone https://github.com/DepthAnything/Depth-Anything-V2.git\n"
        "    cd Depth-Anything-V2 && pip install -e . && cd ..\n"
    )
    sys.exit(1)

# ── Local dataloader import ──
sys.path.insert(0, str(Path(__file__).parent))
try:
    from UTD_dataloader import build_dataloaders
except ImportError:
    print("\n  Error: dataloader.py not found. Place it in the same folder as train.py.\n")
    sys.exit(1)


# ──────────────────────────────────────────────
# MODEL CONFIG
# ──────────────────────────────────────────────

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────

def load_model(encoder: str, device: str) -> DepthAnythingV2:
    weights_path = Path(__file__).parent / "checkpoints" / f"depth_anything_v2_{encoder}.pth"

    if not weights_path.exists():
        print(
            f"\n  Error: weights not found at {weights_path}\n"
            f"  Download depth_anything_v2_{encoder}.pth from:\n"
            f"  https://huggingface.co/depth-anything/Depth-Anything-V2-Small\n"
            f"  and place it in checkpoints/ next to this script.\n"
        )
        sys.exit(1)

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model = model.to(device)

    return model


# ──────────────────────────────────────────────
# FREEZE STRATEGY
# ──────────────────────────────────────────────

def apply_strategy(model: DepthAnythingV2, strategy: str) -> list[dict]:
    """
    Apply the chosen fine-tuning strategy and return parameter groups
    with per-group learning rates for the optimiser.

    decoder_only:
        Freezes model.pretrained (DINOv2 encoder).
        Only model.depth_head (DPT decoder) is updated.

    full:
        All weights updated.
        Encoder gets a lower LR (1e-5) to avoid catastrophic forgetting.
        Decoder gets a higher LR (1e-4) for faster adaptation.
    """
    if strategy == "decoder_only":
        # Freeze encoder
        for param in model.pretrained.parameters():
            param.requires_grad = False
        # Ensure decoder is trainable
        for param in model.depth_head.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  Strategy:    decoder_only")
        print(f"  Trainable:   {trainable/1e6:.2f}M / {total/1e6:.2f}M parameters")

        return [
            {"params": model.depth_head.parameters(), "lr": 1e-4, "name": "decoder"},
        ]

    elif strategy == "full":
        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters())
        print(f"  Strategy:    full fine-tune")
        print(f"  Trainable:   {trainable/1e6:.2f}M parameters (all)")

        # Differential learning rates — encoder slower to avoid forgetting
        return [
            {"params": model.pretrained.parameters(), "lr": 1e-5, "name": "encoder"},
            {"params": model.depth_head.parameters(), "lr": 1e-4, "name": "decoder"},
        ]

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'decoder_only' or 'full'.")


# ──────────────────────────────────────────────
# LOSS FUNCTIONS
# ──────────────────────────────────────────────

def silog_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, lam: float = 0.85) -> torch.Tensor:
    """
    Scale-Invariant Log Loss (SILog).

    The standard loss for relative depth estimation. Tolerates the global
    scale ambiguity of affine-invariant depth — two depth maps that are
    related by a multiplicative constant produce zero loss.

    pred, target: [B, 1, H, W] float32, values in [0, 1]
    mask:         [B, 1, H, W] float32, 1.0 = valid pixel

    Reference: Eigen et al. 2014, "Depth Map Prediction from a Single Image"
    """
    eps   = 1e-6
    valid = mask > 0.5

    # Clamp to avoid log(0)
    pred_c   = pred.clamp(min=eps)
    target_c = target.clamp(min=eps)

    d = torch.log(pred_c) - torch.log(target_c)           # log-difference

    # Apply validity mask
    d = d * valid.float()
    n = valid.float().sum(dim=[1, 2, 3]).clamp(min=1)     # valid pixel count per image

    d_mean = d.sum(dim=[1, 2, 3]) / n
    d_var  = (d ** 2).sum(dim=[1, 2, 3]) / n

    # SILog = sqrt(variance - lambda * mean^2)
    loss = torch.sqrt((d_var - lam * d_mean ** 2).clamp(min=0) + eps)

    return loss.mean()


def edge_aware_loss(pred: torch.Tensor, target: torch.Tensor, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Edge-Aware Gradient Loss.

    Computes L1 error between depth prediction gradients and target depth
    gradients, weighted by the inverse of image edge strength. This means:
      - Where the RGB image has strong edges (object boundaries), the loss
        strongly penalises gradient mismatch — sharp depth transitions are enforced.
      - Where the RGB image is smooth (flat surfaces), gradient mismatch is
        down-weighted — we tolerate soft depth transitions there.

    pred, target: [B, 1, H, W] float32
    image:        [B, 3, H, W] float32  ImageNet-normalised RGB
    mask:         [B, 1, H, W] float32
    """
    def gradient(x: torch.Tensor):
        # Sobel-style finite differences
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]   # [B, C, H, W-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]   # [B, C, H-1, W]
        return dx, dy

    # Depth gradients
    pred_dx,   pred_dy   = gradient(pred)
    target_dx, target_dy = gradient(target)

    # Image gradients (use mean across channels for edge weight)
    img_gray = image.mean(dim=1, keepdim=True)   # [B, 1, H, W]
    img_dx,  img_dy      = gradient(img_gray)

    # Edge weight: strong image edge -> weight=1, smooth region -> weight~0
    # exp(-|∇I|) suppresses the loss where the image is already smooth
    weight_x = torch.exp(-img_dx.abs())           # [B, 1, H, W-1]
    weight_y = torch.exp(-img_dy.abs())           # [B, 1, H-1, W]

    # Validity mask trimmed to gradient size
    mask_x = mask[:, :, :, 1:]                   # [B, 1, H, W-1]
    mask_y = mask[:, :, 1:, :]                   # [B, 1, H-1, W]

    loss_x = (weight_x * (pred_dx - target_dx).abs() * mask_x).sum()
    loss_y = (weight_y * (pred_dy - target_dy).abs() * mask_y).sum()

    n = mask.sum().clamp(min=1)

    return (loss_x + loss_y) / n


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Structural Similarity (SSIM) Loss.

    SSIM measures the perceptual similarity of two images across three
    dimensions: luminance, contrast, and structure. As a depth loss it
    captures local spatial patterns that pixel-wise losses miss — two
    depth maps with the same per-pixel error but different spatial
    structure produce very different SSIM values.

    Loss = 1 - SSIM(pred, target), so lower is better.

    pred, target: [B, 1, H, W] float32
    mask:         [B, 1, H, W] float32
    """
    C1 = 0.01 ** 2   # stability constant for luminance
    C2 = 0.03 ** 2   # stability constant for contrast

    # Gaussian kernel for local statistics
    def gaussian_kernel(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        return kernel / kernel.sum()

    kernel = gaussian_kernel(window_size).to(pred.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)               # [1, 1, k, k]

    pad = window_size // 2

    def local_mean(x):
        return F.conv2d(x, kernel, padding=pad)

    mu_p  = local_mean(pred)
    mu_t  = local_mean(target)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p2  = local_mean(pred  * pred)  - mu_p2
    sigma_t2  = local_mean(target * target) - mu_t2
    sigma_pt  = local_mean(pred  * target) - mu_pt

    numerator   = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
    denominator = (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2)

    ssim_map = numerator / denominator.clamp(min=1e-8)      # [B, 1, H, W]

    # Apply validity mask and compute mean
    valid    = mask > 0.5
    ssim_val = (ssim_map * valid.float()).sum() / valid.float().sum().clamp(min=1)

    return 1.0 - ssim_val


class DepthLoss(nn.Module):
    """
    Composite depth loss combining SILog, edge-aware, and SSIM terms.
    Active losses and their weights are configured at construction time.

    Args:
        losses:       List of active loss names. Subset of ['silog', 'edge', 'ssim']
        loss_weights: Corresponding scalar weights for each loss.
                      Defaults: silog=1.0, edge=0.5, ssim=0.15
    """

    DEFAULTS = {"silog": 1.0, "edge": 0.5, "ssim": 0.15}

    def __init__(self, losses: list[str], loss_weights: list[float] | None = None):
        super().__init__()

        valid = {"silog", "edge", "ssim"}
        for name in losses:
            if name not in valid:
                raise ValueError(f"Unknown loss '{name}'. Choose from {valid}.")

        self.active = losses

        if loss_weights is not None:
            if len(loss_weights) != len(losses):
                raise ValueError("--loss_weights must have the same number of values as --losses")
            self.weights = {name: w for name, w in zip(losses, loss_weights)}
        else:
            self.weights = {name: self.DEFAULTS.get(name, 1.0) for name in losses}

        print(f"  Losses:")
        for name in self.active:
            print(f"    {name:6s}  weight={self.weights[name]}")

    def forward(
        self,
        pred:   torch.Tensor,    # [B, 1, H, W]
        target: torch.Tensor,    # [B, 1, H, W]
        mask:   torch.Tensor,    # [B, 1, H, W]
        image:  torch.Tensor,    # [B, 3, H, W]  needed for edge loss
    ) -> tuple[torch.Tensor, dict]:

        total      = torch.tensor(0.0, device=pred.device)
        components = {}

        if "silog" in self.active:
            l = silog_loss(pred, target, mask)
            total = total + self.weights["silog"] * l
            components["silog"] = l.item()

        if "edge" in self.active:
            l = edge_aware_loss(pred, target, image, mask)
            total = total + self.weights["edge"] * l
            components["edge"] = l.item()

        if "ssim" in self.active:
            l = ssim_loss(pred, target, mask)
            total = total + self.weights["ssim"] * l
            components["ssim"] = l.item()

        components["total"] = total.item()
        return total, components


# ──────────────────────────────────────────────
# LEARNING RATE SCHEDULER
# Cosine annealing with linear warmup
# ──────────────────────────────────────────────

def build_scheduler(optimizer: AdamW, warmup_epochs: int, total_epochs: int) -> LambdaLR:
    """
    Linear warmup for the first `warmup_epochs`, then cosine annealing
    to zero over the remaining epochs.

    Warmup prevents large gradient updates at the start of training
    from immediately destroying the pretrained encoder weights.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────
# VALIDATION METRICS
# ──────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    Compute standard depth evaluation metrics on valid pixels.

    For relative (normalised) depth:
      MAE    — Mean absolute error in normalised depth units.
      RMSE   — Root mean squared error.
      AbsRel — Mean absolute relative error.
      delta1 — % pixels where max(pred/target, target/pred) < 1.25

    All computed only on pixels where mask == 1.
    """
    eps   = 1e-6
    valid = mask > 0.5

    p = pred[valid].clamp(min=eps)
    t = target[valid].clamp(min=eps)

    if p.numel() == 0:
        return {"mae": 0.0, "rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}

    mae     = (p - t).abs().mean().item()
    rmse    = ((p - t) ** 2).mean().sqrt().item()
    abs_rel = ((p - t).abs() / t).mean().item()

    ratio   = torch.max(p / t, t / p)
    delta1  = (ratio < 1.25).float().mean().item() * 100

    return {
        "mae":     round(mae,     5),
        "rmse":    round(rmse,    5),
        "abs_rel": round(abs_rel, 5),
        "delta1":  round(delta1,  2),
    }


# ──────────────────────────────────────────────
# FORWARD PASS HELPER
# ──────────────────────────────────────────────

def model_forward(model: DepthAnythingV2, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Run the model and resize output to match target spatial dimensions.

    DAv2 internally pads/resizes input for the encoder patch size.
    The output depth map may differ in spatial size from the input,
    so we bilinearly upsample back to match the target/mask size.
    """
    depth = model(image)                                    # [B, H', W'] or [B, 1, H', W']

    # Normalise output shape to [B, 1, H, W]
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)

    if depth.shape[2] != target_h or depth.shape[3] != target_w:
        depth = F.interpolate(depth, size=(target_h, target_w), mode="bilinear", align_corners=True)

    # Normalise prediction to [0, 1] per sample — matches our normalised labels
    B = depth.shape[0]
    d_min = depth.view(B, -1).min(dim=1).values.view(B, 1, 1, 1)
    d_max = depth.view(B, -1).max(dim=1).values.view(B, 1, 1, 1)
    depth = (depth - d_min) / (d_max - d_min + 1e-6)

    return depth


# ──────────────────────────────────────────────
# TRAIN / VAL LOOPS
# ──────────────────────────────────────────────

def train_epoch(
    model:      DepthAnythingV2,
    loader:     torch.utils.data.DataLoader,
    criterion:  DepthLoss,
    optimizer:  AdamW,
    device:     str,
    epoch:      int,
    total_epochs: int,
) -> dict:

    model.train()
    epoch_loss   = 0.0
    epoch_comps  = {}
    n_batches    = 0
    t_start      = time.time()

    for batch_idx, batch in enumerate(loader):
        image  = batch["image"].to(device)   # [B, 3, H, W]
        target = batch["depth"].to(device)   # [B, 1, H, W]
        mask   = batch["mask"].to(device)    # [B, 1, H, W]

        H, W = target.shape[2], target.shape[3]

        optimizer.zero_grad()

        pred = model_forward(model, image, H, W)

        loss, components = criterion(pred, target, mask, image)

        loss.backward()

        # Gradient clipping — stabilises early training, especially for full fine-tune
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()

        epoch_loss += loss.item()
        for k, v in components.items():
            epoch_comps[k] = epoch_comps.get(k, 0.0) + v
        n_batches += 1

        # Progress print every 5 batches
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch:3d}/{total_epochs}  "
                f"batch {batch_idx+1:3d}/{len(loader)}  "
                f"loss={loss.item():.4f}  "
                f"elapsed={elapsed:.1f}s"
            )

    avg = {k: v / max(n_batches, 1) for k, v in epoch_comps.items()}
    avg["loss"] = epoch_loss / max(n_batches, 1)
    return avg


@torch.no_grad()
def val_epoch(
    model:     DepthAnythingV2,
    loader:    torch.utils.data.DataLoader,
    criterion: DepthLoss,
    device:    str,
) -> dict:

    model.eval()
    epoch_loss  = 0.0
    epoch_comps = {}
    all_metrics = {"mae": [], "rmse": [], "abs_rel": [], "delta1": []}
    n_batches   = 0

    for batch in loader:
        image  = batch["image"].to(device)
        target = batch["depth"].to(device)
        mask   = batch["mask"].to(device)

        H, W = target.shape[2], target.shape[3]

        pred = model_forward(model, image, H, W)

        loss, components = criterion(pred, target, mask, image)

        epoch_loss += loss.item()
        for k, v in components.items():
            epoch_comps[k] = epoch_comps.get(k, 0.0) + v

        metrics = compute_metrics(pred, target, mask)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in epoch_comps.items()}
    avg["loss"] = epoch_loss / max(n_batches, 1)

    for k, v in all_metrics.items():
        avg[k] = round(float(np.mean(v)), 5) if v else 0.0

    return avg


# ──────────────────────────────────────────────
# CHECKPOINT HELPERS
# ──────────────────────────────────────────────

def save_checkpoint(model: DepthAnythingV2, path: Path, epoch: int, val_loss: float, args):
    torch.save({
        "epoch":      epoch,
        "val_loss":   val_loss,
        "model":      model.state_dict(),
        "encoder":    args.encoder,
        "strategy":   args.strategy,
        "losses":     args.losses,
    }, str(path))


def load_checkpoint(model: DepthAnythingV2, path: Path, device: str) -> tuple[int, float]:
    ckpt      = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_loss   = ckpt.get("val_loss", float("inf"))
    print(f"  Resumed from epoch {ckpt['epoch']}  val_loss={best_loss:.5f}")
    return start_epoch, best_loss


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Depth-Anything-V2 on a custom depth dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Paths ──
    parser.add_argument(
        "--processed_dir", type=str, required=True,
        help="Path to the processed/ directory (must contain images/, depths/, masks/, splits/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save checkpoints. Default: <processed_dir>/checkpoints/",
    )

    # ── Model ──
    parser.add_argument(
        "--encoder", type=str, default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Model encoder size (default: vits)",
    )

    # ── Fine-tuning strategy ──
    parser.add_argument(
        "--strategy", type=str, default="decoder_only",
        choices=["decoder_only", "full"],
        help=(
            "decoder_only  Freeze encoder, train DPT head only. Fast, stable.\n"
            "full           Update all weights with differential LR."
        ),
    )

    # ── Loss functions ──
    parser.add_argument(
        "--losses", type=str, nargs="+",
        default=["silog", "edge", "ssim"],
        choices=["silog", "edge", "ssim"],
        help="Loss functions to combine. Default: silog edge ssim",
    )
    parser.add_argument(
        "--loss_weights", type=float, nargs="+", default=None,
        help=(
            "Scalar weight per loss (same order as --losses).\n"
            "Defaults: silog=1.0  edge=0.5  ssim=0.15"
        ),
    )

    # ── Training hyper-parameters ──
    parser.add_argument("--epochs",      type=int,   default=30,   help="Total training epochs (default: 30)")
    parser.add_argument("--batch_size",  type=int,   default=4,    help="Batch size (default: 4)")
    parser.add_argument("--lr",          type=float, default=None, help="Override learning rate for all param groups")
    parser.add_argument("--warmup",      type=int,   default=3,    help="Warmup epochs (default: 3)")
    parser.add_argument("--num_workers", type=int,   default=4,    help="DataLoader workers (default: 4)")

    # ── Resume ──
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from",
    )

    args = parser.parse_args()

    # ── Resolve paths ──
    processed_dir = Path(args.processed_dir).resolve()
    output_dir    = Path(args.output_dir).resolve() if args.output_dir else processed_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──
    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ── Print run config ──
    print(f"\n{'='*55}")
    print(f"DEPTH-ANYTHING-V2 FINE-TUNING")
    print(f"{'='*55}")
    print(f"  Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Processed dir: {processed_dir}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Device:        {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  Encoder:       {args.encoder}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Warmup epochs: {args.warmup}")

    # ── Load model ──
    print(f"\n  Loading model...")
    model = load_model(args.encoder, DEVICE)

    # ── Apply fine-tuning strategy and get param groups ──
    param_groups = apply_strategy(model, args.strategy)

    # ── Override LR if specified ──
    if args.lr is not None:
        for g in param_groups:
            g["lr"] = args.lr
        print(f"  LR override:   {args.lr}")

    # ── Loss function ──
    criterion = DepthLoss(losses=args.losses, loss_weights=args.loss_weights)

    # ── Optimiser ──
    optimizer = AdamW(param_groups, weight_decay=1e-4)

    # ── LR scheduler ──
    scheduler = build_scheduler(optimizer, args.warmup, args.epochs)

    # ── DataLoaders ──
    print(f"\n  Building dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(
        processed_dir = processed_dir,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        pin_memory    = DEVICE == "cuda",
    )

    # ── Resume from checkpoint if requested ──
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch, best_val_loss = load_checkpoint(model, resume_path, DEVICE)
        else:
            print(f"  Warning: resume checkpoint not found at {resume_path} — starting fresh.")

    # ── Training log ──
    run_tag     = f"batch{args.batch_size}_strat{args.strategy}_enc{args.encoder}_epoch{args.epochs}"
    log_path    = output_dir / f"training_log_{run_tag}.json"
    best_path   = output_dir / f"best_model_{run_tag}.pt"
    latest_path = output_dir / f"latest_model_{run_tag}.pt"

    training_log = {
        "config": {
            "encoder":    args.encoder,
            "strategy":   args.strategy,
            "losses":     args.losses,
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "warmup":     args.warmup,
            "device":     DEVICE,
            "started_at": datetime.now().isoformat(),
        },
        "epochs": [],
    }

    # ── Training loop ──
    print(f"\n{'─'*55}")
    print(f"  Training for {args.epochs} epochs")
    print(f"{'─'*55}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        current_lrs = [g["lr"] * scheduler.get_last_lr()[0]
                       if hasattr(scheduler, "get_last_lr") else g["lr"]
                       for g in optimizer.param_groups]

        print(f"\n── Epoch {epoch}/{args.epochs}  LR={[f'{lr:.2e}' for lr in current_lrs]} ──")

        # Train
        train_stats = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch, args.epochs
        )

        # Validate
        val_stats = val_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step()

        # Save latest checkpoint every epoch
        save_checkpoint(model, latest_path, epoch, val_stats["loss"], args)

        # Save best checkpoint
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(model, best_path, epoch, val_stats["loss"], args)
            best_marker = "  ← best"
        else:
            best_marker = ""

        # Log
        epoch_record = {
            "epoch":      epoch,
            "train":      train_stats,
            "val":        val_stats,
        }
        training_log["epochs"].append(epoch_record)

        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        # Print epoch summary
        print(
            f"\n  Summary epoch {epoch:3d}:\n"
            f"    train  loss={train_stats['loss']:.5f}"
            + (f"  silog={train_stats.get('silog', 0):.5f}" if "silog" in args.losses else "")
            + (f"  edge={train_stats.get('edge', 0):.5f}"   if "edge"  in args.losses else "")
            + (f"  ssim={train_stats.get('ssim', 0):.5f}"   if "ssim"  in args.losses else "")
            + f"\n    val    loss={val_stats['loss']:.5f}"
            + f"  mae={val_stats.get('mae', 0):.5f}"
            + f"  rmse={val_stats.get('rmse', 0):.5f}"
            + f"  abs_rel={val_stats.get('abs_rel', 0):.5f}"
            + f"  delta1={val_stats.get('delta1', 0):.2f}%"
            + best_marker
        )

    print(f"\n{'='*55}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*55}")
    print(f"  Best val loss: {best_val_loss:.5f}")
    print(f"  Best model:    {best_path}")
    print(f"  Training log:  {log_path}")
    print()


if __name__ == "__main__":
    main()