"""
ablation_study.py
-----------------
Skeleton for comparing fine-tuned Depth-Anything-V2 variants
against each other and the baseline pretrained model.

Each "experiment" is a combination of:
  - model checkpoint  (baseline pretrained, or a fine-tuned .pt file)
  - dataset split     (your custom test set, or any other test set path)
  - optional label    (human-readable name shown in the results table)

Results are written to:
  ablation_results.json   machine-readable per-experiment metrics
  ablation_report.txt     human-readable comparison table

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Compare baseline vs one fine-tuned checkpoint on your custom test set:
python ablation_study.py \\
    --experiments \\
        baseline::checkpoints/depth_anything_v2_vits.pth::/path/to/processed \\
        finetuned_full::checkpoints/best_model_batch16_stratfull_encvits.pt::/path/to/processed \\
    --output_dir ./ablation_results

# Compare two fine-tuned runs against each other:
python ablation_study.py \\
    --experiments \\
        decoder_only::checkpoints/best_model_batch4_stratdecoder_only_encvits.pt::/path/to/processed \\
        full_finetune::checkpoints/best_model_batch16_stratfull_encvits.pt::/path/to/processed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPERIMENT STRING FORMAT
    "<label>::<checkpoint_path>::<processed_dir>"

    label           Name shown in the results table
    checkpoint_path Path to .pth (pretrained) or .pt (fine-tuned) weights
    processed_dir   Path to a processed/ directory with images/, depths/,
                    masks/, and splits/test.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np


_repo = Path(__file__).parent / "Depth-Anything-V2"
sys.path.insert(0, str(_repo if _repo.exists() else Path(__file__).parent)) 
try:
    from UTD_dataloader import DepthDataset
except ImportError:
    print("\n  Error: dataloader.py not found next to this script.\n")
    sys.exit(1)

from torch.utils.data import DataLoader


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
# CHECKPOINT LOADING
# Handles both:
#   .pth  pretrained DAv2 weights (bare state_dict)
#   .pt   fine-tuned checkpoints saved by train.py
#         (dict with "model", "encoder", "strategy" keys)
# ──────────────────────────────────────────────

def load_model_for_eval(checkpoint_path: Path, encoder: str, device: str):
    from depth_anything_v2.dpt import DepthAnythingV2
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    # Detect fine-tuned checkpoint vs bare pretrained weights
    if isinstance(ckpt, dict) and "model" in ckpt:
        # Fine-tuned checkpoint from train.py
        state_dict     = ckpt["model"]
        ckpt_encoder   = ckpt.get("encoder", encoder)
        ckpt_strategy  = ckpt.get("strategy", "unknown")
        print(f"    Fine-tuned checkpoint  encoder={ckpt_encoder}  strategy={ckpt_strategy}")
    else:
        # Bare pretrained state_dict
        state_dict   = ckpt
        ckpt_encoder = encoder
        print(f"    Pretrained checkpoint  encoder={ckpt_encoder}")

    model = DepthAnythingV2(**MODEL_CONFIGS[ckpt_encoder])
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    return model


# ──────────────────────────────────────────────
# METRICS
# All computed only on valid pixels (mask == 1).
# ──────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    Standard depth evaluation metrics.

    pred, target : [B, 1, H, W] float32, values normalised to [0, 1]
    mask         : [B, 1, H, W] float32, 1.0 = valid pixel

    Returns a dict of scalar metrics.

    ── ADD YOUR OWN METRICS HERE ──────────────────────
    This is the primary extension point. To add a metric:
      1. Compute it from p and t (already masked and clamped below)
      2. Add it to the returned dict with a descriptive key
      3. Add the key to METRIC_COLS in the report section below
    ────────────────────────────────────────────────────
    """
    eps   = 1e-6
    valid = mask > 0.5

    p = pred[valid].clamp(min=eps)
    t = target[valid].clamp(min=eps)

    if p.numel() == 0:
        return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "sq_rel", "rmse_log", "delta1", "delta2", "delta3"]}

    # ── Core metrics ──────────────────────────────────
    mae      = (p - t).abs().mean().item()
    rmse     = ((p - t) ** 2).mean().sqrt().item()
    abs_rel  = ((p - t).abs() / t).mean().item()
    sq_rel   = (((p - t) ** 2) / t).mean().item()
    rmse_log = ((torch.log(p) - torch.log(t)) ** 2).mean().sqrt().item()

    ratio    = torch.max(p / t, t / p)
    delta1   = (ratio < 1.25  ).float().mean().item() * 100
    delta2   = (ratio < 1.25**2).float().mean().item() * 100
    delta3   = (ratio < 1.25**3).float().mean().item() * 100

    # ── ADD CUSTOM METRICS BELOW ──────────────────────
    # Example: boundary accuracy (requires edge detection)
    # boundary_acc = compute_boundary_accuracy(pred, target, mask)

    return {
        "mae":      round(mae,      5),
        "rmse":     round(rmse,     5),
        "abs_rel":  round(abs_rel,  5),
        "sq_rel":   round(sq_rel,   5),
        "rmse_log": round(rmse_log, 5),
        "delta1":   round(delta1,   2),
        "delta2":   round(delta2,   2),
        "delta3":   round(delta3,   2),
    }


# ──────────────────────────────────────────────
# FORWARD PASS
# ──────────────────────────────────────────────

def model_forward(model: DepthAnythingV2, image: torch.Tensor, h: int, w: int) -> torch.Tensor:
    depth = model(image)

    if depth.dim() == 3:
        depth = depth.unsqueeze(1)

    if depth.shape[2] != h or depth.shape[3] != w:
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)

    # Normalise prediction to [0, 1] per sample
    b     = depth.shape[0]
    d_min = depth.view(b, -1).min(dim=1).values.view(b, 1, 1, 1)
    d_max = depth.view(b, -1).max(dim=1).values.view(b, 1, 1, 1)
    depth = (depth - d_min) / (d_max - d_min + 1e-6)

    return depth


# ──────────────────────────────────────────────
# SINGLE EXPERIMENT EVALUATION
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:         DepthAnythingV2,
    processed_dir: Path,
    device:        str,
    batch_size:    int  = 1,
    num_workers:   int  = 2,
) -> dict:
    """
    Run the model over the test split and return averaged metrics.

    ── SWAP DATASET HERE ──────────────────────────────
    To evaluate on a different dataset (NYUv2, KITTI, your
    second custom set, etc.) swap DepthDataset for your own
    Dataset class, or pass a different processed_dir.
    The only contract is that each batch returns:
        batch["image"]  [B, 3, H, W]  ImageNet-normalised
        batch["depth"]  [B, 1, H, W]  float32 [0, 1]
        batch["mask"]   [B, 1, H, W]  float32 {0, 1}
    ────────────────────────────────────────────────────
    """
    dataset = DepthDataset(processed_dir, split="test", augment=False)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = device == "cuda",
    )

    all_metrics = {k: [] for k in ["mae", "rmse", "abs_rel", "sq_rel", "rmse_log", "delta1", "delta2", "delta3"]}

    for batch in loader:
        image  = batch["image"].to(device)
        target = batch["depth"].to(device)
        mask   = batch["mask"].to(device)
        h, w   = target.shape[2], target.shape[3]

        pred = model_forward(model, image, h, w)
        m    = compute_metrics(pred, target, mask)

        for k, v in m.items():
            all_metrics[k].append(v)

    return {k: round(float(np.mean(v)), 5) for k, v in all_metrics.items() if v}


# ──────────────────────────────────────────────
# RESULTS TABLE
# ──────────────────────────────────────────────

# ── ADD / REMOVE COLUMNS HERE ─────────────────
# To add a new metric column, add its key here
# (must match the key returned by compute_metrics)
METRIC_COLS = ["mae", "rmse", "abs_rel", "sq_rel", "rmse_log", "delta1", "delta2", "delta3"]
# ──────────────────────────────────────────────

# Which direction is "better" for each metric
LOWER_IS_BETTER = {"mae", "rmse", "abs_rel", "sq_rel", "rmse_log"}
HIGHER_IS_BETTER = {"delta1", "delta2", "delta3"}


def find_best(results: list[dict], col: str) -> str:
    """Return the label of the best experiment for a given metric column."""
    values = [(r["label"], r["metrics"].get(col, None)) for r in results if r["metrics"].get(col) is not None]
    if not values:
        return ""
    if col in LOWER_IS_BETTER:
        return min(values, key=lambda x: x[1])[0]
    return max(values, key=lambda x: x[1])[0]


def build_report(results: list[dict]) -> str:
    col_w     = 12
    label_w   = max(len(r["label"]) for r in results) + 2
    best      = {col: find_best(results, col) for col in METRIC_COLS}

    lines = []
    lines.append("=" * (label_w + col_w * len(METRIC_COLS) + 4))
    lines.append("ABLATION STUDY RESULTS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * (label_w + col_w * len(METRIC_COLS) + 4))
    lines.append("")

    # Header
    header = f"{'Experiment':<{label_w}}" + "".join(f"{c:>{col_w}}" for c in METRIC_COLS)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for r in results:
        row = f"{r['label']:<{label_w}}"
        for col in METRIC_COLS:
            val = r["metrics"].get(col, "N/A")
            cell = f"{val:.5f}" if isinstance(val, float) else str(val)
            # Mark best value with *
            marker = "*" if r["label"] == best[col] else " "
            row += f"{marker + cell:>{col_w}}"
        lines.append(row)

    lines.append("")
    lines.append("* = best value for that metric")
    lines.append("")
    lines.append("Lower is better:  " + ", ".join(LOWER_IS_BETTER))
    lines.append("Higher is better: " + ", ".join(HIGHER_IS_BETTER))
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def parse_experiment(s: str) -> dict:
    """
    Parse a single experiment string.
    Format: "<label>::<checkpoint_path>::<processed_dir>"
    """
    parts = s.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid experiment string: '{s}'\n"
            f"Expected format: label::checkpoint_path::processed_dir"
        )
    return {
        "label":          parts[0].strip(),
        "checkpoint":     Path(parts[1].strip()),
        "processed_dir":  Path(parts[2].strip()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study — compare multiple depth model checkpoints.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        metavar="LABEL::CKPT::PROCESSED_DIR",
        help=(
            "One or more experiment strings.\n"
            "Format: label::checkpoint_path::processed_dir\n"
            "Example:\n"
            "  baseline::checkpoints/depth_anything_v2_vits.pth::/data/processed\n"
            "  finetuned::checkpoints/best_model_batch16_stratfull_encvits.pt::/data/processed"
        ),
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Encoder size (used for pretrained .pth checkpoints — fine-tuned .pt files store this internally)",
    )
    parser.add_argument(
        "--batch_size",  type=int, default=1,
        help="Batch size for evaluation (default: 1)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader workers (default: 2)",
    )
    parser.add_argument(
        "--output_dir",  type=str, default="./ablation_results",
        help="Where to save results (default: ./ablation_results)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"\n{'='*55}")
    print(f"ABLATION STUDY")
    print(f"{'='*55}")
    print(f"  Device:      {DEVICE}")
    print(f"  Experiments: {len(args.experiments)}")
    print(f"  Output dir:  {output_dir}\n")

    experiments = [parse_experiment(s) for s in args.experiments]
    all_results = []

    for exp in experiments:
        print(f"── {exp['label']} ──────────────────────────────────────")
        print(f"  Checkpoint:    {exp['checkpoint']}")
        print(f"  Processed dir: {exp['processed_dir']}")

        try:
            model   = load_model_for_eval(exp["checkpoint"], args.encoder, DEVICE)
            metrics = evaluate(
                model,
                exp["processed_dir"],
                DEVICE,
                args.batch_size,
                args.num_workers,
            )

            all_results.append({
                "label":          exp["label"],
                "checkpoint":     str(exp["checkpoint"]),
                "processed_dir":  str(exp["processed_dir"]),
                "metrics":        metrics,
            })

            print(f"  Results:")
            for k, v in metrics.items():
                print(f"    {k:<12} {v}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "label":    exp["label"],
                "metrics":  {},
                "error":    str(e),
            })

        print()

    # ── Save JSON results ──
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "device":       DEVICE,
            "results":      all_results,
        }, f, indent=2)

    # ── Save text report ──
    report     = build_report(all_results)
    txt_path   = output_dir / "ablation_report.txt"
    with open(txt_path, "w") as f:
        f.write(report)

    print(report)
    print(f"  JSON results: {json_path}")
    print(f"  Text report:  {txt_path}\n")


if __name__ == "__main__":
    main()