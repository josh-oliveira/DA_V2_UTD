"""
generate_depths.py
------------------
Step 4 of the Depth-Anything-V2 fine-tuning pipeline.

Runs the pretrained Depth-Anything-V2 ViT-L model over every processed
image in processed/images/ and saves:
  - Float32 depth maps  → processed/depths/<name>.npy
  - Binary validity masks → processed/masks/<name>.png
  - A depth metadata log → processed/depth_meta.json

The model is used in inference-only mode. No training happens here.
Larger model = better pseudo-labels = better fine-tuning downstream.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP (run once before this script)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Install PyTorch 2.6+ (required for RTX 5070 Ti / Blackwell):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

2. Install remaining dependencies:
   pip install opencv-python pillow numpy matplotlib huggingface_hub

3. Clone Depth-Anything-V2 into the same folder as this script:
   git clone https://github.com/DepthAnything/Depth-Anything-V2.git

4. Install DAv2 package:
   cd Depth-Anything-V2 && pip install -e . && cd ..

The ViT-L weights (~1.3GB) are downloaded automatically from HuggingFace
on first run and cached in ~/.cache/huggingface/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
    python generate_depths.py --image_dir /path/to/processed/images

    # All options:
    python generate_depths.py \\
        --image_dir    /path/to/processed/images \\
        --output_dir   /path/to/processed          \\   # depths/ and masks/ created here
        --metadata     /path/to/processed/metadata.json \\
        --encoder      vitl                         \\   # vitl | vitb | vits
        --input_size   518                          \\   # must match your resize target
        --visualize                                     # also save colourmap PNGs for review

Requirements:
    torch>=2.6, opencv-python, pillow, numpy, matplotlib, huggingface_hub
    Depth-Anything-V2 cloned and installed (see SETUP above)
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent))
from depth_anything_v2.dpt import DepthAnythingV2

# ── Suppress minor warnings from the DAv2 repo ──
warnings.filterwarnings("ignore", category=UserWarning)

# ── Torch import with version check ──
try:
    import torch
    import torchvision
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version < (2, 6):
        print(
            f"\n  WARNING: PyTorch {torch.__version__} detected.\n"
            f"  The RTX 5070 Ti (Blackwell) requires PyTorch >= 2.6 for full GPU support.\n"
            f"  Run: pip install torch torchvision "
            f"--index-url https://download.pytorch.org/whl/cu124\n"
        )
except ImportError:
    print("Error: PyTorch not installed. See SETUP instructions at top of this file.")
    sys.exit(1)




# ──────────────────────────────────────────────
# MODEL CONFIG
# ──────────────────────────────────────────────

ENCODER_CONFIGS = {
    "vits": {
        "encoder":      "vits",
        "features":     64,
        "out_channels": [48, 96, 192, 384],
        "hf_repo":      "depth-anything/Depth-Anything-V2-Small",
        "hf_filename":  "depth_anything_v2_vits.pth",
        "vram_gb":      2,
    },
    "vitb": {
        "encoder":      "vitb",
        "features":     128,
        "out_channels": [96, 192, 384, 768],
        "hf_repo":      "depth-anything/Depth-Anything-V2-Base",
        "hf_filename":  "depth_anything_v2_vitb.pth",
        "vram_gb":      4,
    },
    "vitl": {
        "encoder":      "vitl",
        "features":     256,
        "out_channels": [256, 512, 1024, 1024],
        "hf_repo":      "depth-anything/Depth-Anything-V2-Large",
        "hf_filename":  "depth_anything_v2_vitl.pth",
        "vram_gb":      8,
    },
}


# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────

def load_model(encoder: str, device: torch.device, weights_path: str) -> DepthAnythingV2:
    cfg = ENCODER_CONFIGS[encoder]

    print(f"\n  Loading Depth-Anything-V2 ({encoder.upper()})...")
    print(f"  Estimated VRAM usage: ~{cfg['vram_gb']}GB")
    print(f"  Loading weights from: {weights_path}")

    model = DepthAnythingV2(
        encoder=cfg["encoder"],
        features=cfg["features"],
        out_channels=cfg["out_channels"],
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded: {param_count:.1f}M parameters  device={device}\n")

    return model


# ──────────────────────────────────────────────
# DEPTH INFERENCE
# ──────────────────────────────────────────────

def infer_depth(
    model: DepthAnythingV2,
    image_path: Path,
    input_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run depth inference on a single image.

    Returns raw float32 depth map resized to (input_size, input_size).
    Values are affine-invariant (relative) — not in metres.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"OpenCV could not read: {image_path}")

    # DAv2 infer_image expects BGR uint8 numpy array
    # It handles its own internal normalisation
    with torch.no_grad():
        depth = model.infer_image(img, input_size)

    # depth is a numpy float32 array, same H×W as the resized input
    return depth.astype(np.float32)


# ──────────────────────────────────────────────
# DEPTH POST-PROCESSING
# ──────────────────────────────────────────────

def normalise_depth(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Normalise a raw depth map to [0, 1] using per-image min-max.

    DAv2 relative depth output is affine-invariant — the absolute
    scale and shift are arbitrary and differ per image. Normalising
    to [0, 1] makes loss computation stable across the dataset.

    Returns:
        normalised depth (float32, [0, 1])
        stats dict with raw min/max for potential de-normalisation later
    """
    d_min = float(depth.min())
    d_max = float(depth.max())
    d_range = d_max - d_min

    if d_range < 1e-6:
        # Degenerate case — completely flat depth (blank wall, pure sky)
        normalised = np.zeros_like(depth)
    else:
        normalised = (depth - d_min) / d_range

    stats = {
        "raw_min":   d_min,
        "raw_max":   d_max,
        "raw_range": d_range,
        "is_degenerate": d_range < 1e-6,
    }

    return normalised.astype(np.float32), stats


def build_validity_mask(depth_norm: np.ndarray) -> np.ndarray:
    """
    Create a binary validity mask (uint8, 0 or 255).

    Marks pixels as INVALID (0) when:
      - Depth is exactly 0.0 (sensor dropout, degenerate regions)
      - Depth is exactly 1.0 (clipped maximum — often sky or very far background)
      - Depth is NaN or Inf

    Everything else is marked VALID (255).

    During training the loss is computed only on valid pixels.
    Including invalid pixels corrupts gradients.
    """
    valid = np.ones(depth_norm.shape, dtype=bool)
    valid &= np.isfinite(depth_norm)
    valid &= depth_norm > 0.0
    valid &= depth_norm < 1.0

    mask = np.where(valid, 255, 0).astype(np.uint8)

    valid_pct = float(valid.mean() * 100)
    return mask, valid_pct


# ──────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────

COLOURMAP = "inferno"  # perceptually uniform, works well for depth


def save_colourmap(depth_norm: np.ndarray, mask: np.ndarray, out_path: Path):
    """
    Save a colourmap PNG for visual inspection.
    Invalid pixels are shown in grey.
    """
    coloured = cm.get_cmap(COLOURMAP)(depth_norm)[..., :3]  # HxWx3, float [0,1]
    coloured_uint8 = (coloured * 255).astype(np.uint8)

    # Grey out invalid pixels
    invalid = mask == 0
    coloured_uint8[invalid] = [80, 80, 80]

    Image.fromarray(coloured_uint8).save(str(out_path))


def save_side_by_side(rgb_path: Path, depth_norm: np.ndarray, mask: np.ndarray, out_path: Path):
    """
    Save a side-by-side comparison: original RGB | depth colourmap.
    Useful for quick visual QA of label quality.
    """
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    coloured = cm.get_cmap(COLOURMAP)(depth_norm)[..., :3]
    coloured_uint8 = (coloured * 255).astype(np.uint8)

    # Grey out invalid pixels in depth view
    invalid = mask == 0
    coloured_uint8[invalid] = [80, 80, 80]

    # Add a 4px white divider
    divider = np.ones((rgb.shape[0], 4, 3), dtype=np.uint8) * 220
    combined = np.concatenate([rgb, divider, coloured_uint8], axis=1)
    Image.fromarray(combined).save(str(out_path))


# ──────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ──────────────────────────────────────────────

def process_dataset(
    image_dir: Path,
    depths_dir: Path,
    masks_dir: Path,
    vis_dir: Path | None,
    model: DepthAnythingV2,
    input_size: int,
    device: torch.device,
    visualize: bool,
) -> dict:
    depths_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if visualize and vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    total = len(image_paths)
    print(f"  Processing {total} images...\n")

    depth_meta = {}
    failed     = []
    t_start    = time.time()

    for idx, img_path in enumerate(image_paths, start=1):
        stem = img_path.stem  # e.g. "000001"

        depth_path = depths_dir / f"{stem}.npy"
        mask_path  = masks_dir  / f"{stem}.png"

        try:
            t0 = time.time()

            # ── Inference ──
            raw_depth = infer_depth(model, img_path, input_size, device)

            # ── Normalise ──
            depth_norm, depth_stats = normalise_depth(raw_depth)

            # ── Validity mask ──
            mask, valid_pct = build_validity_mask(depth_norm)

            # ── Save depth map (float32 .npy) ──
            np.save(str(depth_path), depth_norm)

            # ── Save validity mask (uint8 PNG) ──
            cv2.imwrite(str(mask_path), mask)

            # ── Optional visualisation ──
            if visualize and vis_dir:
                vis_path = vis_dir / f"{stem}_compare.jpg"
                save_side_by_side(img_path, depth_norm, mask, vis_path)

            elapsed = time.time() - t0
            eta     = (time.time() - t_start) / idx * (total - idx)

            depth_meta[img_path.name] = {
                "sequential_name":  img_path.name,
                "depth_path":       str(depth_path),
                "mask_path":        str(mask_path),
                "depth_stats":      depth_stats,
                "valid_pixel_pct":  round(valid_pct, 2),
                "inference_sec":    round(elapsed, 3),
                "status":           "ok",
            }

            flag = " [DEGENERATE]" if depth_stats["is_degenerate"] else ""
            print(
                f"  [{idx:>4}/{total}]  {img_path.name}  "
                f"valid={valid_pct:.1f}%  "
                f"range=[{depth_stats['raw_min']:.3f}, {depth_stats['raw_max']:.3f}]  "
                f"{elapsed:.2f}s  ETA {eta:.0f}s{flag}"
            )

        except Exception as e:
            failed.append({"filename": img_path.name, "error": str(e)})
            depth_meta[img_path.name] = {
                "sequential_name": img_path.name,
                "status":          "failed",
                "error":           str(e),
            }
            print(f"  [ERROR]  {img_path.name}: {e}")

    total_time = time.time() - t_start
    return depth_meta, failed, total_time


# ──────────────────────────────────────────────
# METADATA & REPORT
# ──────────────────────────────────────────────

def write_depth_metadata(
    depth_meta: dict,
    failed: list,
    output_dir: Path,
    encoder: str,
    device: str,
    total_time: float,
):
    successful   = {k: v for k, v in depth_meta.items() if v["status"] == "ok"}
    failed_keys  = {k: v for k, v in depth_meta.items() if v["status"] == "failed"}

    # Dataset-level depth statistics
    if successful:
        valid_pcts  = [v["valid_pixel_pct"] for v in successful.values()]
        raw_ranges  = [v["depth_stats"]["raw_range"] for v in successful.values()]
        degenerate  = [k for k, v in successful.items() if v["depth_stats"]["is_degenerate"]]

        dataset_stats = {
            "mean_valid_pixel_pct":   round(float(np.mean(valid_pcts)),  2),
            "min_valid_pixel_pct":    round(float(np.min(valid_pcts)),   2),
            "mean_raw_depth_range":   round(float(np.mean(raw_ranges)),  4),
            "degenerate_images":      degenerate,
            "degenerate_count":       len(degenerate),
        }
    else:
        dataset_stats = {}

    report = {
        "generated_at":    datetime.now().isoformat(),
        "encoder":         encoder,
        "depth_type":      "relative (affine-invariant)",
        "normalisation":   "per-image min-max to [0, 1]",
        "device":          device,
        "total_time_sec":  round(total_time, 1),
        "summary": {
            "total":            len(depth_meta),
            "successful":       len(successful),
            "failed":           len(failed_keys),
        },
        "dataset_stats":   dataset_stats,
        "images":          depth_meta,
    }

    meta_path = output_dir / "depth_meta.json"
    with open(meta_path, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print(f"\n{'='*55}")
    print(f"DEPTH LABEL GENERATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Encoder:              {encoder.upper()}")
    print(f"  Device:               {device}")
    print(f"  Total time:           {total_time:.1f}s  ({total_time/len(depth_meta):.2f}s/image)")
    print(f"  Successful:           {len(successful)}")
    if failed_keys:
        print(f"  Failed:               {len(failed_keys)}")
        for k, v in failed_keys.items():
            print(f"    -> {k}: {v['error']}")
    if dataset_stats:
        print(f"  Mean valid pixels:    {dataset_stats['mean_valid_pixel_pct']:.1f}%")
        print(f"  Min valid pixels:     {dataset_stats['min_valid_pixel_pct']:.1f}%")
        if dataset_stats["degenerate_count"] > 0:
            print(f"  Degenerate images:    {dataset_stats['degenerate_count']}")
            print(f"    (flat/uniform depth — consider removing from training)")
            for name in dataset_stats["degenerate_images"]:
                print(f"    -> {name}")
    print(f"\n  Depths:    {output_dir / 'depths'}")
    print(f"  Masks:     {output_dir / 'masks'}")
    print(f"  Metadata:  {meta_path}")
    print()

    return meta_path


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate depth labels using Depth-Anything-V2."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to processed/images/ folder containing resized JPEGs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root output folder. depths/ and masks/ created inside. "
             "Default: parent of image_dir (i.e. processed/)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl"],
        help="Model encoder size (default: vitl — best label quality)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=518,
        help="Inference resolution — must match your resize target (default: 518)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save side-by-side RGB|depth PNGs to processed/vis/ for visual QA",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device: 'cuda', 'cpu'. Default: auto-detect.",
    )
    parser.add_argument(
    "--weights",
    type=str,
    required=True,
    help="Path to local .pth weights file",
    )
    args = parser.parse_args()

    # ── Resolve paths ──
    image_dir  = Path(args.image_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else image_dir.parent
    )
    depths_dir = output_dir / "depths"
    masks_dir  = output_dir / "masks"
    vis_dir    = output_dir / "vis" if args.visualize else None

    if not image_dir.exists():
        print(f"Error: image_dir not found: {image_dir}")
        sys.exit(1)

    # ── Device selection ──
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("\n  WARNING: CUDA not available — running on CPU. This will be slow.")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {gpu_name}  ({vram_gb:.1f}GB VRAM)")

    # ── Load model ──
    model = load_model(args.encoder, device, args.weights)

    # ── Print run config ──
    print(f"  Image dir:    {image_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Input size:   {args.input_size}x{args.input_size}")
    print(f"  Depth type:   relative (affine-invariant, normalised [0, 1])")
    print(f"  Visualise:    {'yes -> ' + str(vis_dir) if args.visualize else 'no'}")

    # ── Run ──
    depth_meta, failed, total_time = process_dataset(
        image_dir  = image_dir,
        depths_dir = depths_dir,
        masks_dir  = masks_dir,
        vis_dir    = vis_dir,
        model      = model,
        input_size = args.input_size,
        device     = device,
        visualize  = args.visualize,
    )

    # ── Save metadata ──
    write_depth_metadata(
        depth_meta = depth_meta,
        failed     = failed,
        output_dir = output_dir,
        encoder    = args.encoder,
        device     = str(device),
        total_time = total_time,
    )


if __name__ == "__main__":
    
    main()