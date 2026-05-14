"""
resize_images.py
----------------
Step 2 of the Depth-Anything-V2 fine-tuning pipeline.

Reads the audit_report.json produced by audit_images.py, processes only
images that passed the audit, resizes them to a consistent resolution using
resize-shorter-edge + center-crop, and renames them sequentially.

Usage:
    python resize_images.py --input_dir /path/to/raw/photos
                            --output_dir /path/to/dataset/processed/images
                            --audit_report /path/to/audit_report.json

    # To use a different target resolution (default 518x518):
    python resize_images.py --input_dir ... --target_size 392

Requirements:
    pip install opencv-python pillow numpy

Outputs:
    processed/images/
        000001.jpg
        000002.jpg
        ...
    processed/metadata.json    — maps new sequential name → original filename + crop info
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
# RESIZE LOGIC
# ──────────────────────────────────────────────

def resize_and_crop(img: np.ndarray, target_size: int) -> tuple[np.ndarray, dict]:
    """
    Resize shorter edge to target_size, then center crop to target_size × target_size.
    Maintains aspect ratio during resize — no stretching or squashing.

    Returns:
        cropped image (np.ndarray, HWC uint8)
        crop_meta dict with original size, resize size, and crop offset
    """
    h, w = img.shape[:2]

    # ── Step 1: Resize shorter edge to target_size ──
    if h < w:
        # Height is shorter edge
        new_h = target_size
        new_w = int(round(w * (target_size / h)))
    else:
        # Width is shorter edge (or square)
        new_w = target_size
        new_h = int(round(h * (target_size / w)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # ── Step 2: Center crop to target_size × target_size ──
    crop_y = (new_h - target_size) // 2
    crop_x = (new_w - target_size) // 2

    cropped = resized[crop_y:crop_y + target_size, crop_x:crop_x + target_size]

    crop_meta = {
        "original_width": w,
        "original_height": h,
        "resized_width": new_w,
        "resized_height": new_h,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "final_width": target_size,
        "final_height": target_size,
    }

    return cropped, crop_meta


def get_jpeg_quality(path: Path) -> int:
    """
    Try to read the original JPEG quality from EXIF/quantization tables.
    Falls back to a safe default of 95 if unreadable.
    """
    try:
        img = Image.open(path)
        # Pillow exposes quantization tables — use them to estimate quality
        if hasattr(img, "quantization"):
            # Rough heuristic: sum of luma table values
            luma_sum = sum(img.quantization[0])
            # Lower sum = higher quality
            if luma_sum < 500:
                return 97
            elif luma_sum < 1000:
                return 92
            elif luma_sum < 2000:
                return 85
            else:
                return 80
    except Exception:
        pass
    return 95


# ──────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ──────────────────────────────────────────────

def load_passing_images(audit_report_path: Path) -> list[str]:
    """
    Read audit_report.json and return filenames of images that passed.
    Both hard-passed and warning-passed images are included — the user
    already reviewed warnings manually.
    """
    with open(audit_report_path) as f:
        report = json.load(f)

    passing = []
    for name, entry in report["images"].items():
        if entry["passed"]:
            passing.append(entry["filepath"])

    return sorted(passing)  # Sort for deterministic sequential numbering


def process_images(
    passing_paths: list[str],
    output_dir: Path,
    target_size: int,
    start_index: int = 1,
) -> dict:
    """
    Resize, crop, rename, and save all passing images.
    Returns the metadata dict mapping new names to originals.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    total = len(passing_paths)
    # Zero-pad width based on total count (e.g. 50 images → 2 digits minimum, use 6 for future-proofing)
    pad_width = max(6, len(str(total + start_index)))

    print(f"\nProcessing {total} images → {output_dir}")
    print(f"Target resolution: {target_size}×{target_size}\n")

    failed_during_resize = []

    for idx, filepath in enumerate(passing_paths, start=start_index):
        src_path = Path(filepath)
        seq_name = f"{str(idx).zfill(pad_width)}.jpg"
        dst_path = output_dir / seq_name

        try:
            # Load with OpenCV (handles most JPEG variants robustly)
            img = cv2.imread(str(src_path))
            if img is None:
                raise ValueError("OpenCV returned None")

            # Convert BGR → RGB for consistent processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize + crop
            cropped_rgb, crop_meta = resize_and_crop(img_rgb, target_size)

            # Convert back to BGR for saving with OpenCV
            cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)

            # Determine save quality
            quality = get_jpeg_quality(src_path)

            # Save as JPEG
            cv2.imwrite(
                str(dst_path),
                cropped_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )

            # Verify the saved file loads correctly
            verify = cv2.imread(str(dst_path))
            if verify is None:
                raise ValueError("Verification read failed after save")

            metadata[seq_name] = {
                "sequential_name": seq_name,
                "original_filename": src_path.name,
                "original_filepath": str(src_path),
                "target_size": target_size,
                "crop_info": crop_meta,
                "save_quality": quality,
                "output_size_kb": round(dst_path.stat().st_size / 1024, 1),
                "status": "ok",
            }

            print(f"  [{idx:>{pad_width}}] {src_path.name:40s} → {seq_name}  "
                  f"({crop_meta['original_width']}×{crop_meta['original_height']} "
                  f"→ {target_size}×{target_size})")

        except Exception as e:
            failed_during_resize.append({"filename": src_path.name, "error": str(e)})
            print(f"  [ERROR] {src_path.name}: {e}")
            metadata[seq_name] = {
                "sequential_name": seq_name,
                "original_filename": src_path.name,
                "original_filepath": str(src_path),
                "status": "failed",
                "error": str(e),
            }

    return metadata, failed_during_resize


# ──────────────────────────────────────────────
# METADATA & REPORT
# ──────────────────────────────────────────────

def write_metadata(metadata: dict, failed: list, output_dir: Path, target_size: int):
    """
    Save metadata.json alongside the processed images.
    This file is essential for later steps — it maps every processed
    image back to its original file and records crop offsets.
    """
    successful = {k: v for k, v in metadata.items() if v["status"] == "ok"}
    failed_entries = {k: v for k, v in metadata.items() if v["status"] == "failed"}

    meta_output = {
        "generated_at": datetime.now().isoformat(),
        "target_size": target_size,
        "total_processed": len(successful),
        "total_failed": len(failed_entries),
        "images": metadata,
    }

    meta_path = output_dir.parent / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_output, f, indent=2)

    # Console summary
    print(f"\n{'='*55}")
    print(f"RESIZE COMPLETE")
    print(f"{'='*55}")
    print(f"  Successfully processed: {len(successful)}")
    if failed_entries:
        print(f"  Failed:                 {len(failed_entries)}")
        for name, entry in failed_entries.items():
            print(f"    → {entry['original_filename']}: {entry['error']}")
    print(f"\n  Output:   {output_dir}")
    print(f"  Metadata: {meta_path}")
    print()

    return meta_path


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Resize and rename passing images for depth model training."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing your raw JPEG images (same folder used for audit)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save processed images. Default: <input_dir>/../processed/images",
    )
    parser.add_argument(
        "--audit_report",
        type=str,
        default=None,
        help="Path to audit_report.json. Default: <input_dir>/audit_report.json",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=518,
        help="Target square resolution in pixels (default: 518 for Depth-Anything-V2)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Starting number for sequential naming (default: 1)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()

    audit_report_path = (
        Path(args.audit_report).resolve()
        if args.audit_report
        else input_dir / "audit_report.json"
    )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir.parent / "processed" / "images"
    )

    # Validate inputs
    if not input_dir.exists():
        print(f"Error: input_dir does not exist: {input_dir}")
        sys.exit(1)

    if not audit_report_path.exists():
        print(f"Error: audit_report.json not found at: {audit_report_path}")
        print("Run audit_images.py first.")
        sys.exit(1)

    # Load passing image paths from audit report
    passing_paths = load_passing_images(audit_report_path)

    if not passing_paths:
        print("No passing images found in audit report. Check audit_report.json.")
        sys.exit(1)

    print(f"\nAudit report:   {audit_report_path}")
    print(f"Passing images: {len(passing_paths)}")
    print(f"Output dir:     {output_dir}")
    print(f"Target size:    {args.target_size}×{args.target_size}")

    # Process
    metadata, failed = process_images(
        passing_paths,
        output_dir,
        args.target_size,
        args.start_index,
    )

    # Save metadata
    write_metadata(metadata, failed, output_dir, args.target_size)


if __name__ == "__main__":
    main()