"""
audit_images.py
---------------
Step 1 of the Depth-Anything-V2 fine-tuning pipeline.

Scans a directory of JPEG images and produces a full audit report
BEFORE any files are modified or moved. Run this first to understand
the quality of your dataset.

Usage:
    python audit_images.py --input_dir /path/to/your/photos

Requirements:
    pip install opencv-python pillow numpy imagehash

Outputs:
    audit_report.json     — machine-readable full report
    audit_summary.txt     — human-readable summary with pass/fail per image
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import imagehash


# ──────────────────────────────────────────────
# CONFIGURATION — adjust these thresholds to
# suit your images if needed
# ──────────────────────────────────────────────
CONFIG = {
    # Blur: Laplacian variance below this = too blurry
    "blur_threshold": 100.0,

    # Exposure: mean pixel brightness (0–255)
    "min_brightness": 20,
    "max_brightness": 235,

    # Minimum resolution — images smaller than this
    # in either dimension will be flagged
    "min_dimension": 392,

    # Perceptual hash distance — images with pHash
    # difference <= this are considered near-duplicates
    "duplicate_hash_distance": 8,

    # JPEG quality proxy — files smaller than this
    # (in KB) relative to their resolution suggest
    # heavy compression artifacts
    "min_kb_per_megapixel": 10,
}


# ──────────────────────────────────────────────
# CHECKS
# ──────────────────────────────────────────────

def check_loadable(path: Path) -> tuple[bool, str]:
    """Try to open the file with both PIL and OpenCV."""
    try:
        img_pil = Image.open(path)
        img_pil.verify()  # Checks for corruption without fully decoding
    except Exception as e:
        return False, f"PIL failed to open: {e}"

    img_cv = cv2.imread(str(path))
    if img_cv is None:
        return False, "OpenCV returned None — file may be corrupt or unsupported"

    return True, "ok"


def check_resolution(path: Path) -> tuple[bool, str, dict]:
    """Check image dimensions against minimum requirements."""
    img = Image.open(path)
    w, h = img.size
    min_dim = CONFIG["min_dimension"]
    meta = {"width": w, "height": h, "megapixels": round((w * h) / 1_000_000, 2)}

    if w < min_dim or h < min_dim:
        return False, f"Too small: {w}×{h} (minimum {min_dim}px on shortest side)", meta

    return True, f"{w}×{h}", meta


def check_blur(path: Path) -> tuple[bool, str, float]:
    """
    Laplacian variance method for blur detection.
    Low variance = soft edges = blurry image.
    """
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    threshold = CONFIG["blur_threshold"]

    if variance < threshold:
        return False, f"Blurry (Laplacian variance: {variance:.1f} < {threshold})", variance

    return True, f"Sharp (Laplacian variance: {variance:.1f})", variance


def check_exposure(path: Path) -> tuple[bool, str, dict]:
    """
    Check mean brightness and highlight/shadow clipping.
    Returns per-channel stats for debugging.
    """
    img = cv2.imread(str(path))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean_brightness = float(gray.mean())
    min_b = CONFIG["min_brightness"]
    max_b = CONFIG["max_brightness"]

    # Percentage of pixels that are clipped black or white
    pct_black = float((gray < 5).mean() * 100)
    pct_white = float((gray > 250).mean() * 100)

    stats = {
        "mean_brightness": round(mean_brightness, 1),
        "pct_clipped_black": round(pct_black, 2),
        "pct_clipped_white": round(pct_white, 2),
    }

    if mean_brightness < min_b:
        return False, f"Underexposed (mean brightness: {mean_brightness:.1f} < {min_b})", stats
    if mean_brightness > max_b:
        return False, f"Overexposed (mean brightness: {mean_brightness:.1f} > {max_b})", stats
    if pct_black > 40:
        return False, f"Mostly black — {pct_black:.1f}% pixels clipped dark", stats
    if pct_white > 40:
        return False, f"Mostly white — {pct_white:.1f}% pixels clipped bright", stats

    return True, f"Exposure OK (mean: {mean_brightness:.1f})", stats


def check_compression(path: Path, resolution_meta: dict) -> tuple[bool, str]:
    """
    Rough proxy for JPEG compression quality.
    Very small files relative to their resolution suggest heavy artifacts.
    """
    file_kb = path.stat().st_size / 1024
    megapixels = resolution_meta.get("megapixels", 1)
    kb_per_mp = file_kb / max(megapixels, 0.001)
    threshold = CONFIG["min_kb_per_megapixel"]

    if kb_per_mp < threshold:
        return False, f"Possibly heavy JPEG compression ({kb_per_mp:.1f} KB/MP < {threshold} KB/MP)"

    return True, f"Compression OK ({kb_per_mp:.1f} KB/MP)"


def compute_phash(path: Path) -> str:
    """Compute perceptual hash for duplicate detection."""
    img = Image.open(path)
    return str(imagehash.phash(img))


def find_duplicates(hash_map: dict) -> dict:
    """
    Compare all pHash values pairwise.
    Returns a dict of {filename: [list of near-duplicate filenames]}.
    """
    names = list(hash_map.keys())
    hashes = {k: imagehash.hex_to_hash(v) for k, v in hash_map.items()}
    duplicates = {}
    threshold = CONFIG["duplicate_hash_distance"]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            distance = hashes[a] - hashes[b]
            if distance <= threshold:
                duplicates.setdefault(a, []).append(b)
                duplicates.setdefault(b, []).append(a)

    return duplicates


# ──────────────────────────────────────────────
# MAIN AUDIT LOOP
# ──────────────────────────────────────────────

def audit_directory(input_dir: Path) -> dict:
    extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix in extensions
    ])

    if not image_paths:
        print(f"No JPEG images found in {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(image_paths)} JPEG images in {input_dir}")
    print("Running checks...\n")

    results = {}
    hash_map = {}

    for path in image_paths:
        name = path.name
        print(f"  Checking: {name}")
        entry = {
            "filename": name,
            "filepath": str(path),
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "checks": {},
            "issues": [],
            "passed": True,
        }

        # 1. Can it be loaded?
        ok, msg = check_loadable(path)
        entry["checks"]["loadable"] = {"passed": ok, "message": msg}
        if not ok:
            entry["issues"].append(msg)
            entry["passed"] = False
            results[name] = entry
            continue  # Skip remaining checks if file is corrupt

        # 2. Resolution
        ok, msg, res_meta = check_resolution(path)
        entry["checks"]["resolution"] = {"passed": ok, "message": msg, **res_meta}
        if not ok:
            entry["issues"].append(msg)
            entry["passed"] = False

        # 3. Blur
        ok, msg, blur_score = check_blur(path)
        entry["checks"]["blur"] = {"passed": ok, "message": msg, "laplacian_variance": round(blur_score, 2)}
        if not ok:
            entry["issues"].append(msg)
            entry["passed"] = False

        # 4. Exposure
        ok, msg, exp_stats = check_exposure(path)
        entry["checks"]["exposure"] = {"passed": ok, "message": msg, **exp_stats}
        if not ok:
            entry["issues"].append(msg)
            entry["passed"] = False

        # 5. Compression
        res_meta_for_compression = entry["checks"]["resolution"]
        ok, msg = check_compression(path, res_meta_for_compression)
        entry["checks"]["compression"] = {"passed": ok, "message": msg}
        if not ok:
            entry["issues"].append(msg)
            # Compression is a warning, not a hard fail
            entry["warnings"] = entry.get("warnings", []) + [msg]

        # 6. Compute pHash for duplicate detection (done after main loop)
        try:
            hash_map[name] = compute_phash(path)
        except Exception as e:
            hash_map[name] = None
            entry["issues"].append(f"pHash failed: {e}")

        results[name] = entry

    # ── Duplicate detection (needs all hashes first) ──
    valid_hashes = {k: v for k, v in hash_map.items() if v is not None}
    duplicate_map = find_duplicates(valid_hashes)

    for name, dupes in duplicate_map.items():
        if name in results:
            results[name]["checks"]["duplicates"] = {
                "passed": False,
                "message": f"Near-duplicate of: {', '.join(dupes)}",
                "near_duplicates": dupes,
            }
            results[name]["issues"].append(f"Near-duplicate of: {', '.join(dupes)}")
            # Mark as warning, not hard fail — let user decide which to keep
            results[name]["warnings"] = results[name].get("warnings", []) + [
                f"Near-duplicate detected — review manually"
            ]

    return results


# ──────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────

def write_reports(results: dict, output_dir: Path):
    passed = [r for r in results.values() if r["passed"] and not r.get("warnings")]
    warnings = [r for r in results.values() if r["passed"] and r.get("warnings")]
    failed = [r for r in results.values() if not r["passed"]]

    # ── JSON report ──
    report = {
        "generated_at": datetime.now().isoformat(),
        "config": CONFIG,
        "summary": {
            "total": len(results),
            "passed_clean": len(passed),
            "passed_with_warnings": len(warnings),
            "failed": len(failed),
        },
        "images": results,
    }

    json_path = output_dir / "audit_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Human-readable summary ──
    txt_path = output_dir / "audit_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("IMAGE AUDIT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"TOTAL IMAGES:          {len(results)}\n")
        f.write(f"  ✅ Passed (clean):   {len(passed)}\n")
        f.write(f"  ⚠️  Passed (warning): {len(warnings)}\n")
        f.write(f"  ❌ Failed:           {len(failed)}\n\n")

        if failed:
            f.write("-" * 40 + "\n")
            f.write("FAILED IMAGES (recommend excluding):\n")
            f.write("-" * 40 + "\n")
            for r in failed:
                f.write(f"\n  {r['filename']}\n")
                for issue in r["issues"]:
                    f.write(f"    → {issue}\n")

        if warnings:
            f.write("\n" + "-" * 40 + "\n")
            f.write("WARNINGS (review manually):\n")
            f.write("-" * 40 + "\n")
            for r in warnings:
                f.write(f"\n  {r['filename']}\n")
                for w in r.get("warnings", []):
                    f.write(f"    ⚠  {w}\n")

        if passed:
            f.write("\n" + "-" * 40 + "\n")
            f.write("PASSED IMAGES:\n")
            f.write("-" * 40 + "\n")
            for r in passed:
                res = r["checks"].get("resolution", {})
                blur = r["checks"].get("blur", {})
                f.write(
                    f"  ✅ {r['filename']:40s}  "
                    f"{res.get('width', '?')}×{res.get('height', '?')}  "
                    f"blur={blur.get('laplacian_variance', '?')}\n"
                )

    return json_path, txt_path


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit JPEG images for depth model training.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the folder containing your raw JPEG images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the reports (default: same as input_dir)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"Error: input_dir does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = audit_directory(input_dir)
    json_path, txt_path = write_reports(results, output_dir)

    # Print summary to console
    passed = sum(1 for r in results.values() if r["passed"] and not r.get("warnings"))
    warnings = sum(1 for r in results.values() if r["passed"] and r.get("warnings"))
    failed = sum(1 for r in results.values() if not r["passed"])

    print(f"\n{'='*50}")
    print(f"AUDIT COMPLETE")
    print(f"{'='*50}")
    print(f"  Total:             {len(results)}")
    print(f"  ✅ Passed (clean): {passed}")
    print(f"  ⚠️  Warnings:       {warnings}")
    print(f"  ❌ Failed:         {failed}")
    print(f"\nReports saved to:")
    print(f"  {json_path}")
    print(f"  {txt_path}")
    print()


if __name__ == "__main__":
    main()