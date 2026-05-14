"""
generate_splits.py
------------------
Step 5 of the Depth-Anything-V2 fine-tuning pipeline.

Reads the processed/images/ directory and depth_meta.json, then
produces train/val/test split text files under processed/splits/.

Split strategy:
  - Degenerate images (flat/uniform depth) are excluded automatically
  - Remaining images are shuffled with a fixed seed for reproducibility
  - Split at 80 / 10 / 10

Usage:
    python generate_splits.py --processed_dir /path/to/processed

    # Custom split ratios:
    python generate_splits.py --processed_dir /path/to/processed \\
        --train_ratio 0.8 --val_ratio 0.1 --seed 42
"""

import json
import random
import argparse
import sys
from pathlib import Path


def generate_splits(
    processed_dir: Path,
    train_ratio:   float = 0.8,
    val_ratio:     float = 0.1,
    seed:          int   = 42,
):
    images_dir     = processed_dir / "images"
    depth_meta_path = processed_dir / "depth_meta.json"
    splits_dir     = processed_dir / "splits"

    if not images_dir.exists():
        print(f"Error: images dir not found: {images_dir}")
        sys.exit(1)

    # ── Collect all successfully processed stems ──
    all_stems = sorted([p.stem for p in images_dir.glob("*.jpg")])
    if not all_stems:
        print(f"Error: no images found in {images_dir}")
        sys.exit(1)

    print(f"\n  Total images found:    {len(all_stems)}")

    # ── Exclude degenerate samples flagged by depth generation ──
    excluded = set()
    if depth_meta_path.exists():
        with open(depth_meta_path) as f:
            depth_meta = json.load(f)
        degenerate = depth_meta.get("dataset_stats", {}).get("degenerate_images", [])
        for name in degenerate:
            excluded.add(Path(name).stem)
        if excluded:
            print(f"  Excluded (degenerate): {len(excluded)}  {sorted(excluded)}")
    else:
        print("  Note: depth_meta.json not found — no degenerate exclusions applied.")

    usable = [s for s in all_stems if s not in excluded]
    print(f"  Usable after exclusions: {len(usable)}")

    if len(usable) < 3:
        print("Error: need at least 3 usable images to create a split.")
        sys.exit(1)

    # ── Shuffle with fixed seed ──
    rng = random.Random(seed)
    rng.shuffle(usable)

    # ── Compute split sizes ──
    n           = len(usable)
    n_train     = max(1, int(round(n * train_ratio)))
    n_val       = max(1, int(round(n * val_ratio)))
    n_test      = max(1, n - n_train - n_val)

    # Guard: make sure we don't exceed total
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val

    train_stems = usable[:n_train]
    val_stems   = usable[n_train:n_train + n_val]
    test_stems  = usable[n_train + n_val:n_train + n_val + n_test]

    # ── Write split files ──
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        split_path = splits_dir / f"{split_name}.txt"
        with open(split_path, "w") as f:
            for stem in stems:
                f.write(f"{stem}.jpg\n")
        print(f"  {split_name:5s}  {len(stems):3d} images  -> {split_path}")

    # ── Summary ──
    print(f"\n  Split ratios:  train={len(train_stems)/n*100:.1f}%  "
          f"val={len(val_stems)/n*100:.1f}%  "
          f"test={len(test_stems)/n*100:.1f}%")
    print(f"  Seed:          {seed}")
    print(f"  Splits saved:  {splits_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test splits.")
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--train_ratio",   type=float, default=0.8)
    parser.add_argument("--val_ratio",     type=float, default=0.1)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    generate_splits(
        processed_dir = Path(args.processed_dir).resolve(),
        train_ratio   = args.train_ratio,
        val_ratio     = args.val_ratio,
        seed          = args.seed,
    )


if __name__ == "__main__":
    main()