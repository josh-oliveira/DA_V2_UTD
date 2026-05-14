"""
dataloader.py
-------------
Step 5 of the Depth-Anything-V2 fine-tuning pipeline.

Defines the PyTorch Dataset and DataLoader for training.
Handles loading RGB images, float32 depth maps, and validity masks,
applies correct augmentations, and returns properly normalised tensors.

Can be imported into your training script or run directly to verify
the dataset is loading correctly before training begins:

    python dataloader.py --processed_dir /path/to/processed

Directory structure expected:
    processed/
        images/         -> resized JPEGs  (000001.jpg ...)
        depths/         -> float32 maps   (000001.npy ...)
        masks/          -> validity masks  (000001.png ...)
        metadata.json
        splits/
            train.txt
            val.txt
            test.txt

If splits/ does not exist yet, the dataset will load all images
and you can pass a filenames list directly. Run generate_splits.py
first to create the split files.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ──────────────────────────────────────────────
# IMAGENET NORMALISATION
# Required because the DINOv2 encoder inside
# DAv2 was pretrained on ImageNet.
# ──────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

imagenet_normalise = transforms.Normalize(
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
)


# ──────────────────────────────────────────────
# AUGMENTATION HELPERS
# All geometric augmentations must be applied
# identically to both the RGB image and its
# depth map. Colour augmentations apply to
# RGB only — never to depth.
# ──────────────────────────────────────────────

class DepthAugmenter:
    """
    Paired augmentation pipeline for RGB + depth training.

    Geometric ops  → applied to both image and depth (keeps pixel alignment).
    Colour ops     → applied to image only (depth has no colour).
    Never used     → vertical flip, large warps, random erasing.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

        # Colour-only augmentation — image only, never depth
        self.colour_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        )

    def __call__(
        self,
        image: np.ndarray,   # HxWx3 uint8 RGB
        depth: np.ndarray,   # HxW float32 [0, 1]
        mask:  np.ndarray,   # HxW uint8 {0, 255}
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not self.enabled:
            return image, depth, mask

        # ── Horizontal flip ──
        # Safe: depth relationships are preserved under mirroring
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            mask  = np.fliplr(mask).copy()

        # ── Small rotation (±5°) ──
        # Keep small — large rotations distort horizon priors
        if random.random() < 0.3:
            angle  = random.uniform(-5.0, 5.0)
            h, w   = image.shape[:2]
            centre = (w / 2, h / 2)
            M      = cv2.getRotationMatrix2D(centre, angle, 1.0)

            image = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            depth = cv2.warpAffine(
                depth, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            mask = cv2.warpAffine(
                mask, M, (w, h),
                flags=cv2.INTER_NEAREST,   # no interpolation for binary mask
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # ── Colour jitter — image only ──
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        pil_img = self.colour_jitter(pil_img)
        image   = np.array(pil_img)

        # ── Random grayscale — image only ──
        if random.random() < 0.1:
            gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.stack([gray, gray, gray], axis=-1)

        return image, depth, mask


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────

class DepthDataset(Dataset):
    """
    PyTorch Dataset for Depth-Anything-V2 fine-tuning.

    Returns a dict per sample:
        {
            "image":    FloatTensor [3, H, W]  — ImageNet-normalised RGB
            "depth":    FloatTensor [1, H, W]  — depth in [0, 1]
            "mask":     FloatTensor [1, H, W]  — 1.0 = valid, 0.0 = invalid
            "name":     str                    — filename stem e.g. "000001"
        }

    Args:
        processed_dir   Path to the processed/ root folder.
        split           "train", "val", or "test". Reads from splits/<split>.txt.
                        Pass None to load all images (useful before splits exist).
        filenames       Optional explicit list of stems/filenames to load,
                        overrides split loading.
        augment         Whether to apply augmentation (True for train, False for val/test).
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split:         str | None = "train",
        filenames:     list[str] | None = None,
        augment:       bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.images_dir    = self.processed_dir / "images"
        self.depths_dir    = self.processed_dir / "depths"
        self.masks_dir     = self.processed_dir / "masks"
        self.augmenter     = DepthAugmenter(enabled=augment)

        # Validate directories exist
        for d in [self.images_dir, self.depths_dir, self.masks_dir]:
            if not d.exists():
                raise FileNotFoundError(
                    f"Expected directory not found: {d}\n"
                    f"Make sure you have run resize_images.py and generate_depths.py first."
                )

        # ── Resolve file list ──
        if filenames is not None:
            # Explicit list provided — normalise to stems
            self.stems = [Path(f).stem for f in filenames]

        elif split is not None:
            split_file = self.processed_dir / "splits" / f"{split}.txt"
            if not split_file.exists():
                raise FileNotFoundError(
                    f"Split file not found: {split_file}\n"
                    f"Run generate_splits.py first, or pass filenames= directly."
                )
            with open(split_file) as f:
                self.stems = [Path(line.strip()).stem for line in f if line.strip()]

        else:
            # No split specified — load everything
            self.stems = sorted([
                p.stem for p in self.images_dir.glob("*.jpg")
            ])

        if not self.stems:
            raise ValueError(
                f"No samples found for split='{split}' in {self.processed_dir}"
            )

        # ── Validate triplets — every stem must have image + depth + mask ──
        missing = []
        for stem in self.stems:
            if not (self.images_dir / f"{stem}.jpg").exists():
                missing.append(f"  image missing: {stem}.jpg")
            if not (self.depths_dir / f"{stem}.npy").exists():
                missing.append(f"  depth missing: {stem}.npy")
            if not (self.masks_dir  / f"{stem}.png").exists():
                missing.append(f"  mask missing:  {stem}.png")

        if missing:
            raise FileNotFoundError(
                f"Missing files for {len(missing)} sample(s):\n" + "\n".join(missing[:10])
            )

        print(f"  DepthDataset  split={split or 'all'}  "
              f"samples={len(self.stems)}  augment={augment}")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        # ── Load RGB image ──
        img_path = self.images_dir / f"{stem}.jpg"
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise IOError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # HxWx3 uint8

        # ── Load depth map ──
        depth_path = self.depths_dir / f"{stem}.npy"
        depth = np.load(str(depth_path))                     # HxW float32 [0, 1]

        # ── Load validity mask ──
        mask_path = self.masks_dir / f"{stem}.png"
        mask_raw  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask      = (mask_raw > 127).astype(np.float32)      # HxW float32 {0.0, 1.0}

        # ── Augmentation (geometric + colour) ──
        image, depth, mask = self.augmenter(image, depth, mask)

        # ── Convert to tensors ──

        # Image: uint8 HxWx3 -> float32 [0,1] CxHxW -> ImageNet normalised
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = imagenet_normalise(image_tensor)      # [3, H, W]

        # Depth: HxW float32 -> [1, H, W]
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

        # Mask: HxW float32 -> [1, H, W]
        mask_tensor  = torch.from_numpy(mask).unsqueeze(0)   # [1, H, W]

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "mask":  mask_tensor,
            "name":  stem,
        }


# ──────────────────────────────────────────────
# DATALOADER FACTORY
# ──────────────────────────────────────────────

def build_dataloaders(
    processed_dir: str | Path,
    batch_size:    int = 4,
    num_workers:   int = 4,
    pin_memory:    bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders from the processed/ directory.

    Augmentation is enabled for train only.
    Val and test use no augmentation and a fixed sample order.

    Args:
        processed_dir   Path to processed/ root
        batch_size      Images per batch (4–8 for ViT-L on 16GB VRAM)
        num_workers     CPU workers for data loading (4 is a safe default)
        pin_memory      Faster GPU transfers when True (use with CUDA)

    Returns:
        train_loader, val_loader, test_loader
    """
    processed_dir = Path(processed_dir)

    train_dataset = DepthDataset(processed_dir, split="train", augment=True)
    val_dataset   = DepthDataset(processed_dir, split="val",   augment=False)
    test_dataset  = DepthDataset(processed_dir, split="test",  augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,         # Shuffle every epoch during training
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,         # Drop incomplete final batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,        # Fixed order for reproducible validation metrics
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = 1,            # One at a time for final evaluation
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# VERIFICATION — run this script directly to
# confirm the dataset loads correctly before
# starting training
# ──────────────────────────────────────────────

def verify_dataset(processed_dir: Path):
    """
    Load the first batch from the training set and print a full
    shape/dtype/range report. Catches mismatches before training begins.
    """
    print("\n" + "=" * 55)
    print("DATASET VERIFICATION")
    print("=" * 55)

    try:
        dataset = DepthDataset(processed_dir, split=None, augment=False)
    except FileNotFoundError:
        print(
            "\n  Split files not found — loading all images without a split.\n"
            "  Run generate_splits.py to create train/val/test splits.\n"
        )
        dataset = DepthDataset(processed_dir, split=None, augment=False)

    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False)
    batch  = next(iter(loader))

    image = batch["image"]
    depth = batch["depth"]
    mask  = batch["mask"]
    names = batch["name"]

    print(f"\n  Batch size loaded:  {image.shape[0]}")
    print(f"\n  image tensor:")
    print(f"    shape:  {tuple(image.shape)}   (B, C, H, W)")
    print(f"    dtype:  {image.dtype}")
    print(f"    min:    {image.min().item():.4f}   (ImageNet-normalised, negative values are normal)")
    print(f"    max:    {image.max().item():.4f}")
    print(f"    mean:   {image.mean().item():.4f}")

    print(f"\n  depth tensor:")
    print(f"    shape:  {tuple(depth.shape)}   (B, 1, H, W)")
    print(f"    dtype:  {depth.dtype}")
    print(f"    min:    {depth.min().item():.4f}   (should be >= 0.0)")
    print(f"    max:    {depth.max().item():.4f}   (should be <= 1.0)")
    print(f"    mean:   {depth.mean().item():.4f}")

    print(f"\n  mask tensor:")
    print(f"    shape:  {tuple(mask.shape)}   (B, 1, H, W)")
    print(f"    dtype:  {mask.dtype}")
    print(f"    unique values: {torch.unique(mask).tolist()}   (should be [0.0, 1.0])")
    mean_valid = mask.mean().item() * 100
    print(f"    mean valid pixels: {mean_valid:.1f}%")

    print(f"\n  sample names: {list(names)}")

    # Sanity checks
    print(f"\n  Sanity checks:")
    checks = [
        ("image dtype is float32",       image.dtype == torch.float32),
        ("depth dtype is float32",        depth.dtype == torch.float32),
        ("mask dtype is float32",         mask.dtype  == torch.float32),
        ("depth min >= 0.0",              depth.min().item() >= 0.0),
        ("depth max <= 1.0",              depth.max().item() <= 1.0),
        ("mask contains only 0s and 1s",  set(torch.unique(mask).tolist()).issubset({0.0, 1.0})),
        ("image H == depth H",            image.shape[2] == depth.shape[2]),
        ("image W == depth W",            image.shape[3] == depth.shape[3]),
        ("mask H == depth H",             mask.shape[2]  == depth.shape[2]),
        ("at least 50% valid pixels",     mean_valid >= 50.0),
    ]
    all_passed = True
    for label, passed in checks:
        status = "  PASS" if passed else "  FAIL"
        print(f"    {status}  {label}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All checks passed — dataset is ready for training.")
    else:
        print("  Some checks failed — review the output above before training.")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify the depth training dataset.")
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to your processed/ directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size to use when verifying (default: 4)",
    )
    args = parser.parse_args()

    verify_dataset(Path(args.processed_dir))