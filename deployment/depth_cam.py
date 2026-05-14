#!/usr/bin/env python3
"""
Live depth inference on NVIDIA Jetson Orin Nano.
Supports both a TensorRT .engine file and a raw PyTorch .pt checkpoint.
Displays the RGB camera feed alongside a colourised depth map.

Usage:
  # TensorRT engine (fast, recommended after export)
  python depth_cam.py --engine models/best_e400_L_1_03_01.engine

  # PyTorch checkpoint (no export step needed, slower)
  python depth_cam.py --engine models/best_e400_L_1_03_01.pt --encoder vits

  # With frame saving
  python depth_cam.py --engine models/best_e400_L_1_03_01.engine --save-dir ./captures

Keys:
  q / ESC  : quit
  s        : save current side-by-side frame (requires --save-dir)
  c        : cycle depth colourmap
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Optional TensorRT imports (only used for .engine files) ──────────────────
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 — initialises CUDA context
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# ── PyTorch + Depth-Anything-V2 imports ──────────────────────────────────────
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_anything_v2.dpt import DepthAnythingV2


# ── Constants ────────────────────────────────────────────────────────────────
INPUT_H = INPUT_W = 518        # must match what the model was trained / exported at
CAM_W   = 640
CAM_H   = 480

COLORMAPS = [
    ("INFERNO", cv2.COLORMAP_INFERNO),
    ("MAGMA",   cv2.COLORMAP_MAGMA),
    ("PLASMA",  cv2.COLORMAP_PLASMA),
    ("VIRIDIS", cv2.COLORMAP_VIRIDIS),
    ("JET",     cv2.COLORMAP_JET),
]

MODEL_CFGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def put_text(img: np.ndarray, text: str, org=(10, 30),
             scale: float = 0.7, thickness: int = 2) -> None:
    """Draw outlined white text onto an image in-place."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame → contiguous float32 (1, 3, 518, 518) numpy array."""
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
    nchw    = resized.transpose(2, 0, 1)[np.newaxis]
    return np.ascontiguousarray(nchw)


def colorize_depth(depth_2d: np.ndarray, colormap_id: int,
                   out_w: int, out_h: int) -> np.ndarray:
    """Normalise a 2-D depth array, apply a colourmap, and resize."""
    normed  = cv2.normalize(depth_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normed, colormap_id)
    return cv2.resize(colored, (out_w, out_h))


# ── TensorRT loader ──────────────────────────────────────────────────────────
def load_engine(engine_path: str, logger):
    """Deserialise a TensorRT .engine file and return the ICudaEngine."""
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialise engine: {engine_path}")
    return engine


# ── PyTorch loader & inference ───────────────────────────────────────────────
def load_pt_model(pt_path: str, encoder: str, device: torch.device) -> torch.nn.Module:
    """Load a fine-tuned Depth-Anything-V2 .pt checkpoint."""
    if encoder not in MODEL_CFGS:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from: {list(MODEL_CFGS)}")
    model = DepthAnythingV2(**MODEL_CFGS[encoder])
    ckpt  = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.to(device).eval()


def pt_infer(model: torch.nn.Module,
             frame_bgr: np.ndarray,
             device: torch.device) -> np.ndarray:
    """Run a single BGR frame through the PyTorch model; return (H, W) depth array."""
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = model(tensor)
    return depth.squeeze().cpu().numpy()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Live depth inference — Depth-Anything-V2 (.engine or .pt)"
    )
    parser.add_argument("--engine",   required=True,
                        help="Path to a TensorRT .engine file or a PyTorch .pt checkpoint")
    parser.add_argument("--encoder",  default="vits", choices=["vits", "vitb", "vitl"],
                        help="Encoder variant — only required when loading a .pt file (default: vits)")
    parser.add_argument("--cam-id",   type=int, default=0,
                        help="USB camera device ID (default: 0)")
    parser.add_argument("--save-dir", default="",
                        help="Directory to save frames when 's' is pressed")
    args = parser.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────
    ext = Path(args.engine).suffix.lower()

    if ext == ".engine":
        if not TRT_AVAILABLE:
            raise RuntimeError(
                "tensorrt / pycuda not installed. "
                "Install them or use a .pt file instead."
            )
        print(f"Loading TensorRT engine: {args.engine}")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        engine     = load_engine(args.engine, TRT_LOGGER)
        context    = engine.create_execution_context()

        # Pre-allocate GPU I/O buffers (reused every frame)
        input_np   = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
        output_np  = np.zeros((1, INPUT_H, INPUT_W),    dtype=np.float32)
        input_mem  = cuda.mem_alloc(input_np.nbytes)
        output_mem = cuda.mem_alloc(output_np.nbytes)
        bindings   = [int(input_mem), int(output_mem)]

        pt_model = None
        device   = None
        print("TensorRT engine ready.")

    elif ext == ".pt":
        print(f"Loading PyTorch checkpoint: {args.engine}  (encoder={args.encoder})")
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pt_model = load_pt_model(args.engine, args.encoder, device)

        # Set unused TRT variables to None
        engine = context = input_mem = output_mem = bindings = None
        output_np = np.zeros((1, INPUT_H, INPUT_W), dtype=np.float32)
        print(f"PyTorch model ready on {device}.")

    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Pass a .engine (TensorRT) or .pt (PyTorch) file."
        )

    # ── Open USB camera ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera id={args.cam_id}. "
            "Try --cam-id 1 or 2 if /dev/video0 is not the USB device."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # ── Runtime state ────────────────────────────────────────────────────────
    cmap_idx = 0
    ema_fps  = None
    prev_t   = time.time()
    mode_str = "TensorRT" if ext == ".engine" else f"PyTorch [{args.encoder}]"

    print("Running — q/ESC: quit  |  s: save frame  |  c: cycle colourmap")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Warning: missed frame, retrying...")
                continue

            # ── Inference ────────────────────────────────────────────────────
            if pt_model is not None:
                depth_2d  = pt_infer(pt_model, frame_bgr, device)
                output_np = depth_2d[np.newaxis]
            else:
                input_np = preprocess(frame_bgr)
                cuda.memcpy_htod(input_mem, input_np)
                context.execute_v2(bindings)
                cuda.memcpy_dtoh(output_np, output_mem)

            # ── EMA FPS ──────────────────────────────────────────────────────
            now     = time.time()
            dt      = now - prev_t
            prev_t  = now
            inst    = 1.0 / dt if dt > 0 else 0.0
            ema_fps = inst if ema_fps is None else (0.9 * ema_fps + 0.1 * inst)

            # ── Build side-by-side display ───────────────────────────────────
            cmap_name, cmap_id = COLORMAPS[cmap_idx]

            rgb_panel   = frame_bgr.copy()
            depth_panel = colorize_depth(output_np.squeeze(), cmap_id, CAM_W, CAM_H)

            # Left — RGB + FPS + mode
            put_text(rgb_panel, f"FPS:  {ema_fps:.1f}",  (10, 30))
            put_text(rgb_panel, f"Mode: {mode_str}",      (10, 58), scale=0.6)

            # Right — depth + colourmap name
            put_text(depth_panel, f"Depth  [{cmap_name}]",  (10, 30))
            put_text(depth_panel, "c: cycle  |  s: save",   (10, 58), scale=0.55)

            display = np.hstack([rgb_panel, depth_panel])
            cv2.imshow("Depth-Anything-V2-UTD-FT  (q to quit)", display)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("c"):
                cmap_idx = (cmap_idx + 1) % len(COLORMAPS)
                print(f"Colourmap → {COLORMAPS[cmap_idx][0]}")

            elif key == ord("s"):
                if save_dir:
                    fname = save_dir / f"frame_{int(time.time() * 1000)}.jpg"
                    cv2.imwrite(str(fname), display)
                    print(f"Saved → {fname}")
                else:
                    print("Pass --save-dir <path> to enable frame saving.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ext == ".engine" and TRT_AVAILABLE:
            input_mem.free()
            output_mem.free()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
