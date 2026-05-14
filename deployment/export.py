import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import argparse
import torch
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from depth_anything_v2.dpt import DepthAnythingV2

def main():
    parser = argparse.ArgumentParser(description="Export Depth-Anything-V2 checkpoint to TensorRT")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--encoder",     default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-size",  default=518, type=int)
    parser.add_argument("--test-image",  default=None)
    parser.add_argument("--fp16",        action="store_true")
    args = parser.parse_args()

    CKPT_PATH   = args.checkpoint
    ENCODER     = args.encoder
    INPUT_H     = INPUT_W = args.input_size
    ONNX_PATH   = CKPT_PATH.replace(".pt", ".onnx")
    ENGINE_PATH = CKPT_PATH.replace(".pt", ".engine")
    DEVICE      = torch.device("cuda")

    # ── 1. Load checkpoint ───────────────────────────────────────────────────
    model_cfg = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**model_cfg[ENCODER])
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    # ── 2. Export to ONNX (fixed shape, no dynamic axes) ────────────────────
    model.cpu().eval()
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        opset_version=18,
        input_names=["image"],
        output_names=["depth"],
        # no dynamic_axes — fixed shape avoids TRT profile requirement
    )
    print(f"ONNX saved → {ONNX_PATH}")

    # ── 3. Convert ONNX → TensorRT engine ───────────────────────────────────
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser_trt = trt.OnnxParser(network, TRT_LOGGER)
    config     = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    with open(ONNX_PATH, "rb") as f:
        if not parser_trt.parse(f.read()):
            for i in range(parser_trt.num_errors):
                print(parser_trt.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed — check logs above")

    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized)
    print(f"TensorRT engine saved → {ENGINE_PATH}")

    # ── 4. Run inference (optional) ──────────────────────────────────────────
    if args.test_image:
        runtime = trt.Runtime(TRT_LOGGER)
        with open(ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        img       = cv2.imread(args.test_image)
        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res   = cv2.resize(img_rgb, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_res).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        input_np   = img_tensor.contiguous().cpu().numpy()
        output_np  = np.zeros((1, INPUT_H, INPUT_W), dtype=np.float32)

        input_mem  = cuda.mem_alloc(input_np.nbytes)
        output_mem = cuda.mem_alloc(output_np.nbytes)

        cuda.memcpy_htod(input_mem, input_np)
        context.execute_v2([int(input_mem), int(output_mem)])
        cuda.memcpy_dtoh(output_np, output_mem)

        depth_np      = output_np.squeeze()
        
        depth_vis     = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite("depth_result.jpg", depth_colored)
        print("Saved → depth_result.jpg")

if __name__ == "__main__":
    main()