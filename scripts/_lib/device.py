"""Device selection + perf config for CUDA / MPS / CPU portability.

Used by every wilor pipeline script. The goal is that the canonical
pipeline (scripts/pipeline/08_wilor_canonical.py) runs correctly on:

  - macOS / Apple Silicon: MPS for the WiLoR ViT, CPU for YOLO (ultralytics
    MPS Pose-model bug #4031), no fp64 (MPS doesn't support it), no
    autocast (MPS autocast is unstable for our ViT).

  - Linux + NVIDIA: CUDA for both ViT and YOLO, fp16 autocast for ViT
    inference, TF32 for fp32 matmul (Ampere+), cudnn benchmark on, multi-
    worker DataLoader with pinned memory so H2D copies overlap compute.

  - CPU only: everything on CPU; fp32 throughout; no autocast.

Same script, different device — no per-platform code branches at the call
site, just `device = pick_device(); cfg = configure_perf(device)`.

Usage:
    from device import pick_device, configure_perf, to_device_safe, \
                       autocast_ctx, cuda_sync
    device = pick_device()
    cfg = configure_perf(device)

    detections = detector(img, device=cfg["yolo_device"], ...)
    dataset = ViTDetDataset(..., fp16=cfg["fp16"])
    loader = DataLoader(dataset, num_workers=cfg["num_workers"],
                        pin_memory=cfg["pin_memory"])
    for batch in loader:
        batch = to_device_safe(batch, device)
        with torch.inference_mode(), autocast_ctx(device, cfg["autocast_dtype"]):
            out = model(batch)
"""

from contextlib import nullcontext

import torch


def pick_device() -> torch.device:
    """CUDA > MPS > CPU. Industry-standard precedence."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_perf(device: torch.device) -> dict:
    """Set global perf flags for the picked device, return a config dict.

    The dict carries device-specific knobs that the script then plumbs into
    each library:
      yolo_device     -> ultralytics YOLO  (string: "cuda" / "mps" / "cpu")
      fp16            -> ViTDetDataset     (bool)
      num_workers     -> DataLoader        (int)
      pin_memory      -> DataLoader        (bool)
      autocast_dtype  -> torch.autocast    (torch.dtype | None)
    """
    if device.type == "cuda":
        # Ampere+ TF32 path: ~2x speedup on fp32 matmul with no measurable
        # accuracy loss for inference. cudnn.benchmark trades a one-time
        # autotune cost on the first batch for faster steady-state.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return {
            "yolo_device": "cuda",
            "fp16": True,
            "num_workers": 4,
            "pin_memory": True,
            "autocast_dtype": torch.float16,
        }
    if device.type == "mps":
        return {
            # ultralytics issue #4031: Pose models break on MPS. WiLoR uses
            # YOLOv8-Pose for hand detection, so we keep YOLO on CPU on Mac.
            "yolo_device": "cpu",
            "fp16": False,
            "num_workers": 0,
            "pin_memory": False,
            "autocast_dtype": None,
        }
    return {
        "yolo_device": "cpu",
        "fp16": False,
        "num_workers": 0,
        "pin_memory": False,
        "autocast_dtype": None,
    }


def to_device_safe(obj, device: torch.device):
    """Recursive obj.to(device) with one device-specific fix.

    MPS doesn't support fp64; any fp64 tensor must be cast before transfer.
    On CUDA / CPU we leave dtypes alone — callers may legitimately want
    fp64 (e.g. for camera-frame math), and silently downcasting would lose
    precision.

    `non_blocking=True` is only safe with pinned source memory + CUDA target;
    elsewhere it has no benefit and can mask races on MPS. Use it only on CUDA.
    """
    if torch.is_tensor(obj):
        if device.type == "mps" and obj.dtype == torch.float64:
            obj = obj.to(torch.float32)
        non_blocking = device.type == "cuda"
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: to_device_safe(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device_safe(v, device) for v in obj)
    return obj


def autocast_ctx(device: torch.device, dtype):
    """Return torch.autocast for CUDA + a non-None dtype, else a no-op.

    We deliberately don't autocast on MPS (unstable for our ViT under
    PyTorch 2.11) or CPU (rarely a win for inference of this size).
    """
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def cuda_sync(device: torch.device) -> None:
    """torch.cuda.synchronize() if applicable. Wrap timing blocks with this
    on CUDA — kernel launches are async, so without sync the recorded
    seconds reflect launch latency, not actual GPU work."""
    if device.type == "cuda":
        torch.cuda.synchronize()
