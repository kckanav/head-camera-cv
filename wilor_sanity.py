"""Stage 8 / Phase 1: WiLoR sanity check on a real Pi-Cam frame.

What this validates end-to-end:
  - The .venv-hamer environment can load WiLoR (ckpt + MANO model).
  - YOLO detection runs and finds a hand on a real frame from our rig.
  - WiLoR's MANO regressor produces a sensible mesh (extent ~hand-sized).
  - 21 reprojected 2D keypoints overlay onto the actual hand in the frame.
  - Inference runs on MPS (Apple Metal) on this M2 host.

This script lives at the project root so it survives a re-clone of wilor/.
The wilor/ dir is gitignored (large weights, separate license), so anything
inside it can be lost; this entry point is what we keep.

Run:
    .venv-hamer/bin/python wilor_sanity.py

Notes on the workarounds applied here (kept intentionally inline — they are
specific to: WiLoR @ rolpotamias/main, ultralytics 8.1.34, torch 2.11):
  * pyrender doesn't load on macOS without OSMesa, so we stub the three
    renderer-using submodules. Inference works; only visualisation is gated.
  * PyTorch 2.6 flipped torch.load's default to weights_only=True. The YOLO
    detector ckpt has pickled class refs and can't load under that default;
    we monkey-patch torch.load back to weights_only=False (we trust both
    checkpoint sources from the upstream HuggingFace mirror).
  * The YOLO ckpt was pickled with `dill`, which is not in WiLoR's
    requirements.txt — we install it separately.

If any of these stop being needed (upstream pinning a newer ultralytics,
PyTorch reverting the default, etc.), the workaround can be deleted.
"""

import os
import sys
import time
import types
from pathlib import Path

# --- Workaround 1: pyrender on macOS ------------------------------------------
# Stub the three renderer-using submodules of wilor.utils so model imports work.
# Anything that actually tries to render will explode loudly - fine, this is
# inference-only.


class _NoopRenderer:
    def __init__(self, *a, **kw):
        pass

    def __getattr__(self, name):
        raise RuntimeError(
            f"sanity stubbed out the renderer; nothing should be calling "
            f"{name!r} during inference"
        )


def _cam_crop_to_full(*a, **kw):
    raise RuntimeError("cam_crop_to_full stubbed out in sanity")


for _mod_name, _attrs in [
    ("wilor.utils.renderer",          {"Renderer": _NoopRenderer, "cam_crop_to_full": _cam_crop_to_full}),
    ("wilor.utils.mesh_renderer",     {"MeshRenderer": _NoopRenderer}),
    ("wilor.utils.skeleton_renderer", {"SkeletonRenderer": _NoopRenderer}),
]:
    _stub = types.ModuleType(_mod_name)
    for _k, _v in _attrs.items():
        setattr(_stub, _k, _v)
    sys.modules[_mod_name] = _stub

# --- Workaround 2: PyTorch 2.6 torch.load default -----------------------------
import torch

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

import cv2
import numpy as np
from ultralytics import YOLO

# --- Path setup
# WiLoR's repo and python package have an awkward layout on disk:
#   cameramount/wilor/        (repo root, no __init__.py)
#     wilor/                  (the actual python package, also no __init__.py)
#       models/, utils/, ...
#     models/                 (where the user uploaded MANO_RIGHT.pkl etc.)
# If PROJECT_ROOT (cameramount/) is on sys.path, Python merges
# cameramount/wilor/ and cameramount/wilor/wilor/ into a single namespace
# package, and `wilor.models` then ambiguously resolves between the real
# package and cameramount/wilor/models (the MANO uploads). Fix: grab dated.py
# first while PROJECT_ROOT is still on sys.path, then drop PROJECT_ROOT and
# put WILOR_DIR alone in front.
PROJECT_ROOT = Path(__file__).resolve().parent
WILOR_DIR = PROJECT_ROOT / "wilor"
INPUT_IMG = PROJECT_ROOT / "inputs/24th April 2026 - photo cam0.jpg"

if not WILOR_DIR.is_dir():
    sys.exit(f"missing {WILOR_DIR}: did Phase 0 setup run? see PLAN.md")
if not INPUT_IMG.is_file():
    sys.exit(f"missing {INPUT_IMG}: input photo not found")

sys.path.insert(0, str(PROJECT_ROOT))
from dated import today_pretty   # noqa: E402

OUT_OVERLAY = PROJECT_ROOT / f"outputs/{today_pretty()} - wilor sanity overlay.jpg"
OUT_OBJ = PROJECT_ROOT / f"outputs/{today_pretty()} - wilor sanity hand0.obj"

# Drop PROJECT_ROOT (and any equivalent script-dir entry) before importing wilor.
sys.path = [p for p in sys.path if Path(p).resolve() != PROJECT_ROOT.resolve()]
sys.path.insert(0, str(WILOR_DIR))
os.chdir(WILOR_DIR)

from wilor.models import load_wilor                                   # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset                # noqa: E402


def to_device(obj, device):
    """recursive_to + float64 -> float32 cast (MPS doesn't support float64)."""
    if torch.is_tensor(obj):
        if obj.dtype == torch.float64:
            obj = obj.to(torch.float32)
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def save_obj(path: Path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    print("loading WiLoR ...")
    t0 = time.time()
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    model = model.to(device).eval()
    print(f"loaded in {time.time()-t0:.1f}s")

    print("loading YOLO detector ...")
    detector = YOLO("./pretrained_models/detector.pt")

    img = cv2.imread(str(INPUT_IMG))
    h, w = img.shape[:2]
    print(f"image: {INPUT_IMG.name}  ({w}x{h})")

    print("running YOLO ...")
    t0 = time.time()
    # ultralytics has a known MPS bug for Pose models, so YOLO runs on CPU
    # while WiLoR (the heavy ViT) runs on MPS. See ultralytics issue #4031.
    detections = detector(img, device="cpu", conf=0.3, verbose=False)[0]
    bboxes, is_right = [], []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())
    print(f"YOLO: {len(bboxes)} hands detected in {time.time()-t0:.2f}s")
    if not bboxes:
        sys.exit("no hands detected; aborting sanity check")

    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    dataset = ViTDetDataset(model_cfg, img, boxes, right, rescale_factor=2.0, fp16=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print("running WiLoR ...")
    t0 = time.time()
    overlay = img.copy()
    saved_first_obj = False
    for batch_idx, batch in enumerate(loader):
        batch = to_device(batch, device)
        with torch.no_grad():
            out = model(batch)
        verts = out["pred_vertices"].detach().cpu().numpy()        # (B, 778, 3)
        joints2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # (B, 21, 2) in [-0.5, 0.5]
        for n in range(batch["img"].shape[0]):
            hand_idx = batch_idx * 16 + n
            handed = "right" if batch["right"][n].item() > 0.5 else "left"

            v = verts[n].copy()
            if handed == "left":
                v[:, 0] *= -1   # MANO is right-handed by convention; mirror left

            if not saved_first_obj:
                save_obj(OUT_OBJ, v, model.mano.faces)
                saved_first_obj = True
                print(f"  hand {hand_idx} ({handed}): mesh -> {OUT_OBJ.name}")

            extent_cm = (v.max(0) - v.min(0)).max() * 100.0
            print(f"  hand {hand_idx} ({handed}): mesh extent ~{extent_cm:.1f} cm "
                  f"(MANO local frame, monocular scale)")

            box_center = batch["box_center"][n].cpu().numpy()
            box_size = batch["box_size"][n].cpu().item()
            kp = joints2d[n].copy()
            kp[:, 0] = box_center[0] + kp[:, 0] * box_size
            kp[:, 1] = box_center[1] + kp[:, 1] * box_size
            color = (0, 200, 255) if handed == "right" else (255, 100, 0)
            for x, y in kp.astype(int):
                cv2.circle(overlay, (int(x), int(y)), 6, color, -1)
                cv2.circle(overlay, (int(x), int(y)), 7, (255, 255, 255), 1)
            x0, y0, x1, y1 = boxes[hand_idx].astype(int)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
            cv2.putText(overlay, handed, (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imwrite(str(OUT_OVERLAY), overlay)
    print(f"\nWiLoR ran in {time.time()-t0:.1f}s")
    print(f"overlay -> {OUT_OVERLAY}")
    print(f"mesh    -> {OUT_OBJ}")

    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(OUT_OVERLAY)], check=False)


if __name__ == "__main__":
    main()
