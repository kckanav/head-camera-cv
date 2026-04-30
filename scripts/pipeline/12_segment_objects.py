"""Stage 9 / Phase 2: SAM2 video propagation on stereo rectified frames.

Reads the clicks JSON from 11_click_objects.py — one positive click per
object per view at frame 0 — then runs SAM2's video predictor on each
rectified view independently and writes per-frame, per-object masks for
both views to an .npz, plus a side-by-side preview overlay video.

The two views are segmented independently. We're not stereo-conditioning
SAM2 — it's good enough at single-view propagation that adding stereo
constraints would be more code than it's worth. The stereo step
(triangulating each object's 3D position from the two mask centroids in
the rectified frames) is in `13_triangulate_objects.py`.

Output:
    outputs/<date> - object masks [<tag>].npz
        masks_left   (N, K, H, W) uint8   0 / 255 per pixel
        masks_right  (N, K, H, W) uint8
        labels       (K,) str
        clicks_left  (K, 2) float32       seed click on frame 0
        clicks_right (K, 2) float32
        fps          float
        image_size   (2,) int             (W, H), rectified resolution
        + provenance: source_clicks, source_calibration, generated_on
    outputs/<date> - object masks [<tag>] preview.mp4
        SBS rectified left | right with per-object mask overlays
        (alpha-blended, one colour per object).

Run:
    .venv-hamer/bin/python scripts/pipeline/12_segment_objects.py        # MPS smoke test on Mac
    .venv-sam2/bin/python  scripts/pipeline/12_segment_objects.py        # CUDA full run on the L4
    .venv-sam2/bin/python  scripts/pipeline/12_segment_objects.py --max-frames 60   # quick check

Device portability is via `_lib/device.py`; same script runs on CUDA /
MPS / CPU with no edits.

Convention reminder: cam1 = LEFT, cam0 = RIGHT.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty
from device import pick_device, configure_perf, cuda_sync, autocast_ctx
from sam2_setup import SAM2_CKPT, SAM2_CFG, assert_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Resolved at runtime from the latest object-clicks JSON if not supplied.
DEFAULT_CLICKS = None

# Tab10 in BGR (OpenCV) for mask overlay.
OVERLAY_COLORS_BGR = [
    (255, 165,   0),   # blue-ish
    (  0, 165, 255),   # orange
    (  0, 255,   0),   # green
    (255,   0, 255),   # magenta
    (  0, 255, 255),   # yellow
    (255, 255,   0),   # cyan
    (128,   0, 128),   # purple
    (255,   0, 128),   # pink
    (128, 255,   0),   # lime
    (  0, 128, 255),   # amber
]


def from_root(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def find_default_clicks() -> Path:
    """Pick the most recently modified `outputs/*object clicks*.json`.

    Lets the user just run `12_segment_objects.py` with no args after
    finishing `11_click_objects.py`.
    """
    candidates = sorted((PROJECT_ROOT / "outputs").glob("* - object clicks *.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.exit("no object-clicks JSON in outputs/. "
                 "Run 11_click_objects.py first, or pass --clicks.")
    return candidates[0]


def extract_rectified_frames(video_path: Path, map_x: np.ndarray, map_y: np.ndarray,
                             out_dir: Path, max_frames: int | None = None,
                             ) -> tuple[int, float]:
    """Read `video_path`, rectify each frame, save as 00000.jpg, 00001.jpg,
    ... in `out_dir`. SAM2's video predictor expects exactly this layout.

    Returns (frame_count, source_fps).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rect = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), rect,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    cap.release()
    return n, fps


def propagate_view(predictor, frames_dir: Path, clicks_xy: list[list[float]],
                   device: torch.device, perf: dict, n_frames: int, h: int, w: int,
                   ) -> np.ndarray:
    """Run SAM2 video propagation on a single view's pre-rectified frames.

    `clicks_xy` is one (x, y) per tracked object, anchored on frame 0.
    Returns a (N, K, H, W) bool array. K = len(clicks_xy).
    """
    K = len(clicks_xy)
    masks = np.zeros((n_frames, K, h, w), dtype=bool)

    state = predictor.init_state(video_path=str(frames_dir))
    for obj_id, (x, y) in enumerate(clicks_xy):
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=obj_id,
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),   # 1 = positive click
        )

    cuda_sync(device)
    t0 = time.time()
    with torch.inference_mode(), autocast_ctx(device, perf["autocast_dtype"]):
        for frame_idx, obj_ids, logits in predictor.propagate_in_video(state):
            for k, oid in enumerate(obj_ids):
                masks[frame_idx, oid] = (logits[k] > 0.0).squeeze(0).cpu().numpy()
    cuda_sync(device)
    elapsed = time.time() - t0

    print(f"    {n_frames} frames × {K} objects in {elapsed:.1f}s "
          f"({n_frames / max(elapsed, 1e-6):.1f} fps)")

    # Free per-video GPU buffers before the next view.
    predictor.reset_state(state)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return masks


def write_preview_video(left_dir: Path, right_dir: Path,
                        masks_left: np.ndarray, masks_right: np.ndarray,
                        labels: list[str], fps: float, out_path: Path) -> None:
    """Side-by-side rectified left | right with mask overlays. Streams
    one frame at a time so we don't blow up memory."""
    n = masks_left.shape[0]
    K = masks_left.shape[1]

    left_files = sorted(left_dir.glob("*.jpg"))[:n]
    right_files = sorted(right_dir.glob("*.jpg"))[:n]

    sample = cv2.imread(str(left_files[0]))
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w * 2, h))
    if not writer.isOpened():
        sys.exit(f"failed to open VideoWriter at {out_path}")

    for i in range(n):
        l = cv2.imread(str(left_files[i]))
        r = cv2.imread(str(right_files[i]))
        for k in range(K):
            color = np.array(OVERLAY_COLORS_BGR[k % len(OVERLAY_COLORS_BGR)],
                             dtype=np.uint8)
            sel_l = masks_left[i, k]
            sel_r = masks_right[i, k]
            if sel_l.any():
                l[sel_l] = (0.5 * l[sel_l] + 0.5 * color).astype(np.uint8)
            if sel_r.any():
                r[sel_r] = (0.5 * r[sel_r] + 0.5 * color).astype(np.uint8)

        # Label legend, top-left of left view (drawn once per frame to
        # survive the alpha blend above).
        for k, lab in enumerate(labels):
            color = OVERLAY_COLORS_BGR[k % len(OVERLAY_COLORS_BGR)]
            cv2.rectangle(l, (8, 8 + 22 * k), (28, 24 + 22 * k), color, -1)
            cv2.putText(l, lab, (34, 24 + 22 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

        writer.write(np.hstack([l, r]))

    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--clicks", default=None,
                    help="object-clicks JSON from 11_click_objects.py "
                         "(default: most-recently-modified one in outputs/)")
    ap.add_argument("--cam0", default=None, help="override cam0 (right) video path")
    ap.add_argument("--cam1", default=None, help="override cam1 (left) video path")
    ap.add_argument("--calib", default=None, help="override stereo calibration npz")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="process only the first N frames (smoke testing)")
    ap.add_argument("--tag", default=None,
                    help="output filename tag "
                         "(default: extracted from the clicks JSON name)")
    ap.add_argument("--checkpoint", default=str(SAM2_CKPT),
                    help=f"SAM2 checkpoint .pt (default {SAM2_CKPT})")
    ap.add_argument("--cfg", default=SAM2_CFG,
                    help=f"SAM2 hydra config name (default {SAM2_CFG})")
    args = ap.parse_args()

    clicks_path = from_root(args.clicks) if args.clicks else find_default_clicks()
    if not clicks_path.is_file():
        sys.exit(f"missing clicks JSON: {clicks_path}")
    print(f"clicks: {clicks_path.name}")

    clicks = json.loads(clicks_path.read_text())
    cam0_path = from_root(args.cam0 or clicks["right_video"])
    cam1_path = from_root(args.cam1 or clicks["left_video"])
    calib_path = from_root(args.calib or clicks["calib"])
    for p in (cam0_path, cam1_path, calib_path):
        if not p.is_file():
            sys.exit(f"missing: {p}")

    # Tag derived from clicks JSON name unless overridden:
    #   "30th April 2026 - object clicks [27th April interaction].json"
    #     → "27th April interaction"
    if args.tag:
        tag = args.tag
    else:
        stem = clicks_path.stem
        tag = stem.split("[", 1)[1].rstrip("]") if "[" in stem else "default"

    objects = clicks["objects"]
    K = len(objects)
    if K == 0:
        sys.exit("clicks JSON has zero objects")
    labels = [o["label"] for o in objects]
    left_clicks = [o["left_xy"] for o in objects]
    right_clicks = [o["right_xy"] for o in objects]

    print(f"objects ({K}):")
    for i, o in enumerate(objects):
        print(f"  {i}: {o['label']:<20s}"
              f"  L=({o['left_xy'][0]:6.1f}, {o['left_xy'][1]:6.1f})"
              f"  R=({o['right_xy'][0]:6.1f}, {o['right_xy'][1]:6.1f})")
    print()

    # ---- Device + SAM2 ----
    device = pick_device()
    perf = configure_perf(device)
    print(f"device: {device}  fp16={perf['fp16']}  "
          f"autocast={perf['autocast_dtype']}")

    ckpt = assert_checkpoint(Path(args.checkpoint))

    print("building SAM2 video predictor...")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.cfg, str(ckpt), device=device)

    # ---- Pre-rectify frames into a tempdir layout SAM2 likes ----
    calib = np.load(calib_path, allow_pickle=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="sam2_seg_"))
    left_dir = tmp_root / "left"
    right_dir = tmp_root / "right"
    left_dir.mkdir()
    right_dir.mkdir()

    try:
        print("rectifying + extracting frames...")
        n_left, fps_left = extract_rectified_frames(
            cam1_path, calib["map_left_x"], calib["map_left_y"],
            left_dir, args.max_frames)
        n_right, fps_right = extract_rectified_frames(
            cam0_path, calib["map_right_x"], calib["map_right_y"],
            right_dir, args.max_frames)
        n = min(n_left, n_right)
        if n_left != n_right:
            print(f"  warning: left={n_left} right={n_right}; using min={n}")
        fps = fps_left
        sample = cv2.imread(str(left_dir / "00000.jpg"))
        h, w = sample.shape[:2]
        print(f"  {n} frames per view at {w}x{h}, fps={fps:.2f}")
        print()

        print("LEFT view propagation...")
        masks_left = propagate_view(predictor, left_dir, left_clicks,
                                    device, perf, n, h, w)[:n]
        print("RIGHT view propagation...")
        masks_right = propagate_view(predictor, right_dir, right_clicks,
                                     device, perf, n, h, w)[:n]

        # ---- Save NPZ ----
        out_npz = (PROJECT_ROOT
                   / f"outputs/{today_pretty()} - object masks [{tag}].npz")
        print(f"\nsaving masks → {out_npz.name}...")
        np.savez_compressed(
            out_npz,
            masks_left=masks_left.astype(np.uint8) * 255,
            masks_right=masks_right.astype(np.uint8) * 255,
            labels=np.array(labels),
            clicks_left=np.array(left_clicks, dtype=np.float32),
            clicks_right=np.array(right_clicks, dtype=np.float32),
            fps=float(fps),
            image_size=np.array([w, h], dtype=np.int32),
            source_clicks=clicks_path.name,
            source_calibration=calib_path.name,
            source_left_video=cam1_path.name,
            source_right_video=cam0_path.name,
            generated_on=today_pretty(),
            schema_version="1.0",
        )
        print(f"  size: {out_npz.stat().st_size / (1024 * 1024):.1f} MB")

        # ---- Preview overlay video ----
        out_video = (PROJECT_ROOT
                     / f"outputs/{today_pretty()} - object masks [{tag}] preview.mp4")
        print(f"\nrendering preview → {out_video.name}...")
        write_preview_video(left_dir, right_dir, masks_left, masks_right,
                            labels, fps, out_video)
        print(f"  done")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\n=== done ===")
    print(f"  {out_npz}")
    print(f"  {out_video}")

    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(out_video)])


if __name__ == "__main__":
    main()
