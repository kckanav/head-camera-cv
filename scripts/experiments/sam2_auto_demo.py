"""Experiment: SAM2 'segment anything' automatic mask generator.

Per-frame fully automatic segmentation of the LEFT RAW video — no
clicks, no propagation, no temporal consistency. Just SAM2 sampling a
grid of points, predicting a mask per point, deduplicating, and
returning every object it could find. Demonstrates SAM2's pure
"everything" mode the way the SAM webdemo does.

Output: a single-view mp4 with each detected mask alpha-blended in a
random colour and a thin bbox drawn around it. A small HUD prints
frame index + running fps + mask count per frame.

Performance: ~1–1.5 s/frame on L4 with sam2.1_hiera_large at the
default 32x32 grid. 301-frame clip ≈ 5–8 min wall time. Halve grid
density to 16 for ~4x speed at the cost of missing smaller objects.

Why "experiments/" not "pipeline/": this is a one-off visualisation,
not feeding triangulation. The canonical pipeline uses click-prompted
SAM2 video propagation in pipeline/12_segment_objects.py.

Run on the L4:
    /root/.venv-sam2/bin/python scripts/experiments/sam2_auto_demo.py
    /root/.venv-sam2/bin/python scripts/experiments/sam2_auto_demo.py --max-frames 30
    /root/.venv-sam2/bin/python scripts/experiments/sam2_auto_demo.py --points-per-side 16
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty
from device import pick_device, configure_perf
from sam2_setup import SAM2_CKPT, SAM2_CFG, assert_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = (PROJECT_ROOT
    / "inputs/27th April 2026 wide - cam1 interaction first 10s.mp4")
DEFAULT_TAG = "27th April interaction"


def overlay_masks(image_bgr: np.ndarray, masks: list,
                  alpha: float = 0.5, draw_bboxes: bool = True,
                  seed: int = 0) -> np.ndarray:
    """Alpha-blend each mask in a random colour, then draw its bbox in
    the same colour. Larger masks rendered first so smaller ones are
    on top (avoids huge masks burying everything else).
    """
    out = image_bgr.copy()
    rng = np.random.default_rng(seed=seed)
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    for m in masks:
        color = tuple(int(c) for c in rng.integers(64, 255, size=3))
        sel = m["segmentation"]
        if not sel.any():
            continue
        out[sel] = (alpha * out[sel] + (1 - alpha) * np.array(color)).astype(np.uint8)
        if draw_bboxes:
            x, y, w, h = m["bbox"]
            cv2.rectangle(out, (int(x), int(y)),
                          (int(x + w), int(y + h)), color, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", default=str(DEFAULT_VIDEO),
                    help="raw video to segment (default: 27th April left interaction clip)")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="output filename tag")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="process only first N frames (smoke testing)")
    ap.add_argument("--points-per-side", type=int, default=32,
                    help="grid density for SAM2 prompt points "
                         "(default 32 → 1024 points/frame; 16 ≈ 4× faster)")
    ap.add_argument("--pred-iou-thresh", type=float, default=0.7)
    ap.add_argument("--stability-thresh", type=float, default=0.7)
    ap.add_argument("--checkpoint", default=str(SAM2_CKPT))
    ap.add_argument("--cfg", default=SAM2_CFG)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / args.video
    if not video_path.is_file():
        sys.exit(f"missing video: {video_path}")

    device = pick_device()
    configure_perf(device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cap_d = torch.cuda.get_device_capability(0)
        print(f"device: cuda  ({gpu_name}, {gpu_mem_gb:.1f} GiB, sm_{cap_d[0]}{cap_d[1]})  "
              f"autocast=bfloat16")
    else:
        print(f"device: {device}")

    assert_checkpoint(Path(args.checkpoint))

    print("building SAM2 + automatic mask generator...")
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam2_model = build_sam2(args.cfg, str(args.checkpoint),
                            device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_total, args.max_frames) if args.max_frames else n_total
    print(f"video:  {video_path.name}  {w}x{h}  {n} frames @ {fps:.1f} fps")
    print(f"prompt: {args.points_per_side}x{args.points_per_side} "
          f"= {args.points_per_side ** 2} grid points/frame")
    print()

    out_path = (PROJECT_ROOT
                / f"outputs/{today_pretty()} - sam2 auto demo [{args.tag}].mp4")
    # avc1 (h264) for native macOS playback; fall back to mp4v if the
    # OpenCV install lacks the codec.
    for fourcc_tag in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"writer: {fourcc_tag}")
            break
    else:
        sys.exit(f"failed to open VideoWriter at {out_path}")

    autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else torch.autocast("cpu", enabled=False))

    t0 = time.time()
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), autocast_ctx:
            masks = mask_generator.generate(rgb)
        # `i` is the random seed → deterministic colours frame to frame for
        # the same mask index; doesn't track identity (auto mask gen has no
        # temporal model) but keeps the visual stable.
        overlaid = overlay_masks(frame, masks, alpha=0.5,
                                 draw_bboxes=True, seed=i)

        elapsed = time.time() - t0
        fps_running = (i + 1) / max(elapsed, 1e-6)
        eta = (n - i - 1) / max(fps_running, 1e-6)
        hud = (f"frame {i + 1}/{n}  masks: {len(masks):3d}  "
               f"{fps_running:.2f} fps  ETA {eta:.0f}s")
        cv2.rectangle(overlaid, (4, h - 28), (16 + 9 * len(hud), h - 6),
                      (0, 0, 0), -1)
        cv2.putText(overlaid, hud, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(overlaid)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  frame {i + 1}/{n}  {fps_running:.2f} fps  "
                  f"masks={len(masks)}  ETA {eta:.0f}s")

    cap.release()
    writer.release()

    total = time.time() - t0
    print(f"\n=== done in {total:.1f}s ({n / total:.2f} fps) → {out_path.name} ===")

    # Re-encode to plain h264 if ffmpeg is available — guarantees clean
    # macOS playback regardless of what fourcc the writer picked.
    if shutil.which("ffmpeg") is not None:
        h264 = out_path.with_suffix(".h264.mp4")
        print(f"re-encoding to h264 → {h264.name}")
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(out_path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "22",
             "-pix_fmt", "yuv420p", str(h264)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            out_path.unlink()
            h264.rename(out_path)
            print(f"  done")
        else:
            print(f"  ffmpeg failed: {result.stderr.strip()[:200]}")


if __name__ == "__main__":
    main()
