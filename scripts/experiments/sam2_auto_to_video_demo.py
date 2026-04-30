"""Experiment: SAM2 auto-init + video propagation hybrid.

Run SAM2's automatic mask generator on frame 0 to find every object
without clicking. Filter to a sensible subset by area + predicted IOU.
Feed each surviving mask as the *initial state* to SAM2's video
predictor (via `add_new_mask`). Propagate through all remaining frames
with stable identity per object across the whole clip.

This combines the two SAM2 modes shown earlier:
  - sam2_auto_demo.py             : auto generator only, per-frame
                                    independent (color flickers).
  - pipeline/12_segment_objects   : video predictor only, but needs a
                                    manual click on every object.
This script removes the clicks while keeping stable identity.

Output:
  outputs/<date> - sam2 auto-to-video [<tag>].mp4
  Single-view raw cam1 with each persistent object's mask alpha-blended
  in a stable per-id colour, plus a thin bbox per frame and a HUD with
  active mask count.

Speed: ~5 s for the frame-0 auto pass + ~5–8 fps propagation (memory
attention scales linearly in object count, so ~15 objects → ~5 fps).
For the 301-frame test clip, total wall ≈ 60–90 s.

Run on the L4:
    /root/.venv-sam2/bin/python scripts/experiments/sam2_auto_to_video_demo.py
    /root/.venv-sam2/bin/python scripts/experiments/sam2_auto_to_video_demo.py --max-objects 25
    /root/.venv-sam2/bin/python scripts/experiments/sam2_auto_to_video_demo.py --max-frames 60
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty
from device import pick_device, configure_perf, cuda_sync
from sam2_setup import SAM2_CKPT, SAM2_CFG, assert_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = (PROJECT_ROOT
    / "inputs/27th April 2026 wide - cam1 interaction first 10s.mp4")
DEFAULT_TAG = "27th April interaction"


def stable_color(obj_id: int) -> tuple[int, int, int]:
    """Deterministic distinguishable-ish BGR colour from an object id."""
    # Golden-ratio HSV stepping → spread hues evenly without clustering.
    hue = int((obj_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def filter_auto_masks(masks: list, h: int, w: int,
                      max_objects: int,
                      min_frac: float, max_frac: float) -> list:
    """Drop masks that are too small (likely noise) or too large (likely
    background / table surface). Then keep the top `max_objects` by
    predicted_iou (SAM2's confidence)."""
    img_area = h * w
    keep = []
    for m in masks:
        frac = m["area"] / img_area
        if frac < min_frac:
            continue
        if frac > max_frac:
            continue
        keep.append(m)
    keep.sort(key=lambda m: m["predicted_iou"], reverse=True)
    return keep[:max_objects]


def extract_frames(video_path: Path, out_dir: Path,
                   max_frames: int | None = None) -> tuple[int, float, int, int]:
    """Decode the video to a directory of JPGs as required by the
    SAM2 video predictor. Returns (n_frames, fps, w, h)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    cap.release()
    return n, fps, w, h


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="process only first N frames (smoke testing)")
    ap.add_argument("--max-objects", type=int, default=15,
                    help="cap on tracked objects (memory attention scales "
                         "linearly; default 15 ≈ 5 fps on L4)")
    ap.add_argument("--min-area-frac", type=float, default=0.001,
                    help="drop masks smaller than this fraction of the image "
                         "(0.001 ≈ ~920 px on 1280x720; default 0.001)")
    ap.add_argument("--max-area-frac", type=float, default=0.30,
                    help="drop masks larger than this fraction (likely "
                         "background / table); default 0.30")
    ap.add_argument("--points-per-side", type=int, default=32,
                    help="grid density for the frame-0 auto generator")
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

    autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else torch.autocast("cpu", enabled=False))

    print("building SAM2 (image + video) ...")
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam2_image = build_sam2(args.cfg, str(args.checkpoint),
                            device=device, apply_postprocessing=False)
    auto_gen = SAM2AutomaticMaskGenerator(
        model=sam2_image,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
    )
    video_predictor = build_sam2_video_predictor(args.cfg, str(args.checkpoint),
                                                 device=device)

    # Decode all frames the predictor will see.
    tmp_root = Path(tempfile.mkdtemp(prefix="sam2_auto2vid_"))
    frames_dir = tmp_root / "frames"
    frames_dir.mkdir()
    try:
        print(f"decoding frames → {frames_dir}")
        n, fps, w, h = extract_frames(video_path, frames_dir, args.max_frames)
        print(f"  {n} frames at {w}x{h}, {fps:.1f} fps")
        print()

        # ----- Frame 0: auto mask generator -----
        print("frame 0: auto mask generator ...")
        frame0 = cv2.imread(str(frames_dir / "00000.jpg"))
        rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        cuda_sync(device)
        t0 = time.time()
        with torch.inference_mode(), autocast_ctx:
            raw_masks = auto_gen.generate(rgb0)
        cuda_sync(device)
        print(f"  generated {len(raw_masks)} candidate masks in {time.time() - t0:.1f}s")

        kept = filter_auto_masks(raw_masks, h, w,
                                 max_objects=args.max_objects,
                                 min_frac=args.min_area_frac,
                                 max_frac=args.max_area_frac)
        print(f"  kept {len(kept)} after area + iou filter:")
        for k, m in enumerate(kept):
            x, y, mw, mh = m["bbox"]
            print(f"    obj{k:2d}: area={m['area']:6d}  iou={m['predicted_iou']:.3f}  "
                  f"bbox=({int(x):4d},{int(y):4d},{int(mw):4d},{int(mh):4d})")
        print()

        # ----- Video predictor: seed each mask, propagate -----
        print("seeding video predictor with frame-0 masks ...")
        state = video_predictor.init_state(video_path=str(frames_dir))
        with torch.inference_mode(), autocast_ctx:
            for k, m in enumerate(kept):
                video_predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=k,
                    mask=m["segmentation"].astype(bool),
                )

        K = len(kept)
        masks_per_frame = np.zeros((n, K, h, w), dtype=bool)

        print(f"propagating across {n} frames × {K} objects ...")
        cuda_sync(device)
        t0 = time.time()
        with torch.inference_mode(), autocast_ctx:
            for frame_idx, obj_ids, logits in video_predictor.propagate_in_video(state):
                for j, oid in enumerate(obj_ids):
                    masks_per_frame[frame_idx, oid] = (
                        (logits[j] > 0.0).squeeze(0).cpu().numpy())
        cuda_sync(device)
        elapsed = time.time() - t0
        if device.type == "cuda":
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  done in {elapsed:.1f}s ({n / elapsed:.2f} fps)  "
                  f"peak GPU mem: {gpu_mem_mb:.0f} MiB")
        else:
            print(f"  done in {elapsed:.1f}s ({n / elapsed:.2f} fps)")
        print()

        # ----- Render overlay video -----
        out_path = (PROJECT_ROOT
                    / f"outputs/{today_pretty()} - sam2 auto-to-video [{args.tag}].mp4")
        print(f"rendering preview → {out_path.name}")
        for fourcc_tag in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            if writer.isOpened():
                print(f"  writer: {fourcc_tag}")
                break
        else:
            sys.exit(f"failed to open VideoWriter at {out_path}")

        try:
            for i in range(n):
                frame = cv2.imread(str(frames_dir / f"{i:05d}.jpg"))
                active = 0
                # Draw smaller masks on top of larger so nothing is buried.
                order = np.argsort([-masks_per_frame[i, k].sum() for k in range(K)])
                for k in order:
                    sel = masks_per_frame[i, k]
                    if not sel.any():
                        continue
                    active += 1
                    color = np.array(stable_color(k), dtype=np.uint8)
                    frame[sel] = (0.5 * frame[sel] + 0.5 * color).astype(np.uint8)
                    ys, xs = np.where(sel)
                    bb = (int(xs.min()), int(ys.min()),
                          int(xs.max()), int(ys.max()))
                    cv2.rectangle(frame, bb[:2], bb[2:],
                                  tuple(int(c) for c in color), 1)
                    cv2.putText(frame, f"{k}", (bb[0] + 2, bb[1] + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                tuple(int(c) for c in color), 1, cv2.LINE_AA)

                hud = f"frame {i + 1}/{n}  active: {active}/{K}"
                cv2.rectangle(frame, (4, h - 28), (16 + 9 * len(hud), h - 6),
                              (0, 0, 0), -1)
                cv2.putText(frame, hud, (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)
        finally:
            writer.release()

        # Re-encode to h264 for clean macOS playback.
        if shutil.which("ffmpeg") is not None:
            tmp_h264 = out_path.with_suffix(".h264.mp4")
            print(f"re-encoding to h264 → {tmp_h264.name}")
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(out_path),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                 "-pix_fmt", "yuv420p", str(tmp_h264)],
                capture_output=True, text=True)
            if r.returncode == 0:
                out_path.unlink()
                tmp_h264.rename(out_path)
                print("  done")
            else:
                print(f"  ffmpeg failed: {r.stderr.strip()[:200]}")

        print(f"\n=== done → {out_path.name} ===")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
