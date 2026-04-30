"""Side-by-side stereo playback for the dual Pi-Cam recordings.

Project convention: cam1 = left, cam0 = right. The calibration `.npz` carries
remap tables that undo lens distortion AND row-align the two views, so a
correct stereo pair lies on the same scanline after rectify.

Important: `calibrate.py` bakes its maps at `alpha=1` (preserve full source
FOV with black padding) because triangulation wants every pixel. For *viewing*
the wide-FOV streams that's ugly — the rectified content scrunches into a
small region surrounded by black, which looks like "two little spheres".
This script recomputes the rectify maps on the fly from K/dist/R/T using
`--alpha` (default `0` = crop to the all-valid rectangle, i.e. clean
rectilinear left-right view). Pass `--alpha 1` to reproduce the saved
calibration's behaviour, or any value in between.

Usage (from repo root):
  .venv/bin/python scripts/play_stereo.py
  .venv/bin/python scripts/play_stereo.py --left raw/cam1_20260427_185132.h264 \
                                          --right raw/cam0_20260427_185132.h264
  .venv/bin/python scripts/play_stereo.py --alpha 0.5
  .venv/bin/python scripts/play_stereo.py --no-rectify
  .venv/bin/python scripts/play_stereo.py --calib "outputs/27th April 2026 - stereo calibration.npz"

Notes:
  - Raw .h264 has no framerate metadata, so playback fps is set by --fps
    (default 30, matching the rig's capture rate). Muxed .mp4 also plays at
    --fps; if you want true file-fps, pass it explicitly.
  - Scale halves the SBS render by default so 2304x1296 inputs fit on a
    typical laptop display (4608x1296 native -> 2304x648 windowed).

Controls:
  q / Esc   quit
  space     pause / resume
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT = PROJECT_ROOT / "raw/cam1.mp4"
DEFAULT_RIGHT = PROJECT_ROOT / "raw/cam0.mp4"
DEFAULT_CALIB = PROJECT_ROOT / "outputs/30th April 2026 wide - stereo calibration.npz"


def from_root(p):
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_rectify_maps(calib_path, alpha):
    """Build rectify remap tables from the calibration .npz.

    `alpha=None` -> use the maps already baked into the .npz (alpha=1 from
    calibrate.py: preserves full source FOV with black padding around a
    small valid region).

    `alpha` in [0, 1] -> recompute via cv2.stereoRectify + initUndistortRectifyMap.
    alpha=0 crops to the largest all-valid rectangle (clean rectilinear view,
    best for casual viewing); alpha=1 keeps everything (matches the saved maps).
    """
    c = np.load(str(calib_path))
    size = tuple(int(v) for v in c["image_size"])
    if alpha is None:
        return {
            "size": size, "alpha": "calib (=1)",
            "map_lx": c["map_left_x"], "map_ly": c["map_left_y"],
            "map_rx": c["map_right_x"], "map_ry": c["map_right_y"],
        }
    K_l, d_l = c["K_left"], c["dist_left"]
    K_r, d_r = c["K_right"], c["dist_right"]
    R, T = c["R"], c["T"]
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K_l, d_l, K_r, d_r, size, R, T, alpha=alpha,
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, d_l, R1, P1, size, cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, d_r, R2, P2, size, cv2.CV_32FC1)
    return {
        "size": size, "alpha": f"{alpha:.2f}",
        "map_lx": map_lx, "map_ly": map_ly,
        "map_rx": map_rx, "map_ry": map_ry,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Play stereo cam1 (left) + cam0 (right) videos side-by-side, rectified.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--left", default=str(DEFAULT_LEFT), help="left clip (cam1); .h264 or .mp4")
    ap.add_argument("--right", default=str(DEFAULT_RIGHT), help="right clip (cam0); .h264 or .mp4")
    ap.add_argument("--calib", default=str(DEFAULT_CALIB), help="stereo calibration .npz with rectify maps")
    ap.add_argument("--no-rectify", action="store_true", help="show raw fish-eye-ish frames instead of rectified")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="rectify free-scaling: 0=crop to all-valid (clean rectilinear, default), "
                         "1=keep full source FOV (small valid region with black padding). "
                         "Pass 'calib' to use the maps baked into the .npz (alpha=1 by calibrate.py).")
    ap.add_argument("--fps", type=float, default=30.0, help="playback fps (default 30 = rig's capture rate)")
    ap.add_argument("--scale", type=float, default=0.5, help="display scale per view (1.0 = native)")
    args = ap.parse_args()

    left = from_root(args.left)
    right = from_root(args.right)
    if not left.is_file():
        sys.exit(f"missing left clip: {left}")
    if not right.is_file():
        sys.exit(f"missing right clip: {right}")

    cap_l = cv2.VideoCapture(str(left))
    cap_r = cv2.VideoCapture(str(right))
    if not (cap_l.isOpened() and cap_r.isOpened()):
        sys.exit(f"VideoCapture could not open {left} or {right}")

    rectify = not args.no_rectify
    maps = None
    if rectify:
        calib = from_root(args.calib)
        if not calib.is_file():
            sys.exit(f"calibration not found: {calib} - pass --no-rectify or --calib <path>")
        maps = load_rectify_maps(calib, alpha=args.alpha)
        print(f"rectify on:  {calib.name} (target {maps['size'][0]}x{maps['size'][1]}, alpha={maps['alpha']})")
    else:
        print("rectify off (--no-rectify)")
    print(f"left  = {left}")
    print(f"right = {right}")
    print(f"fps   = {args.fps:.2f}, display scale = {args.scale:.2f}")

    target_dt = 1.0 / args.fps
    paused = False
    win = "stereo playback  [q]uit  [space] pause"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    next_frame_t = time.time()
    last_sbs = None
    while True:
        if not paused:
            ok_l, fl = cap_l.read()
            ok_r, fr = cap_r.read()
            if not (ok_l and ok_r):
                print("end of video")
                break
            if rectify:
                fl = cv2.remap(fl, maps["map_lx"], maps["map_ly"], cv2.INTER_LINEAR)
                fr = cv2.remap(fr, maps["map_rx"], maps["map_ry"], cv2.INTER_LINEAR)
            if fl.shape[0] != fr.shape[0]:
                h = min(fl.shape[0], fr.shape[0])
                fl, fr = fl[:h], fr[:h]
            sbs = np.hstack([fl, fr])
            if args.scale != 1.0:
                sbs = cv2.resize(sbs, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            last_sbs = sbs
            next_frame_t += target_dt

        if last_sbs is not None:
            cv2.imshow(win, last_sbs)

        if paused:
            wait_ms = 30
        else:
            wait_ms = max(1, int((next_frame_t - time.time()) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            if not paused:
                next_frame_t = time.time()

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
