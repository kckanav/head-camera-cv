"""Dense stereo depth from a rectified clip pair, rendered as an AR-style
blue depth mask overlaid on the original frame (closer = more solid blue),
or a turbo colormap side-by-side, or both.

Pipeline per frame pair:
  1. Rectify both frames with maps recomputed at alpha=0 (full-frame valid -
     SGBM should match real pixels, not extrapolated black borders).
  2. cv2.StereoSGBM_create on the grayscale rectified pair -> disparity (1/16 px).
  3. Convert disparity -> metric Z via  Z = fx * baseline / disp  (same as
     reprojectImageTo3D's z, just without the (X, Y) channels).
  4. Render. Three styles:
       overlay  (default) — left rectified frame with a blue depth mask
                            alpha-blended on top. Near = solid blue, far =
                            transparent. Looks like the iPhone/ARKit depth
                            visualisation.
       side               — left rectified | turbo-coloured depth, the
                            classic side-by-side render.
       both               — left rectified | overlay, so you can compare the
                            unmasked frame to the masked one.

Defaults match the wide-FOV WiLoR stereo demo (10-s interaction clip + wide
calibration). Pass --left/--right/--calib/--tag to swap to the narrow clip
or any other pair you have rectified.

Outputs:
  outputs/<date> - stereo depthmap [tag].mp4   (style baked into the render)

Usage (from repo root):
  .venv/bin/python scripts/depthmap.py                       # blue overlay (default)
  .venv/bin/python scripts/depthmap.py --style both          # SBS rectified | overlay
  .venv/bin/python scripts/depthmap.py --style side          # SBS rectified | turbo
  .venv/bin/python scripts/depthmap.py --tag "narrow first minute" \\
    --left "inputs/25th April 2026 - cam1 first minute.mp4" \\
    --right "inputs/25th April 2026 - cam0 first minute.mp4" \\
    --calib "outputs/27th April 2026 - stereo calibration.npz"
  .venv/bin/python scripts/depthmap.py --max-frames 60   # quick smoke
  .venv/bin/python scripts/depthmap.py --no-open          # don't auto-open the result
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LEFT = PROJECT_ROOT / "inputs/27th April 2026 wide - cam1 interaction first 10s.mp4"
DEFAULT_RIGHT = PROJECT_ROOT / "inputs/27th April 2026 wide - cam0 interaction first 10s.mp4"
DEFAULT_CALIB = PROJECT_ROOT / "outputs/30th April 2026 wide - stereo calibration.npz"
DEFAULT_TAG = "wide first 10s interaction"

Z_MIN_M = 0.20   # closer than 20 cm = unreliable / outside SGBM disparity range
Z_MAX_M = 1.50   # farther than 1.5 m -> tiny disparity, mostly noise


def from_root(p):
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def build_rectifier(calib_path, alpha=0.0):
    c = np.load(str(calib_path))
    size = tuple(int(v) for v in c["image_size"])
    K_l, d_l = c["K_left"], c["dist_left"]
    K_r, d_r = c["K_right"], c["dist_right"]
    R, T = c["R"], c["T"]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, d_l, K_r, d_r, size, R, T, alpha=alpha,
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, d_l, R1, P1, size, cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, d_r, R2, P2, size, cv2.CV_32FC1)
    return {
        "size": size,
        "map_lx": map_lx, "map_ly": map_ly,
        "map_rx": map_rx, "map_ry": map_ry,
        "Q": Q,
        "fx": float(P1[0, 0]),                # rectified focal length, pixels
        "baseline_m": float(c["baseline_m"]),  # from stereoCalibrate
    }


def make_sgbm(num_disp, block):
    # P1/P2 follow the OpenCV docs' rule of thumb (8*N_channels*block^2,
    # 32*N_channels*block^2). Bigger -> smoother disparity, less detail.
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,            # must be a multiple of 16
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def disparity_to_depth(disp_int16, fx, baseline_m):
    """Convert SGBM's 1/16-fixed-point disparity to metric Z. Invalid -> NaN."""
    disp_f = disp_int16.astype(np.float32) / 16.0
    valid = disp_f > 0.5  # SGBM marks unmatched as <0; very small dist also unreliable
    z = np.full_like(disp_f, np.nan, dtype=np.float32)
    z[valid] = fx * baseline_m / disp_f[valid]
    return z, valid


def colorize_depth(z, valid, z_min=Z_MIN_M, z_max=Z_MAX_M):
    h, w = z.shape
    z_clip = np.where(valid, np.clip(z, z_min, z_max), z_min)
    norm = ((z_clip - z_min) / (z_max - z_min) * 255.0).astype(np.uint8)
    # Invert so near = warm, far = cool (turbo runs blue->red; flip for "near is red").
    color = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def overlay_legend(img, z_min, z_max, near_far_swap=True):
    """Draw a small horizontal colorbar with min/max metric labels."""
    h, w = img.shape[:2]
    bar_h, bar_w = 14, 220
    x0, y0 = 12, h - bar_h - 26
    grad = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, -1)
    grad = np.repeat(grad, bar_h, axis=0)
    if near_far_swap:
        grad = 255 - grad
    bar = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    img[y0:y0 + bar_h, x0:x0 + bar_w] = bar
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    label = f"{z_min*100:.0f} cm  <-  depth  ->  {z_max*100:.0f} cm"
    cv2.putText(img, label, (x0, y0 + bar_h + 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)


def overlay_blue_mask(bgr_frame, z, valid, z_min=Z_MIN_M, z_max=Z_MAX_M, max_alpha=0.75):
    """Composite a blue depth mask on the original frame.

    Per-pixel alpha is the closeness in [0, 1] (1 = right at z_min, 0 = at z_max
    or invalid), capped at max_alpha so the original frame stays visible. The
    blue tint is BGR (255, 90, 30) - reads as a saturated cyan-ish blue.
    """
    bgr_f = bgr_frame.astype(np.float32)
    z_clip = np.where(valid, np.clip(z, z_min, z_max), z_max)
    closeness = (z_max - z_clip) / (z_max - z_min)        # 1 = near, 0 = far
    closeness = np.where(valid, closeness, 0.0).astype(np.float32)
    alpha = (closeness * max_alpha)[..., None]            # (h, w, 1)
    blue = np.zeros_like(bgr_f)
    blue[..., 0] = 255   # B
    blue[..., 1] = 90    # touch of green so it doesn't read as pure neon
    blue[..., 2] = 30
    out = bgr_f * (1.0 - alpha) + blue * alpha
    return out.astype(np.uint8)


def annotate_overlay(img, near_z_cm, max_alpha):
    """Small HUD in the top-left corner of the overlay frame."""
    cv2.putText(img, f"depth mask  near={near_z_cm:.0f}cm  alpha<={max_alpha:.0f}%",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(
        description="Dense stereo depth map for a calibrated clip pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--left", default=str(DEFAULT_LEFT))
    ap.add_argument("--right", default=str(DEFAULT_RIGHT))
    ap.add_argument("--calib", default=str(DEFAULT_CALIB))
    ap.add_argument("--tag", default=DEFAULT_TAG, help="filename suffix for the output mp4")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames (smoke testing)")
    ap.add_argument("--num-disp", type=int, default=256,
                    help="SGBM numDisparities (multiple of 16). 256 covers ~10 cm to inf at the wide rig.")
    ap.add_argument("--block", type=int, default=5, help="SGBM blockSize (odd, 3-11 typical)")
    ap.add_argument("--style", choices=("overlay", "side", "both"), default="overlay",
                    help="render style: blue mask overlay (default), turbo SBS, or both")
    ap.add_argument("--max-alpha", type=float, default=0.75,
                    help="max blue-mask opacity in overlay mode (0..1, default 0.75)")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the result on macOS")
    args = ap.parse_args()

    if args.num_disp % 16 != 0:
        sys.exit(f"--num-disp must be a multiple of 16 (got {args.num_disp})")

    left, right, calib = from_root(args.left), from_root(args.right), from_root(args.calib)
    for p in (left, right, calib):
        if not p.is_file():
            sys.exit(f"missing: {p}")

    rect = build_rectifier(calib, alpha=0.0)
    w, h = rect["size"]
    print(f"calib:    {calib.name}")
    print(f"          size {w}x{h}  fx={rect['fx']:.1f}  baseline={rect['baseline_m']*1000:.1f} mm")
    print(f"left  =   {left}")
    print(f"right =   {right}")

    cap_l = cv2.VideoCapture(str(left))
    cap_r = cv2.VideoCapture(str(right))
    if not (cap_l.isOpened() and cap_r.isOpened()):
        sys.exit("VideoCapture failed")
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT), cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.max_frames is not None:
        n = min(n, args.max_frames)
    print(f"frames:   {n} @ {fps:.2f} fps")

    sgbm = make_sgbm(args.num_disp, args.block)

    style_suffix = {"overlay": "blue", "side": "turbo", "both": "blue+turbo"}[args.style]
    out_path = PROJECT_ROOT / (
        f"outputs/{today_pretty()} - stereo depthmap [{args.tag}] ({style_suffix}).mp4"
    )
    out_w = w if args.style == "overlay" else w * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, h))
    if not writer.isOpened():
        sys.exit(f"could not open writer: {out_path}")
    print(f"style:    {args.style}  -> output {out_w}x{h}")

    z_samples = []
    t0 = time.time()
    for i in range(n):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break
        rl = cv2.remap(fl, rect["map_lx"], rect["map_ly"], cv2.INTER_LINEAR)
        rr = cv2.remap(fr, rect["map_rx"], rect["map_ry"], cv2.INTER_LINEAR)

        gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
        disp = sgbm.compute(gl, gr)  # int16, 1/16 px

        z, valid = disparity_to_depth(disp, rect["fx"], rect["baseline_m"])
        if valid.any():
            z_samples.append(np.nanmedian(z[valid & np.isfinite(z)]))

        if args.style == "overlay":
            frame = overlay_blue_mask(rl, z, valid, max_alpha=args.max_alpha)
            annotate_overlay(frame, Z_MIN_M * 100, args.max_alpha * 100)
        elif args.style == "side":
            depth_vis = colorize_depth(z, valid)
            overlay_legend(depth_vis, Z_MIN_M, Z_MAX_M)
            frame = np.hstack([rl, depth_vis])
        else:  # both
            blue = overlay_blue_mask(rl, z, valid, max_alpha=args.max_alpha)
            annotate_overlay(blue, Z_MIN_M * 100, args.max_alpha * 100)
            frame = np.hstack([rl, blue])
        writer.write(frame)

        if i % 15 == 0 or i == n - 1:
            dt = time.time() - t0
            rate = (i + 1) / dt if dt else 0
            sys.stdout.write(f"\rframe {i+1}/{n}  {rate:.2f} fps  "
                             f"({dt:.1f}s elapsed)   ")
            sys.stdout.flush()
    print()

    cap_l.release()
    cap_r.release()
    writer.release()

    if z_samples:
        zs = np.array(z_samples) * 100
        print(f"per-frame median Z (cm): "
              f"med {np.median(zs):.1f}, "
              f"5-95 [{np.percentile(zs,5):.1f}, {np.percentile(zs,95):.1f}]")
    print(f"wrote {out_path}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(out_path)], check=False)


if __name__ == "__main__":
    main()
