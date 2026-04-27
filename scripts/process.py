"""Stitch the two head-mounted Pi Cam 3 video streams into a single panorama video.

cam0 = right-eye view, cam1 = left-eye view (cameras toed out 24 deg, rigidly mounted).
Because the rig is rigid, we estimate ONE homography from a representative frame pair
and reuse it for every frame.

Usage:
  .venv/bin/python process.py
"""

import sys
import time

import cv2
import numpy as np

from dated import today_pretty

# Raw 4-min recordings live in repo root and are gitignored (too large for GitHub).
LEFT_PATH = "cam1.mp4"
RIGHT_PATH = "cam0.mp4"
OUT_PATH = f"outputs/{today_pretty()} - stitched panorama.mp4"

LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESH = 4.0
CALIB_FRAME_FRACTION = 0.5  # use the middle frame to estimate H


def estimate_homography(left_bgr, right_bgr):
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(left_bgr, None)
    kp_r, des_r = sift.detectAndCompute(right_bgr, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_r, des_l, k=2)
    good = [m for m, n in knn if m.distance < LOWE_RATIO * n.distance]
    if len(good) < 20:
        raise RuntimeError(f"Only {len(good)} good matches - not enough overlap.")

    src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_l[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"calibration: {len(good)} good matches, {inliers} RANSAC inliers")
    return H


def compute_canvas(h_l, w_l, h_r, w_r, H):
    corners_r = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)
    corners_r_in_l = cv2.perspectiveTransform(corners_r, H)
    corners_l = np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners_l, corners_r_in_l], axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_size = (x_max - x_min, y_max - y_min)
    return canvas_size, translation, (-x_min, -y_min)


def stitch_frame(left, right, warp_mat, canvas_size, paste_offset):
    warped = cv2.warpPerspective(right, warp_mat, canvas_size)
    ox, oy = paste_offset
    h_l, w_l = left.shape[:2]
    warped[oy:oy + h_l, ox:ox + w_l] = left
    return warped


def grab_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return frame


def main():
    cap_l = cv2.VideoCapture(LEFT_PATH)
    cap_r = cv2.VideoCapture(RIGHT_PATH)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise FileNotFoundError(f"Could not open {LEFT_PATH} or {RIGHT_PATH}")

    fps = cap_l.get(cv2.CAP_PROP_FPS)
    n_frames = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT), cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"left  {LEFT_PATH}: {w_l}x{h_l}")
    print(f"right {RIGHT_PATH}: {w_r}x{h_r}")
    print(f"frames={n_frames} fps={fps:.3f}")

    calib_idx = int(n_frames * CALIB_FRAME_FRACTION)
    H = estimate_homography(grab_frame(cap_l, calib_idx), grab_frame(cap_r, calib_idx))
    canvas_size, translation, paste_offset = compute_canvas(h_l, w_l, h_r, w_r, H)
    warp_mat = translation @ H
    print(f"canvas: {canvas_size[0]}x{canvas_size[1]}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, canvas_size)
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t0 = time.time()
    for i in range(n_frames):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break
        out = stitch_frame(fl, fr, warp_mat, canvas_size, paste_offset)
        writer.write(out)
        if i % 60 == 0 or i == n_frames - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (n_frames - i - 1) / rate if rate else 0
            sys.stdout.write(f"\rframe {i+1}/{n_frames}  {rate:.1f} fps  eta {eta:.0f}s   ")
            sys.stdout.flush()
    print()

    cap_l.release()
    cap_r.release()
    writer.release()
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
