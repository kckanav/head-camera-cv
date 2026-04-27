"""Stage 4: sparse 3D hand keypoints by stereo triangulation.

Pipeline per frame pair (left = cam1, right = cam0, per the calibration):
  1. Rectify both frames with the precomputed maps from
     `outputs/<date> - stereo calibration.npz`.
  2. Run MediaPipe HandLandmarker on each rectified view independently.
  3. Pair left/right hand detections by wrist-row proximity. After
     rectification the epipolar constraint is "same image row", so a
     correct pair must have similar wrist y-coordinates.
  4. Triangulate the 21 landmarks per paired hand to 3D positions in the
     rectified-left camera frame with cv2.triangulatePoints (units: metres,
     because stereoCalibrate was given object points in metres).
  5. Reject any pair whose triangulated wrist Z falls outside a plausible
     range - catches wrong-hand pairings that slip through the row check.

Outputs:
  outputs/<date> - stereo hands annotated.mp4   side-by-side rectified video
                                                with skeletons, wrist depth,
                                                and pinch (thumb-index)
                                                aperture annotations
  outputs/<date> - stereo hand 3d.npz           per-frame triangulated
                                                landmarks, NaN where missing
"""

import subprocess
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from dated import today_pretty


CALIB_PATH = "outputs/27th April 2026 - stereo calibration.npz"
LEFT_CLIP = "inputs/25th April 2026 - cam1 first minute.mp4"   # cam1 = left
RIGHT_CLIP = "inputs/25th April 2026 - cam0 first minute.mp4"  # cam0 = right
OUT_VIDEO = f"outputs/{today_pretty()} - stereo hands annotated.mp4"
OUT_NPZ = f"outputs/{today_pretty()} - stereo hand 3d.npz"

MODEL_PATH = "models/hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8

# After rectification, a true match should agree in y to within ~5 px
# (landmark precision); 18 px gives slack without admitting cross-pairs
# when both hands are at similar heights.
ROW_TOLERANCE_PX = 18.0
Z_MIN_M = 0.10   # closer than 10 cm = certainly wrong
Z_MAX_M = 2.00   # farther than 2 m = arm doesn't reach


def load_calib(path):
    c = np.load(path)
    return {
        "image_size": tuple(int(v) for v in c["image_size"]),
        "map_lx": c["map_left_x"], "map_ly": c["map_left_y"],
        "map_rx": c["map_right_x"], "map_ry": c["map_right_y"],
        "P1": c["P1"], "P2": c["P2"],
        "baseline_m": float(c["baseline_m"]),
    }


def make_landmarker():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame_bgr, ts_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect_for_video(mp_image, ts_ms)
    if not res.hand_landmarks:
        return []
    h, w = frame_bgr.shape[:2]
    return [
        np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
        for lms in res.hand_landmarks
    ]


def match_hands(left_hands, right_hands):
    """Greedy match by wrist y-coordinate (epipolar constraint after rectification)."""
    pairs = []
    if not left_hands or not right_hands:
        return pairs
    used = set()
    for lpts in left_hands:
        ly = lpts[WRIST, 1]
        best, best_dy = None, ROW_TOLERANCE_PX
        for ri, rpts in enumerate(right_hands):
            if ri in used:
                continue
            dy = abs(rpts[WRIST, 1] - ly)
            if dy < best_dy:
                best, best_dy = ri, dy
        if best is not None:
            pairs.append((lpts, right_hands[best]))
            used.add(best)
    return pairs


def triangulate(P1, P2, lpts, rpts):
    pts4 = cv2.triangulatePoints(
        P1, P2, lpts.T.astype(np.float64), rpts.T.astype(np.float64)
    )
    return (pts4[:3] / pts4[3]).T   # (21, 3) metres in left rectified frame


def draw_skeleton(img, pts2d, color):
    pts = [(int(x), int(y)) for x, y in pts2d]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 4, color, 1)


def annotate(img, pts2d, pts3d, color):
    draw_skeleton(img, pts2d, color)
    z_cm = pts3d[WRIST, 2] * 100.0
    pinch_mm = float(np.linalg.norm(pts3d[THUMB_TIP] - pts3d[INDEX_TIP])) * 1000.0
    wx, wy = int(pts2d[WRIST, 0]), int(pts2d[WRIST, 1])
    cv2.putText(img, f"wrist {z_cm:5.1f} cm", (wx + 8, wy - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, f"pinch {pinch_mm:5.0f} mm", (wx + 8, wy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    calib = load_calib(CALIB_PATH)
    cap_l = cv2.VideoCapture(LEFT_CLIP)
    cap_r = cv2.VideoCapture(RIGHT_CLIP)
    if not (cap_l.isOpened() and cap_r.isOpened()):
        sys.exit(f"could not open {LEFT_CLIP} or {RIGHT_CLIP}")
    fps = cap_l.get(cv2.CAP_PROP_FPS)
    w, h = calib["image_size"]
    n = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT),
                cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"{n} frames @ {fps:.1f} fps, image {w}x{h}, baseline {calib['baseline_m']*1000:.1f} mm")

    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))

    landmarker_l = make_landmarker()
    landmarker_r = make_landmarker()

    # NaN-padded arrays - per-frame slot 0 = leftmost hand (by left-view wrist X), slot 1 = next.
    landmarks_3d = np.full((n, 2, 21, 3), np.nan, dtype=np.float32)
    landmarks_2d_l = np.full((n, 2, 21, 2), np.nan, dtype=np.float32)
    landmarks_2d_r = np.full((n, 2, 21, 2), np.nan, dtype=np.float32)

    n_left, n_right, n_paired_raw, n_paired_kept, n_rejected = 0, 0, 0, 0, 0
    wrist_z = []
    pinch_d = []

    t0 = time.time()
    for i in range(n):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break
        rect_l = cv2.remap(fl, calib["map_lx"], calib["map_ly"], cv2.INTER_LINEAR)
        rect_r = cv2.remap(fr, calib["map_rx"], calib["map_ry"], cv2.INTER_LINEAR)
        ts_ms = int(1000 * i / fps)
        lh = detect_hands(landmarker_l, rect_l, ts_ms)
        rh = detect_hands(landmarker_r, rect_r, ts_ms)
        if lh:
            n_left += 1
        if rh:
            n_right += 1

        raw_pairs = match_hands(lh, rh)
        if raw_pairs:
            n_paired_raw += 1

        # Triangulate, filter by Z bounds, then sort by left-view wrist X for slot stability.
        kept = []
        for lpts, rpts in raw_pairs:
            pts3d = triangulate(calib["P1"], calib["P2"], lpts, rpts)
            z = pts3d[WRIST, 2]
            if not (Z_MIN_M <= z <= Z_MAX_M):
                n_rejected += 1
                continue
            kept.append((lpts, rpts, pts3d))
        kept.sort(key=lambda t: t[0][WRIST, 0])  # leftmost wrist X first

        if kept:
            n_paired_kept += 1
        for slot, (lpts, rpts, pts3d) in enumerate(kept[:2]):
            color = [(0, 200, 255), (255, 100, 0)][slot]
            wrist_z.append(pts3d[WRIST, 2])
            pinch_d.append(np.linalg.norm(pts3d[THUMB_TIP] - pts3d[INDEX_TIP]))
            landmarks_3d[i, slot] = pts3d
            landmarks_2d_l[i, slot] = lpts
            landmarks_2d_r[i, slot] = rpts
            annotate(rect_l, lpts, pts3d, color)
            annotate(rect_r, rpts, pts3d, color)

        sbs = np.hstack([rect_l, rect_r])
        writer.write(sbs)
        if i % 30 == 0 or i == n - 1:
            sys.stdout.write(f"\rframe {i+1}/{n}  paired {n_paired_kept}  "
                             f"({(time.time()-t0):.1f}s)   ")
            sys.stdout.flush()
    print()

    cap_l.release()
    cap_r.release()
    writer.release()
    landmarker_l.close()
    landmarker_r.close()

    np.savez(
        OUT_NPZ,
        landmarks_3d=landmarks_3d,
        landmarks_2d_left=landmarks_2d_l,
        landmarks_2d_right=landmarks_2d_r,
        fps=np.array(fps),
        image_size=np.array([w, h]),
        calib_path=np.array(CALIB_PATH),
    )

    print(f"hands detected (any):           left {n_left}/{n}  right {n_right}/{n}")
    print(f"raw stereo pairs (row-matched): {n_paired_raw}/{n}")
    print(f"after Z sanity filter:          {n_paired_kept}/{n} "
          f"({100*n_paired_kept/max(n,1):.1f}%)   rejected hands: {n_rejected}")
    if wrist_z:
        z = np.array(wrist_z) * 100.0
        p = np.array(pinch_d) * 1000.0
        print(f"wrist depth (cm):  median {np.median(z):.1f}, "
              f"5-95 pct [{np.percentile(z,5):.1f}, {np.percentile(z,95):.1f}], "
              f"min {z.min():.1f}, max {z.max():.1f}")
        print(f"pinch dist (mm):   median {np.median(p):.0f}, "
              f"5-95 pct [{np.percentile(p,5):.0f}, {np.percentile(p,95):.0f}], "
              f"min {p.min():.0f}, max {p.max():.0f}")
    print(f"wrote {OUT_VIDEO}")
    print(f"wrote {OUT_NPZ}")

    if sys.platform == "darwin":
        subprocess.run(["open", OUT_VIDEO], check=False)


if __name__ == "__main__":
    main()
