"""Stereo calibration for the dual Pi Cam 3 head-mounted rig.

Inputs (must be recorded with the rig held still and lens focus locked):
  inputs/<date> - cam0 calibration.mp4
  inputs/<date> - cam1 calibration.mp4

Procedure:
  1. Sample frame pairs at a fixed interval from the two videos.
  2. Detect ChArUco corners in each frame independently.
  3. Per-camera calibration with cv2.calibrateCamera using the matched ChArUco
     object/image points (intrinsics + distortion).
  4. Stereo calibration with CALIB_FIX_INTRINSIC (only solve R, T).
  5. Compute rectification maps with cv2.stereoRectify.
  6. Save everything to outputs/<date> - stereo calibration.npz.
  7. Render a rectified sample pair with horizontal scan lines so the
     calibration can be verified visually (corresponding points must lie on the
     same horizontal scanline).

Three numbers to look at after running:
  - Per-camera reprojection error: < 0.5 px good, < 1 px acceptable.
  - Stereo reprojection error: < 1 px good, < 1.5 px acceptable.
  - Recovered camera-to-camera angle: should be near 24 deg (rig spec).
"""

import sys
import time

import cv2
import cv2.aruco as aruco
import numpy as np

from dated import today_pretty

# --- Board geometry (matches make_calibration_board.py, scaled to actual print) ---
SQUARES_X = 9
SQUARES_Y = 6
SQUARE_M = 0.02909  # measured edge of one printed square, in metres
MARKER_M = 0.02133  # = 22mm * (29.09/30); preserves design ratio
DICT_ID = aruco.DICT_5X5_50

# --- Capture sampling ---
SAMPLE_EVERY_N_FRAMES = 15  # at 30fps, one pair per 0.5s -> ~120 pairs from 60s
MIN_CORNERS_PER_VIEW = 8    # both views must see at least this many ChArUco corners

# --- Paths ---
TODAY = today_pretty()
LEFT_VIDEO = "inputs/27th April 2026 - cam1 calibration.mp4"   # cam1 = left
RIGHT_VIDEO = "inputs/27th April 2026 - cam0 calibration.mp4"  # cam0 = right
OUT_NPZ = f"outputs/{TODAY} - stereo calibration.npz"
OUT_SANITY = f"outputs/{TODAY} - rectified pair sanity.jpg"


def make_board():
    dictionary = aruco.getPredefinedDictionary(DICT_ID)
    board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_M, MARKER_M, dictionary)
    return board, dictionary


def collect_frames(left_path, right_path):
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)
    if not (cap_l.isOpened() and cap_r.isOpened()):
        raise FileNotFoundError(f"could not open {left_path} or {right_path}")
    n = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT), cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pairs = []
    for i in range(0, n, SAMPLE_EVERY_N_FRAMES):
        cap_l.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if ok_l and ok_r:
            pairs.append((fl, fr))
    cap_l.release()
    cap_r.release()
    return pairs, (w, h)


def detect_charuco(detector, gray):
    """Returns (charuco_corners, charuco_ids) or (None, None)."""
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_corners is None or ch_ids is None or len(ch_ids) < MIN_CORNERS_PER_VIEW:
        return None, None
    return ch_corners, ch_ids


def calibrate_camera(board, all_corners, all_ids, image_size):
    obj_pts_list = []
    img_pts_list = []
    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            continue
        obj_pts_list.append(obj_pts)
        img_pts_list.append(img_pts)
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts_list, img_pts_list, image_size, None, None
    )
    return rms, K, dist


def paired_object_image_points(board, left_data, right_data):
    """For frames where BOTH views detected enough corners, build matched
    object/image-point arrays (same physical corners visible in both views)."""
    obj_pts_all, l_pts_all, r_pts_all = [], [], []
    for (lc, li), (rc, ri) in zip(left_data, right_data):
        if lc is None or rc is None:
            continue
        # Find ChArUco corner IDs visible in BOTH views.
        common_ids = np.intersect1d(li.flatten(), ri.flatten())
        if len(common_ids) < MIN_CORNERS_PER_VIEW:
            continue
        l_idx = np.array([np.where(li.flatten() == cid)[0][0] for cid in common_ids])
        r_idx = np.array([np.where(ri.flatten() == cid)[0][0] for cid in common_ids])
        l_corners = lc[l_idx]
        r_corners = rc[r_idx]
        common_ids_col = common_ids.reshape(-1, 1).astype(np.int32)
        obj_pts, _ = board.matchImagePoints(l_corners, common_ids_col)
        if obj_pts is None or len(obj_pts) < MIN_CORNERS_PER_VIEW:
            continue
        obj_pts_all.append(obj_pts.astype(np.float32))
        l_pts_all.append(l_corners.astype(np.float32).reshape(-1, 1, 2))
        r_pts_all.append(r_corners.astype(np.float32).reshape(-1, 1, 2))
    return obj_pts_all, l_pts_all, r_pts_all


def angle_from_R(R):
    """Total rotation angle (deg) encoded in a 3x3 rotation matrix."""
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_theta))


def render_sanity(left, right, K_l, dist_l, K_r, dist_r, R1, R2, P1, P2, image_size):
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, dist_l, R1, P1, image_size, cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, dist_r, R2, P2, image_size, cv2.CV_32FC1)
    rect_l = cv2.remap(left, map_lx, map_ly, cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, map_rx, map_ry, cv2.INTER_LINEAR)
    sbs = np.hstack([rect_l, rect_r])
    h = sbs.shape[0]
    for y in range(0, h, 40):
        cv2.line(sbs, (0, y), (sbs.shape[1], y), (0, 255, 0), 1)
    return sbs, (map_lx, map_ly, map_rx, map_ry)


def main():
    print(f"[1/6] Loading frame pairs (every {SAMPLE_EVERY_N_FRAMES}th frame)...")
    pairs, image_size = collect_frames(LEFT_VIDEO, RIGHT_VIDEO)
    print(f"      {len(pairs)} pairs, image size {image_size}")

    board, _ = make_board()
    detector = aruco.CharucoDetector(board)

    print("[2/6] Detecting ChArUco corners in each frame...")
    left_data, right_data = [], []
    n_both = 0
    t0 = time.time()
    for fl, fr in pairs:
        gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        l = detect_charuco(detector, gl)
        r = detect_charuco(detector, gr)
        left_data.append(l)
        right_data.append(r)
        if l[0] is not None and r[0] is not None:
            n_both += 1
    print(f"      {n_both}/{len(pairs)} pairs had usable detections in BOTH views ({time.time()-t0:.1f}s)")
    if n_both < 12:
        sys.exit("not enough usable pairs - recapture with more board variety / better lighting")

    print("[3/6] Per-camera intrinsics calibration...")
    left_corners = [d[0] for d in left_data if d[0] is not None]
    left_ids = [d[1] for d in left_data if d[1] is not None]
    right_corners = [d[0] for d in right_data if d[0] is not None]
    right_ids = [d[1] for d in right_data if d[1] is not None]
    rms_l, K_l, dist_l = calibrate_camera(board, left_corners, left_ids, image_size)
    rms_r, K_r, dist_r = calibrate_camera(board, right_corners, right_ids, image_size)
    print(f"      cam1 (left)  reprojection rms: {rms_l:.3f} px, fx={K_l[0,0]:.1f}")
    print(f"      cam0 (right) reprojection rms: {rms_r:.3f} px, fx={K_r[0,0]:.1f}")

    print("[4/6] Stereo extrinsics (FIX_INTRINSIC)...")
    obj_pts, l_pts, r_pts = paired_object_image_points(board, left_data, right_data)
    print(f"      {len(obj_pts)} paired views feeding stereoCalibrate")
    rms_s, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, l_pts, r_pts,
        K_l, dist_l, K_r, dist_r, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    angle = angle_from_R(R)
    baseline = float(np.linalg.norm(T))
    print(f"      stereo reprojection rms: {rms_s:.3f} px")
    print(f"      recovered angle between cameras: {angle:.2f} deg (rig spec ~24)")
    print(f"      recovered baseline (cam separation): {baseline*1000:.1f} mm")

    print("[5/6] Rectification...")
    # alpha=1 keeps every source pixel in the rectified output (with black
    # padding where rectification has no source). alpha=0 would crop to the
    # largest all-valid rectangle, which on this rig clipped the top of the
    # FOV - exactly where hands held high need to be visible.
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, dist_l, K_r, dist_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1,
    )

    # Use one of the calibration frames as the sanity image.
    sanity_idx = next((i for i, (l, r) in enumerate(zip(left_data, right_data))
                       if l[0] is not None and r[0] is not None), 0)
    sbs, maps = render_sanity(
        pairs[sanity_idx][0], pairs[sanity_idx][1],
        K_l, dist_l, K_r, dist_r, R1, R2, P1, P2, image_size,
    )
    cv2.imwrite(OUT_SANITY, sbs)
    print(f"      sanity image -> {OUT_SANITY}")

    print("[6/6] Saving calibration...")
    np.savez(
        OUT_NPZ,
        image_size=np.array(image_size),
        K_left=K_l, dist_left=dist_l,
        K_right=K_r, dist_right=dist_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map_left_x=maps[0], map_left_y=maps[1],
        map_right_x=maps[2], map_right_y=maps[3],
        rms_left=np.array(rms_l), rms_right=np.array(rms_r), rms_stereo=np.array(rms_s),
        angle_deg=np.array(angle), baseline_m=np.array(baseline),
        square_m=np.array(SQUARE_M), marker_m=np.array(MARKER_M),
    )
    print(f"      -> {OUT_NPZ}")
    print()
    print("Verdict:")
    print(f"  per-camera rms: {rms_l:.2f} / {rms_r:.2f} px  (good < 0.5, ok < 1)")
    print(f"  stereo rms:     {rms_s:.2f} px           (good < 1, ok < 1.5)")
    print(f"  angle:          {angle:.1f} deg          (rig spec ~24)")
    print(f"Open {OUT_SANITY} - same physical points must lie on the same green scanline.")


if __name__ == "__main__":
    main()
