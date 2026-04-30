"""Stage 8 / Phase 4 prep: per-frame table-frame anchoring from a single ArUco marker.

Detects DICT_4X4_50 ID 0 in both views per frame, solves PnP with the marker's
known physical size, and saves T_world_to_table per frame. Also cross-validates
by detecting in the right view independently and applying the stereo extrinsic —
the two paths (left-view PnP vs right-view PnP composed with stereo R, T) should
give the same T_world_to_table to within sub-centimetre / sub-degree.

Coordinate frames:
  origLeft     — physical left camera frame (where left K and dist apply)
  origRight    — physical right camera frame
  world        — rectified-left frame; same as the canonical pipeline's world
  table        — marker frame: origin at the marker's centre, +Z out of the
                 marker (toward the camera), +X along the marker's first edge

solvePnP returns (R_pnp, t_pnp) such that X_origCam = R_pnp · X_marker + t_pnp
(transform FROM marker frame TO camera frame). So t_pnp is the marker
origin's position in the camera frame.

The canonical anchor we save is T_world_to_table (= T_rectifiedLeft_to_table).
Inverse-direction transform: a point in world goes to table via
    X_table = R_pnp_l.T · (X_world_in_origLeft - t_pnp_l)
where X_world_in_origLeft = R1.T · X_world.

Cross-view check (left-PnP vs right-PnP via stereo):
    t_origRight_marker = R_stereo · t_origLeft_marker + T_stereo
    R_pnp_r            = R_stereo · R_pnp_l
i.e., predicting left-view from right-view:
    t_l_predicted = R_stereo.T · (t_pnp_r - T_stereo)
    R_l_predicted = R_stereo.T · R_pnp_r

Output:
  outputs/<date> - table anchor [<tag>].mp4   SBS, marker detection + 10 cm axes
  outputs/<date> - table anchor [<tag>].npz   per-frame T_world_to_table

Run:
  .venv/bin/python scripts/pipeline/09_anchor_table.py
  .venv/bin/python scripts/pipeline/09_anchor_table.py --marker-mm 77.5
  .venv/bin/python scripts/pipeline/09_anchor_table.py --max-frames 60
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LEFT = PROJECT_ROOT / "raw/cam1_20260430_160751.mp4"
DEFAULT_RIGHT = PROJECT_ROOT / "raw/cam0_20260430_160751.mp4"
DEFAULT_CALIB = PROJECT_ROOT / "outputs/30th April 2026 wide - stereo calibration.npz"
DEFAULT_TAG = "20260430 protocol-A"
DEFAULT_MARKER_MM = 77.5
DEFAULT_DICT = aruco.DICT_4X4_50
DEFAULT_ID = 0


def from_root(p):
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_calib(path):
    c = np.load(str(path))
    return {
        "K_l": c["K_left"], "dist_l": c["dist_left"],
        "K_r": c["K_right"], "dist_r": c["dist_right"],
        "R": c["R"], "T": c["T"],
        "R1": c["R1"],
        "image_size": tuple(int(v) for v in c["image_size"]),
        "baseline_m": float(c["baseline_m"]),
    }


def marker_object_points(side_m):
    """4 corners of the marker in its own frame, ordered the same way as
    cv2.aruco.detectMarkers returns them (top-left, top-right, bottom-right,
    bottom-left when looking AT the marker from the camera).
    Centre at origin, +Z out of the marker, +X to the right, +Y down (image-style)."""
    s = side_m / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float64)


def detect_and_solutions(frame_bgr, K, dist, detector, obj_pts, target_id):
    """Returns (sols, corners_2d) or None.

    `sols` is a list of (R, t, reproj_err) tuples — typically TWO for
    IPPE_SQUARE on a planar marker (front-tilted vs back-tilted). The 4
    corners of a square viewed in 2D have a fundamental two-fold pose
    ambiguity, and any single-view PnP solver can only break it via
    reprojection-error tie-breaking, which is unreliable when the marker is
    near head-on. We return BOTH and let the caller pick the cross-view-
    consistent pair using the stereo extrinsic.

    R, t in each solution express T_origCam_to_marker:  X_marker = R·X_cam + t.
    """
    corners, ids, _ = detector.detectMarkers(frame_bgr)
    if ids is None:
        return None
    for i, mid in enumerate(ids.flatten()):
        if int(mid) != target_id:
            continue
        img_pts = corners[i].reshape(-1, 2).astype(np.float64)
        retval, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj_pts.reshape(-1, 1, 3),
            img_pts.reshape(-1, 1, 2),
            K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if retval == 0:
            continue
        sols = []
        for j in range(retval):
            R, _ = cv2.Rodrigues(rvecs[j])
            sols.append((R, tvecs[j].flatten(),
                         float(errs[j].item()) if errs is not None else 0.0))
        return sols, img_pts
    return None


def predict_left_from_right(R_pnp_r, t_pnp_r, R_stereo, T_stereo):
    """Given a right-view PnP and the stereo extrinsic, predict what the
    left-view PnP should be.

    solvePnP returns (R, t) with X_cam = R · X_marker + t, so t is the
    marker origin in the camera frame. The marker origin is the same
    physical point in both views; the stereo extrinsic relates the two:
        t_pnp_r = R_stereo · t_pnp_l + T_stereo
    Inverting:
        t_pnp_l = R_stereo.T · (t_pnp_r - T_stereo)
    Likewise R_pnp_r = R_stereo · R_pnp_l, so
        R_pnp_l = R_stereo.T · R_pnp_r.
    """
    T_s = T_stereo.flatten()
    R_l_pred = R_stereo.T @ R_pnp_r
    t_l_pred = R_stereo.T @ (t_pnp_r - T_s)
    return R_l_pred, t_l_pred


def pick_consistent_pair(sols_l, sols_r, R_stereo, T_stereo, marker_mm=80.0):
    """Given two PnP solution lists (one per view) and the stereo extrinsic,
    pick the (left_sol, right_sol) combination most cross-view-consistent.

    For IPPE_SQUARE the two solutions per view differ mostly in rotation
    (the marker centre is identical). Picking by translation residual alone
    leaves the rotation ambiguity unresolved — corner displacements of a
    rotation tilt are second-order and small in pixels. We combine both:

        score = t_residual_mm + (marker_mm/2) * sin(r_residual_rad)

    The (marker_mm/2)·sin(θ) term is the in-plane corner displacement
    that a tilt of θ around the marker centre would produce, in mm. So
    translation and rotation are weighed in commensurate millimetres.

    Returns (R_l, t_l, R_r, t_r, t_residual_mm, r_residual_deg).
    """
    best = (None, None, None, None, float("inf"), float("inf"), float("inf"))
    half = marker_mm / 2.0
    for R_l, t_l, _ in sols_l:
        for R_r, t_r, _ in sols_r:
            R_l_pred, t_l_pred = predict_left_from_right(R_r, t_r, R_stereo, T_stereo)
            t_res_mm = float(np.linalg.norm(t_l_pred - t_l)) * 1000.0
            r_res_deg = rot_angle_between(R_l, R_l_pred)
            score = t_res_mm + half * abs(np.sin(np.radians(r_res_deg)))
            if score < best[4]:
                best = (R_l, t_l, R_r, t_r, score, t_res_mm, r_res_deg)
    return best[0], best[1], best[2], best[3], best[5], best[6]


def rot_angle_between(R1, R2):
    """Rotation angle (deg) between two 3x3 rotations."""
    M = R1.T @ R2
    cos_theta = np.clip((np.trace(M) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_theta)))


def annotate(frame, K, dist, R_to_marker, t_to_marker, corners_2d,
             axis_len_m, color, label):
    """Draw marker contour + 3-axis frame, plus a small text label."""
    cv2.polylines(frame, [corners_2d.astype(np.int32)], True, color, 2, cv2.LINE_AA)
    rvec, _ = cv2.Rodrigues(R_to_marker)
    cv2.drawFrameAxes(frame, K, dist, rvec, t_to_marker.reshape(3, 1), axis_len_m, 3)
    x0, y0 = corners_2d[0].astype(int)
    cv2.putText(frame, label, (x0, max(0, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-left", default=str(DEFAULT_LEFT))
    ap.add_argument("--clip-right", default=str(DEFAULT_RIGHT))
    ap.add_argument("--calib", default=str(DEFAULT_CALIB))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_MM,
                    help="Measured side length of the printed marker, mm. "
                         "Pass the value you measured with a ruler — 1 mm error "
                         "is ~1%% scale error in the table frame.")
    ap.add_argument("--marker-id", type=int, default=DEFAULT_ID)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    clip_l = from_root(args.clip_left)
    clip_r = from_root(args.clip_right)
    calib_path = from_root(args.calib)
    if not clip_l.is_file() or not clip_r.is_file():
        sys.exit(f"clip not found: {clip_l} or {clip_r}")
    if not calib_path.is_file():
        sys.exit(f"calibration not found: {calib_path}")

    out_video = PROJECT_ROOT / f"outputs/{today_pretty()} - table anchor [{args.tag}].mp4"
    out_npz = PROJECT_ROOT / f"outputs/{today_pretty()} - table anchor [{args.tag}].npz"

    calib = load_calib(calib_path)
    side_m = args.marker_mm / 1000.0
    obj_pts = marker_object_points(side_m)

    print(f"calib:    {calib_path.name}")
    print(f"          baseline {calib['baseline_m']*1000:.1f} mm, image {calib['image_size']}")
    print(f"left  =   {clip_l.name}")
    print(f"right =   {clip_r.name}")
    print(f"marker:   {args.marker_mm:.1f} mm, DICT_4X4_50 id={args.marker_id}")

    aruco_dict = aruco.getPredefinedDictionary(DEFAULT_DICT)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

    cap_l = cv2.VideoCapture(str(clip_l))
    cap_r = cv2.VideoCapture(str(clip_r))
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT), cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.max_frames:
        n = min(n, args.max_frames)
    w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"frames:   {n} @ {fps:.1f} fps, {w}x{h} per camera")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w * 2, h))

    # NaN-padded; per-frame.
    R_world_to_table = np.full((n, 3, 3), np.nan, dtype=np.float32)
    t_world_to_table = np.full((n, 3),    np.nan, dtype=np.float32)
    R_pnp_left      = np.full((n, 3, 3), np.nan, dtype=np.float32)
    t_pnp_left      = np.full((n, 3),    np.nan, dtype=np.float32)
    R_pnp_right     = np.full((n, 3, 3), np.nan, dtype=np.float32)
    t_pnp_right     = np.full((n, 3),    np.nan, dtype=np.float32)
    cross_view_t_mm = np.full((n,),      np.nan, dtype=np.float32)
    cross_view_r_deg = np.full((n,),     np.nan, dtype=np.float32)
    detected_left   = np.zeros((n,), dtype=bool)
    detected_right  = np.zeros((n,), dtype=bool)

    n_l = n_r = n_both = 0
    t_start = time.time()
    R_stereo = calib["R"]
    T_stereo = calib["T"]
    for i in range(n):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break

        left = detect_and_solutions(fl, calib["K_l"], calib["dist_l"],
                                     detector, obj_pts, args.marker_id)
        right = detect_and_solutions(fr, calib["K_r"], calib["dist_r"],
                                      detector, obj_pts, args.marker_id)

        # If both views see the marker, disambiguate IPPE_SQUARE's two
        # solutions by picking the cross-view-consistent pair. If only one
        # sees it, fall back to the lowest-reprojection-error solution.
        if left is not None and right is not None:
            sols_l, corners_l = left
            sols_r, corners_r = right
            R_l, t_l, R_r, t_r, t_res_mm, r_res_deg = pick_consistent_pair(
                sols_l, sols_r, R_stereo, T_stereo, args.marker_mm)
            n_both += 1
            cross_view_t_mm[i] = t_res_mm
            cross_view_r_deg[i] = r_res_deg
        elif left is not None:
            sols_l, corners_l = left
            R_l, t_l, _ = min(sols_l, key=lambda s: s[2])
            R_r = t_r = corners_r = None
        elif right is not None:
            sols_r, corners_r = right
            R_r, t_r, _ = min(sols_r, key=lambda s: s[2])
            R_l = t_l = corners_l = None
        else:
            R_l = t_l = corners_l = R_r = t_r = corners_r = None

        if R_l is not None:
            n_l += 1
            detected_left[i] = True
            R_pnp_left[i] = R_l
            t_pnp_left[i] = t_l
            # Canonical anchor as a point-transform: X_table = R · X_world + t.
            # solvePnP gives X_origLeft = R_l · X_marker + t_l, and rectification
            # gives X_origLeft = R1.T · X_world. Solving for X_marker:
            #   X_marker = R_l.T · (R1.T · X_world - t_l)
            #            = (R_l.T · R1.T) · X_world + (-R_l.T · t_l)
            R_world_to_table[i] = R_l.T @ calib["R1"].T
            t_world_to_table[i] = -R_l.T @ t_l
            annotate(fl, calib["K_l"], calib["dist_l"], R_l, t_l, corners_l,
                     axis_len_m=0.10, color=(0, 220, 255),
                     label=f"L  d={np.linalg.norm(t_l)*100:.1f}cm")

        if R_r is not None:
            n_r += 1
            detected_right[i] = True
            R_pnp_right[i] = R_r
            t_pnp_right[i] = t_r
            annotate(fr, calib["K_r"], calib["dist_r"], R_r, t_r, corners_r,
                     axis_len_m=0.10, color=(255, 100, 0),
                     label=f"R  d={np.linalg.norm(t_r)*100:.1f}cm")

        # HUD
        hud = []
        hud.append(f"frame {i+1}/{n}")
        if left is not None:
            hud.append(f"L ok ({np.linalg.norm(t_l)*100:.1f} cm)")
        else:
            hud.append("L --")
        if right is not None:
            hud.append(f"R ok ({np.linalg.norm(t_r)*100:.1f} cm)")
        else:
            hud.append("R --")
        if left is not None and right is not None:
            hud.append(f"cross dt={cross_view_t_mm[i]:.1f}mm dr={cross_view_r_deg[i]:.2f}deg")
        text = "  ".join(hud)
        sbs = np.hstack([fl, fr])
        cv2.putText(sbs, text, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)
        writer.write(sbs)

        if i % 30 == 0 or i == n - 1:
            sys.stdout.write(f"\r{i+1}/{n}  L={n_l} R={n_r} both={n_both}   ")
            sys.stdout.flush()
    print()

    cap_l.release()
    cap_r.release()
    writer.release()

    np.savez(
        out_npz,
        R_world_to_table=R_world_to_table,
        t_world_to_table=t_world_to_table,
        R_pnp_left=R_pnp_left,
        t_pnp_left=t_pnp_left,
        R_pnp_right=R_pnp_right,
        t_pnp_right=t_pnp_right,
        cross_view_t_mm=cross_view_t_mm,
        cross_view_r_deg=cross_view_r_deg,
        detected_left=detected_left,
        detected_right=detected_right,
        marker_mm=np.array(args.marker_mm),
        marker_id=np.array(args.marker_id),
        fps=np.array(fps),
        image_size=np.array([w, h]),
        calib_path=str(calib_path),
    )

    # Summary
    print(f"\n--- detection ---")
    print(f"left:  {n_l}/{n} ({100*n_l/max(n,1):.1f}%)")
    print(f"right: {n_r}/{n} ({100*n_r/max(n,1):.1f}%)")
    print(f"both:  {n_both}/{n} ({100*n_both/max(n,1):.1f}%)")

    valid = ~np.isnan(t_pnp_left[:, 2])
    if valid.any():
        d = np.linalg.norm(t_pnp_left[valid], axis=1) * 100
        print(f"\n--- distance camera -> marker (left view, cm) ---")
        print(f"median {np.median(d):.1f}, 5-95 [{np.percentile(d,5):.1f}, "
              f"{np.percentile(d,95):.1f}], min {d.min():.1f}, max {d.max():.1f}")

    valid = ~np.isnan(cross_view_t_mm)
    if valid.any():
        ct = cross_view_t_mm[valid]
        cr = cross_view_r_deg[valid]
        print(f"\n--- cross-view residual (left PnP vs right PnP via stereo) ---")
        print(f"translation (mm):  median {np.median(ct):.2f}, "
              f"5-95 [{np.percentile(ct,5):.2f}, {np.percentile(ct,95):.2f}], max {ct.max():.2f}")
        print(f"rotation (deg):    median {np.median(cr):.3f}, "
              f"5-95 [{np.percentile(cr,5):.3f}, {np.percentile(cr,95):.3f}], max {cr.max():.3f}")
        # Sanity bar: a rigid marker + correct stereo calibration should give
        # cross-view translation residual well below 5 mm and rotation residual
        # below ~0.5 deg. Larger numbers point to either a bad measured
        # marker_mm or a stereo-calibration issue.

    print(f"\nvideo -> {out_video.name}")
    print(f"npz   -> {out_npz.name}")
    print(f"wall:  {time.time()-t_start:.1f}s")

    if sys.platform == "darwin":
        subprocess.run(["open", str(out_video)], check=False)


if __name__ == "__main__":
    main()
