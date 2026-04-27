"""Stage 8 / Phase 3 (anchored): monocular pose + stereo metric depth.

Where Phase 3 (full) re-fits the MANO mesh as a global rigid transform in 3D
(7 DOF Umeyama, losing per-view 2D accuracy in the process), this script
keeps WiLoR's monocular fit untouched and uses stereo for the ONE thing it
can give that monocular cannot: absolute metric scale.

Key insight: uniform 3D scaling preserves perspective 2D projection.
A point at (X, Y, Z) projects to (fX/Z, fY/Z); scale to (kX, kY, kZ) and
the projection is (fkX/kZ, fkY/kZ) = unchanged. So we can change the
mesh's metric size to match stereo without breaking WiLoR's 2D accuracy.

Algorithm per matched-hand pair per frame:
  1. WiLoR per view -> pred_vertices (778, 3 in MANO local frame),
     pred_keypoints_3d (21, 3 same frame, root-centered),
     pred_keypoints_2d_full (21, 2 in real-image pixels).
  2. cv2.solvePnP per view (SQPNP) -> (R_view, t_view) in the REAL camera
     frame, such that K · (R_view·X_local + t_view) projects to WiLoR's
     2D keypoints. t_view[2] is WiLoR's monocular wrist depth estimate.
  3. Stereo triangulate the wrist in the rectified-left frame, then
     transform to each real camera frame to get Z_stereo_wrist_left and
     Z_stereo_wrist_right.
  4. Scalar scale anchor:  k_view = Z_stereo_wrist_view / t_view[2]
     (1 DOF, just a depth ratio).
  5. Metric mesh in each real camera frame:
        mesh_metric_view = k_view · (R_view · verts_local + t_view)
     Project through (K_view, dist_view) to render — by construction of
     PnP + scaling, the 2D projection matches WiLoR's monocular 2D.

For the saved 3D dataset: store the LEFT-view metric mesh transformed to
world (rectified-left) frame as canonical. Also save PnP placements,
per-view scale, wrist 3D, and PnP residuals so downstream analysis can
inspect any per-view disagreement.

Output:
  outputs/<date> - phase3 anchored ar overlay [<tag>].mp4
  outputs/<date> - phase3 anchored fused [<tag>].npz

Run (.venv-hamer Python only):
  .venv-hamer/bin/python scripts/wilor_phase3_anchored.py
  .venv-hamer/bin/python scripts/wilor_phase3_anchored.py --max-frames 10
  .venv-hamer/bin/python scripts/wilor_phase3_anchored.py --no-mesh
"""

import argparse
import os
import sys
import time
import types
from pathlib import Path


# --- Workaround 1: pyrender on macOS (same stubs as siblings) -----------------
class _NoopRenderer:
    def __init__(self, *a, **kw):
        pass

    def __getattr__(self, name):
        raise RuntimeError(f"renderer stubbed; nothing should call {name!r}")


def _cam_crop_to_full(*a, **kw):
    raise RuntimeError("cam_crop_to_full stubbed")


for _mod_name, _attrs in [
    ("wilor.utils.renderer",          {"Renderer": _NoopRenderer, "cam_crop_to_full": _cam_crop_to_full}),
    ("wilor.utils.mesh_renderer",     {"MeshRenderer": _NoopRenderer}),
    ("wilor.utils.skeleton_renderer", {"SkeletonRenderer": _NoopRenderer}),
]:
    _stub = types.ModuleType(_mod_name)
    for _k, _v in _attrs.items():
        setattr(_stub, _k, _v)
    sys.modules[_mod_name] = _stub

# --- Workaround 2: PyTorch 2.6 weights_only default ---------------------------
import torch

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

import cv2
import numpy as np
from ultralytics import YOLO

# --- Path setup ---------------------------------------------------------------
from dated import today_pretty   # scripts/dated.py — siblings on sys.path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WILOR_DIR = PROJECT_ROOT / "wilor"

if not WILOR_DIR.is_dir():
    sys.exit("missing wilor/ - run Phase 0 first; see PLAN.md")

sys.path.insert(0, str(WILOR_DIR))
os.chdir(WILOR_DIR)

from wilor.models import load_wilor                                   # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset                # noqa: E402

# --- Defaults: the wide 10-s interaction clip --------------------------------
DEFAULT_LEFT = PROJECT_ROOT / "inputs/27th April 2026 wide - cam1 interaction first 10s.mp4"
DEFAULT_RIGHT = PROJECT_ROOT / "inputs/27th April 2026 wide - cam0 interaction first 10s.mp4"
DEFAULT_CALIB = PROJECT_ROOT / "outputs/27th April 2026 wide - stereo calibration.npz"
DEFAULT_TAG = "wide first 10s interaction"

WRIST = 0
ROW_TOL_PX_WRIST = 30
Z_MIN_M = 0.05
Z_MAX_M = 2.00

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# --- WiLoR helpers (mirror of wilor_phase3.py) --------------------------------
def to_device(obj, device):
    if torch.is_tensor(obj):
        if obj.dtype == torch.float64:
            obj = obj.to(torch.float32)
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def load_calib(path):
    c = np.load(path)
    return {
        "K_l": c["K_left"], "dist_l": c["dist_left"],
        "K_r": c["K_right"], "dist_r": c["dist_right"],
        "R": c["R"], "T": c["T"],
        "R1": c["R1"], "R2": c["R2"],
        "P1": c["P1"], "P2": c["P2"],
        "image_size": tuple(int(v) for v in c["image_size"]),
        "baseline_m": float(c["baseline_m"]),
    }


def detect_and_regress(detector, model, model_cfg, img, device, batch_size=8):
    detections = detector(img, device="cpu", conf=0.3, verbose=False)[0]
    if len(detections) == 0:
        return []
    bboxes, is_right_list = [], []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())
    boxes = np.stack(bboxes)
    right = np.stack(is_right_list)
    dataset = ViTDetDataset(model_cfg, img, boxes, right, rescale_factor=2.0, fp16=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    hands = []
    idx = 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.no_grad():
            out = model(batch)
        kp2d_arr = out["pred_keypoints_2d"].detach().cpu().numpy()
        verts_arr = out["pred_vertices"].detach().cpu().numpy()
        kp3d_arr = out["pred_keypoints_3d"].detach().cpu().numpy()
        for n in range(batch["img"].shape[0]):
            is_r = batch["right"][n].item() > 0.5
            handed = "right" if is_r else "left"
            box_center = batch["box_center"][n].cpu().numpy()
            box_size = batch["box_size"][n].cpu().item()

            kp_crop = kp2d_arr[n].copy()
            v = verts_arr[n].copy()
            kp3 = kp3d_arr[n].copy()
            if not is_r:
                kp_crop[:, 0] *= -1
                v[:, 0] *= -1
                kp3[:, 0] *= -1

            kp_full = np.empty_like(kp_crop)
            kp_full[:, 0] = box_center[0] + kp_crop[:, 0] * box_size
            kp_full[:, 1] = box_center[1] + kp_crop[:, 1] * box_size

            hands.append({
                "kp2d_full": kp_full,
                "kp3d_local": kp3,
                "verts_local": v,
                "handed": handed,
                "bbox": boxes[idx],
                "box_center": box_center,
                "box_size": box_size,
            })
            idx += 1
    return hands


def rectify_keypoints(kp2d_full, K, dist, R_rect, P_rect):
    pts = kp2d_full.astype(np.float32).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, dist, R=R_rect, P=P_rect)
    return out.reshape(-1, 2)


def match_stereo_pairs(hands_l, hands_r, calib):
    if not hands_l or not hands_r:
        return [(h, None) for h in hands_l] + [(None, h) for h in hands_r]
    wrist_l = np.array([
        rectify_keypoints(h["kp2d_full"][WRIST:WRIST+1], calib["K_l"], calib["dist_l"],
                          calib["R1"], calib["P1"])[0]
        for h in hands_l
    ])
    wrist_r = np.array([
        rectify_keypoints(h["kp2d_full"][WRIST:WRIST+1], calib["K_r"], calib["dist_r"],
                          calib["R2"], calib["P2"])[0]
        for h in hands_r
    ])
    pairs = []
    used_r = set()
    for li, h_l in enumerate(hands_l):
        ly = wrist_l[li, 1]
        best_ri, best_dy = None, ROW_TOL_PX_WRIST
        for ri in range(len(hands_r)):
            if ri in used_r:
                continue
            dy = abs(wrist_r[ri, 1] - ly)
            if dy < best_dy:
                best_ri, best_dy = ri, dy
        if best_ri is None:
            pairs.append((h_l, None))
            continue
        used_r.add(best_ri)
        pairs.append((h_l, hands_r[best_ri]))
    for ri in range(len(hands_r)):
        if ri not in used_r:
            pairs.append((None, hands_r[ri]))
    return pairs


def triangulate_wrist_world(h_l, h_r, calib):
    """Stereo triangulate the wrist; return 3D position in rectified-left (world)
    frame, or None if epipolar/Z checks fail."""
    rect_l = rectify_keypoints(h_l["kp2d_full"][WRIST:WRIST+1], calib["K_l"], calib["dist_l"],
                                calib["R1"], calib["P1"])
    rect_r = rectify_keypoints(h_r["kp2d_full"][WRIST:WRIST+1], calib["K_r"], calib["dist_r"],
                                calib["R2"], calib["P2"])
    if abs(rect_l[0, 1] - rect_r[0, 1]) > ROW_TOL_PX_WRIST:
        return None
    pts4 = cv2.triangulatePoints(
        calib["P1"], calib["P2"],
        rect_l.T.astype(np.float64),
        rect_r.T.astype(np.float64),
    )
    p3d = (pts4[:3] / pts4[3]).flatten()
    if not (Z_MIN_M <= p3d[2] <= Z_MAX_M):
        return None
    return p3d


# --- PnP + anchor scale -------------------------------------------------------
def solve_pnp(kp3d_local, kp2d_full, K, dist):
    """SQPNP: find (R, t) such that K·(R·kp3d_local + t) -> kp2d_full.
    Returns (R 3x3, t 3, residual_px) or None on failure."""
    obj_pts = kp3d_local.astype(np.float64).reshape(-1, 1, 3)
    img_pts = kp2d_full.astype(np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # Reprojection residual
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    residual_px = float(np.median(np.linalg.norm(proj - kp2d_full, axis=1)))
    return R, tvec.flatten(), residual_px


def transform_world_to_left_cam(p_world, calib):
    """world (= rectified-left) -> original-left camera frame."""
    return calib["R1"].T @ np.asarray(p_world)


def transform_world_to_right_cam(p_world, calib):
    """world (= rectified-left) -> original-right camera frame."""
    p_orig_left = calib["R1"].T @ np.asarray(p_world)
    return calib["R"] @ p_orig_left + calib["T"].flatten()


# --- Rendering ----------------------------------------------------------------
def render_mesh(img, verts2d, verts3d_camera, faces,
                base_color=(255, 110, 30), edge_color=(220, 80, 20),
                alpha=0.55, draw_edges=True):
    """Painter's algorithm + Lambertian. verts3d_camera in the rendering camera's frame."""
    overlay = img.copy()
    pts2 = verts2d[faces].astype(np.int32)
    tri_3d = verts3d_camera[faces]
    face_z = tri_3d.mean(axis=1)[:, 2]
    order = np.argsort(-face_z)

    e1 = tri_3d[:, 1] - tri_3d[:, 0]
    e2 = tri_3d[:, 2] - tri_3d[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    n_hat = normals / norms
    light = np.array([0.0, -0.4, -1.0])
    light /= np.linalg.norm(light)
    shade = np.clip(np.abs(n_hat @ light), 0.30, 1.0)
    base = np.array(base_color, dtype=np.float32)

    h, w = img.shape[:2]
    for fi in order:
        p = pts2[fi]
        if p[:, 0].max() < 0 or p[:, 1].max() < 0 or p[:, 0].min() >= w or p[:, 1].min() >= h:
            continue
        col = (base * shade[fi]).astype(np.uint8)
        cv2.fillConvexPoly(overlay, p, (int(col[0]), int(col[1]), int(col[2])), lineType=cv2.LINE_AA)
        if draw_edges:
            cv2.polylines(overlay, [p], True, edge_color, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)


def draw_skeleton(img, kp_xy, color):
    pts = [(int(x), int(y)) for x, y in kp_xy]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 5, color, 1, cv2.LINE_AA)


def annotate_label(img, hand, color, depth_cm=None, k=None, pnp_px=None):
    x0, y0, _, _ = hand["bbox"].astype(int)
    label = hand["handed"]
    if depth_cm is not None:
        label += f"  z={depth_cm:5.1f}cm"
    if k is not None:
        label += f"  k={k:.3f}"
    if pnp_px is not None:
        label += f"  pnp={pnp_px:.1f}px"
    cv2.putText(img, label, (x0, max(0, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def get_mano_faces(model):
    for attr in ("mano", "mano_layer", "smpl", "body_model"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "faces"):
            return np.asarray(m.faces, dtype=np.int32)
    import pickle
    with open(WILOR_DIR / "mano_data" / "MANO_RIGHT.pkl", "rb") as f:
        mano_right = pickle.load(f, encoding="latin1")
    return np.asarray(mano_right["f"], dtype=np.int32)


# --- Main pipeline ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-left", default=str(DEFAULT_LEFT))
    ap.add_argument("--clip-right", default=str(DEFAULT_RIGHT))
    ap.add_argument("--calib", default=str(DEFAULT_CALIB))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--no-mesh", action="store_true",
                    help="skip mesh raster; skeleton + label only")
    ap.add_argument("--no-edges", action="store_true")
    args = ap.parse_args()

    def from_root(p):
        pp = Path(p)
        return pp if pp.is_absolute() else (PROJECT_ROOT / pp).resolve()

    clip_l = from_root(args.clip_left)
    clip_r = from_root(args.clip_right)
    calib_path = from_root(args.calib)
    if not clip_l.is_file() or not clip_r.is_file():
        sys.exit(f"clip not found: {clip_l} or {clip_r}")
    if not calib_path.is_file():
        sys.exit(f"calibration not found: {calib_path}")

    out_video = PROJECT_ROOT / f"outputs/{today_pretty()} - phase3 anchored ar overlay [{args.tag}].mp4"
    out_npz = PROJECT_ROOT / f"outputs/{today_pretty()} - phase3 anchored fused [{args.tag}].npz"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    print(f"left:   {clip_l.name}")
    print(f"right:  {clip_r.name}")
    print(f"calib:  {calib_path.name}")

    calib = load_calib(calib_path)
    print(f"calib baseline {calib['baseline_m']*1000:.1f} mm, image {calib['image_size']}")

    print("loading WiLoR ...")
    t_load = time.time()
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    model = model.to(device).eval()
    print(f"loaded in {time.time() - t_load:.1f}s")

    detector = YOLO("./pretrained_models/detector.pt")
    faces = get_mano_faces(model)
    print(f"MANO faces: {faces.shape}")

    cap_l = cv2.VideoCapture(str(clip_l))
    cap_r = cv2.VideoCapture(str(clip_r))
    fps = cap_l.get(cv2.CAP_PROP_FPS)
    n = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT), cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.max_frames:
        n = min(n, args.max_frames)
    w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{n} frames @ {fps:.1f} fps, {w}x{h} per camera")

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))

    NSLOT = 2
    # Canonical: left-view metric mesh transformed to world frame.
    verts_3d_world = np.full((n, NSLOT, 778, 3), np.nan, dtype=np.float32)
    wrist_3d_world = np.full((n, NSLOT, 3), np.nan, dtype=np.float32)
    pose_R_left = np.full((n, NSLOT, 3, 3), np.nan, dtype=np.float32)
    pose_t_left = np.full((n, NSLOT, 3), np.nan, dtype=np.float32)
    pose_R_right = np.full((n, NSLOT, 3, 3), np.nan, dtype=np.float32)
    pose_t_right = np.full((n, NSLOT, 3), np.nan, dtype=np.float32)
    scale_k_left = np.full((n, NSLOT), np.nan, dtype=np.float32)
    scale_k_right = np.full((n, NSLOT), np.nan, dtype=np.float32)
    pnp_residual_left_px = np.full((n, NSLOT), np.nan, dtype=np.float32)
    pnp_residual_right_px = np.full((n, NSLOT), np.nan, dtype=np.float32)
    handedness = np.full((n, NSLOT), "", dtype=object)

    n_pairs = 0
    n_anchored = 0
    inference_time = 0.0
    fusion_time = 0.0
    render_time = 0.0
    t0 = time.time()
    for i in range(n):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break

        ti0 = time.time()
        hands_l = detect_and_regress(detector, model, model_cfg, fl, device)
        hands_r = detect_and_regress(detector, model, model_cfg, fr, device)
        inference_time += time.time() - ti0

        pairs = match_stereo_pairs(hands_l, hands_r, calib)

        def sort_key(p):
            if p[0] is not None:
                return p[0]["kp2d_full"][WRIST, 0]
            return p[1]["kp2d_full"][WRIST, 0] + 1e9
        pairs.sort(key=sort_key)

        for slot, (h_l, h_r) in enumerate(pairs[:NSLOT]):
            color = [(0, 200, 255), (255, 100, 0)][slot]

            if h_l is None or h_r is None:
                gray = (60, 60, 60)
                if h_l is not None:
                    draw_skeleton(fl, h_l["kp2d_full"], gray)
                    annotate_label(fl, h_l, gray)
                if h_r is not None:
                    draw_skeleton(fr, h_r["kp2d_full"], gray)
                    annotate_label(fr, h_r, gray)
                continue
            n_pairs += 1
            handedness[i, slot] = h_l["handed"]

            tf0 = time.time()
            wrist_world = triangulate_wrist_world(h_l, h_r, calib)
            pnp_l = solve_pnp(h_l["kp3d_local"], h_l["kp2d_full"], calib["K_l"], calib["dist_l"])
            pnp_r = solve_pnp(h_r["kp3d_local"], h_r["kp2d_full"], calib["K_r"], calib["dist_r"])
            fusion_time += time.time() - tf0

            if wrist_world is None or pnp_l is None or pnp_r is None:
                # Skeleton only.
                draw_skeleton(fl, h_l["kp2d_full"], color)
                draw_skeleton(fr, h_r["kp2d_full"], color)
                annotate_label(fl, h_l, color)
                annotate_label(fr, h_r, color)
                continue

            R_l, t_l, res_l = pnp_l
            R_r, t_r, res_r = pnp_r
            wrist_in_l_cam = transform_world_to_left_cam(wrist_world, calib)
            wrist_in_r_cam = transform_world_to_right_cam(wrist_world, calib)

            # 1D scale anchor: depth ratio. Guard divide-by-zero.
            if abs(t_l[2]) < 1e-6 or abs(t_r[2]) < 1e-6:
                continue
            k_l = float(wrist_in_l_cam[2] / t_l[2])
            k_r = float(wrist_in_r_cam[2] / t_r[2])

            # Metric mesh in each real camera frame.
            mesh_l_cam = k_l * ((R_l @ h_l["verts_local"].T).T + t_l)
            mesh_r_cam = k_r * ((R_r @ h_r["verts_local"].T).T + t_r)

            # Save canonical world-frame mesh derived from LEFT view.
            mesh_world_from_l = (calib["R1"] @ mesh_l_cam.T).T
            verts_3d_world[i, slot] = mesh_world_from_l.astype(np.float32)
            wrist_3d_world[i, slot] = wrist_world.astype(np.float32)
            pose_R_left[i, slot] = R_l
            pose_t_left[i, slot] = t_l
            pose_R_right[i, slot] = R_r
            pose_t_right[i, slot] = t_r
            scale_k_left[i, slot] = k_l
            scale_k_right[i, slot] = k_r
            pnp_residual_left_px[i, slot] = res_l
            pnp_residual_right_px[i, slot] = res_r
            n_anchored += 1

            # Project metric meshes through real cameras for rendering.
            tr0 = time.time()
            v2_l, _ = cv2.projectPoints(
                mesh_l_cam.astype(np.float64), np.zeros(3), np.zeros(3),
                calib["K_l"], calib["dist_l"],
            )
            v2_r, _ = cv2.projectPoints(
                mesh_r_cam.astype(np.float64), np.zeros(3), np.zeros(3),
                calib["K_r"], calib["dist_r"],
            )
            v2_l = v2_l.reshape(-1, 2)
            v2_r = v2_r.reshape(-1, 2)

            if not args.no_mesh:
                fl = render_mesh(fl, v2_l, mesh_l_cam, faces,
                                 alpha=args.alpha, draw_edges=not args.no_edges)
                fr = render_mesh(fr, v2_r, mesh_r_cam, faces,
                                 alpha=args.alpha, draw_edges=not args.no_edges)
            draw_skeleton(fl, h_l["kp2d_full"], color)
            draw_skeleton(fr, h_r["kp2d_full"], color)
            depth_cm = float(wrist_world[2] * 100.0)
            annotate_label(fl, h_l, color, depth_cm=depth_cm, k=k_l, pnp_px=res_l)
            annotate_label(fr, h_r, color, depth_cm=depth_cm, k=k_r, pnp_px=res_r)
            render_time += time.time() - tr0

        sbs = np.hstack([fl, fr])
        writer.write(sbs)
        if i % 5 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (n - i - 1) / rate if rate else 0
            sys.stdout.write(f"\rframe {i+1}/{n}  pairs={n_pairs} anchored={n_anchored}  "
                             f"{rate:.2f} fps  eta {eta:.0f}s   ")
            sys.stdout.flush()
    print()

    cap_l.release()
    cap_r.release()
    writer.release()

    np.savez(
        out_npz,
        verts_3d_world=verts_3d_world,
        wrist_3d_world=wrist_3d_world,
        pose_R_left=pose_R_left,
        pose_t_left=pose_t_left,
        pose_R_right=pose_R_right,
        pose_t_right=pose_t_right,
        scale_k_left=scale_k_left,
        scale_k_right=scale_k_right,
        pnp_residual_left_px=pnp_residual_left_px,
        pnp_residual_right_px=pnp_residual_right_px,
        handedness=np.array(handedness.tolist()),
        fps=np.array(fps),
        image_size=np.array([w, h]),
        calib_path=str(calib_path),
    )

    # Summary
    print(f"\npairs detected: {n_pairs}, anchored: {n_anchored}")
    valid_res_l = pnp_residual_left_px[~np.isnan(pnp_residual_left_px)]
    valid_res_r = pnp_residual_right_px[~np.isnan(pnp_residual_right_px)]
    if valid_res_l.size:
        print(f"pnp residual left  (px):  median {np.median(valid_res_l):.2f}, "
              f"5-95 [{np.percentile(valid_res_l,5):.2f}, {np.percentile(valid_res_l,95):.2f}]")
    if valid_res_r.size:
        print(f"pnp residual right (px):  median {np.median(valid_res_r):.2f}, "
              f"5-95 [{np.percentile(valid_res_r,5):.2f}, {np.percentile(valid_res_r,95):.2f}]")
    valid_z = wrist_3d_world[..., 2][~np.isnan(wrist_3d_world[..., 2])] * 100.0
    if valid_z.size:
        print(f"wrist depth (cm):         median {np.median(valid_z):.1f}, "
              f"5-95 [{np.percentile(valid_z,5):.1f}, {np.percentile(valid_z,95):.1f}]")
    valid_kl = scale_k_left[~np.isnan(scale_k_left)]
    valid_kr = scale_k_right[~np.isnan(scale_k_right)]
    if valid_kl.size:
        print(f"scale k left:             median {np.median(valid_kl):.3f}, "
              f"5-95 [{np.percentile(valid_kl,5):.3f}, {np.percentile(valid_kl,95):.3f}]")
    if valid_kr.size:
        print(f"scale k right:            median {np.median(valid_kr):.3f}, "
              f"5-95 [{np.percentile(valid_kr,5):.3f}, {np.percentile(valid_kr,95):.3f}]")
    print(f"inference: {inference_time:.1f}s   fusion: {fusion_time:.1f}s   "
          f"render: {render_time:.1f}s   wall: {time.time()-t0:.1f}s")
    print(f"video -> {out_video.name}")
    print(f"npz   -> {out_npz.name}")

    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(out_video)], check=False)


if __name__ == "__main__":
    main()
