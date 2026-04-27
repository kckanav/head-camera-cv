"""Stage 8 / Phase 3 (full): stereo MANO mesh fusion.

Replaces the per-view monocular meshes from `wilor_ar_overlay.py` with ONE
mesh per matched-hand pair, fit so WiLoR's predicted 21 keypoints (in MANO
local frame) match the 21 stereo-triangulated 3D keypoints (in the
rectified-left camera frame). The same world-frame mesh is then projected
into BOTH unrectified views - direct proof that the stereo rig is now
driving the mesh, not just the wrist.

Pipeline per frame pair:
  1. YOLO + WiLoR per view -> per hand: pred_vertices (778, 3 in MANO local
     frame), pred_keypoints_3d (21, 3 same frame), pred_keypoints_2d in image
     pixels.
  2. Match left/right hand detections by rectified wrist row (existing logic
     from `wilor_stereo_demo.py`).
  3. Phase 3.1 - Triangulate all 21 keypoints per matched pair with
     cv2.triangulatePoints(P1, P2, kp_l_rect, kp_r_rect). Per-keypoint
     validity = (rectified row diff < ROW_TOL_KP_PX) & (Z plausible).
  4. Phase 3.2 - Weighted Umeyama: closed-form (s, R, t) that maps WiLoR's
     local 3D keypoints onto the triangulated keypoints. Down-weight
     fingertips (1 px disparity is huge depth error at our 41 mm baseline);
     up-weight palm joints + wrist. Apply (s, R, t) to all 778 vertices ->
     world-frame mesh.
  5. Phase 3.3 - Project the SAME world-frame mesh into both UNRECTIFIED
     views via R1.T (undo rectification) -> calibration K + dist for each
     camera. Render with painter's algorithm (back-to-front by Z in render
     camera frame, Lambertian shading). Both views show two views of the
     same physical mesh.

Output:
  outputs/<date> - phase3 fused [<tag>].npz
  outputs/<date> - phase3 ar overlay [<tag>].mp4

Run (.venv-hamer Python only - WiLoR isn't installed in .venv):
  .venv-hamer/bin/python scripts/wilor_phase3.py
  .venv-hamer/bin/python scripts/wilor_phase3.py --max-frames 10   # smoke
  .venv-hamer/bin/python scripts/wilor_phase3.py --no-mesh         # skeleton-only debug
  .venv-hamer/bin/python scripts/wilor_phase3.py --calib "outputs/27th April 2026 - stereo calibration.npz"
"""

import argparse
import os
import sys
import time
import types
from pathlib import Path


# --- Workaround 1: pyrender on macOS (same stubs as wilor_stereo_demo.py) -----
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
ROW_TOL_PX_WRIST = 30   # liberal for wrist matching (gates whether to even pair hands)
ROW_TOL_PX_KP = 8.0     # tighter for per-keypoint epipolar validity
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

FINGERTIP_INDICES = (4, 8, 12, 16, 20)
PALM_INDICES = (5, 9, 13, 17)


# --- WiLoR helpers (mirror wilor_stereo_demo.py / wilor_ar_overlay.py) --------
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
    """YOLO + WiLoR per view. Returns list of dicts. Includes pred_keypoints_3d
    needed for Umeyama fusion."""
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
        kp2d_arr = out["pred_keypoints_2d"].detach().cpu().numpy()      # (B,21,2) crop coords
        verts_arr = out["pred_vertices"].detach().cpu().numpy()          # (B,778,3) MANO local
        kp3d_arr = out["pred_keypoints_3d"].detach().cpu().numpy()       # (B,21,3) MANO local
        for n in range(batch["img"].shape[0]):
            is_r = batch["right"][n].item() > 0.5
            handed = "right" if is_r else "left"
            box_center = batch["box_center"][n].cpu().numpy()
            box_size = batch["box_size"][n].cpu().item()

            kp_crop = kp2d_arr[n].copy()
            v = verts_arr[n].copy()
            kp3 = kp3d_arr[n].copy()
            # WiLoR's ViTDetDataset flips left-hand crops; un-flip X on every X-bearing tensor.
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
    """Match by rectified wrist y. Returns list of (h_l, h_r) tuples; either
    side can be None for unmatched hands."""
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


# --- Phase 3.1: full keypoint triangulation -----------------------------------
def triangulate_full_keypoints(h_l, h_r, calib):
    """Triangulate all 21 keypoints in the matched pair. Returns
    (kp_3d_world (21, 3), valid (21,) bool) in the rectified-left frame."""
    rect_l = rectify_keypoints(h_l["kp2d_full"], calib["K_l"], calib["dist_l"],
                                calib["R1"], calib["P1"])
    rect_r = rectify_keypoints(h_r["kp2d_full"], calib["K_r"], calib["dist_r"],
                                calib["R2"], calib["P2"])
    pts4 = cv2.triangulatePoints(
        calib["P1"], calib["P2"],
        rect_l.T.astype(np.float64),
        rect_r.T.astype(np.float64),
    )
    pts3d = (pts4[:3] / pts4[3]).T.astype(np.float64)   # (21, 3)
    row_diff = np.abs(rect_l[:, 1] - rect_r[:, 1])
    valid = row_diff < ROW_TOL_PX_KP
    valid &= (pts3d[:, 2] > Z_MIN_M) & (pts3d[:, 2] < Z_MAX_M)
    return pts3d, valid


# --- Phase 3.2: Umeyama mesh fusion -------------------------------------------
def umeyama_alignment(X, Y, weights=None):
    """Closed-form similarity transform Y_i ≈ s·R·X_i + t (Umeyama 1991, weighted).

    X, Y: (N, 3) point sets. Returns (s, R, t).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        raise ValueError("zero total weight in Umeyama")

    mu_x = (w[:, None] * X).sum(axis=0) / w_sum
    mu_y = (w[:, None] * Y).sum(axis=0) / w_sum
    Xc = X - mu_x
    Yc = Y - mu_y

    Sigma = (w[:, None] * Yc).T @ Xc / w_sum    # 3x3 cross-covariance
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    var_x = (w[:, None] * (Xc * Xc).sum(axis=1, keepdims=True)).sum() / w_sum
    s = float(np.trace(np.diag(D) @ S) / var_x) if var_x > 1e-12 else 1.0
    t = mu_y - s * R @ mu_x
    return s, R, t


def fuse_to_world(h, kp_3d_world, valid):
    """Apply weighted Umeyama to align WiLoR's local keypoints to the triangulated
    world-frame keypoints. Returns dict with placement + world vertices, or None
    if too few valid keypoints to fit."""
    if valid.sum() < 5:
        return None

    weights = np.ones(21)
    weights[list(FINGERTIP_INDICES)] = 0.4   # fingertip Z is noisy at our baseline
    weights[list(PALM_INDICES)] = 1.5        # palm MCPs are reliable
    weights[WRIST] = 2.0                      # wrist most reliable
    weights = weights * valid.astype(np.float64)

    s, R, t = umeyama_alignment(h["kp3d_local"], kp_3d_world, weights=weights)

    aligned_kp = (s * (R @ h["kp3d_local"].T)).T + t
    world_verts = (s * (R @ h["verts_local"].T)).T + t

    residual_per_kp = np.linalg.norm(aligned_kp - kp_3d_world, axis=1)
    residual_mean_m = float(np.mean(residual_per_kp[valid])) if valid.any() else float("nan")

    return {
        "s": float(s), "R": R, "t": t,
        "world_verts": world_verts.astype(np.float32),
        "world_kp": aligned_kp.astype(np.float32),
        "residual_m": residual_mean_m,
    }


# --- Phase 3.3: projection + render -------------------------------------------
def project_to_left(world_pts, calib):
    """world (= rectified-left) -> ORIGINAL-left image pixels.
    Returns (pts2d (N, 2), pts_in_orig_left (N, 3) for sorting/shading)."""
    pts_orig_left = (calib["R1"].T @ np.asarray(world_pts).T).T
    pts2d, _ = cv2.projectPoints(
        pts_orig_left.astype(np.float64),
        np.zeros(3), np.zeros(3),
        calib["K_l"], calib["dist_l"],
    )
    return pts2d.reshape(-1, 2), pts_orig_left


def project_to_right(world_pts, calib):
    """world (= rectified-left) -> ORIGINAL-right image pixels.
    Path: rect-left -> orig-left (via R1.T) -> orig-right (via stereo R, T)."""
    pts_orig_left = (calib["R1"].T @ np.asarray(world_pts).T).T
    pts_orig_right = (calib["R"] @ pts_orig_left.T).T + calib["T"].flatten()
    pts2d, _ = cv2.projectPoints(
        pts_orig_right.astype(np.float64),
        np.zeros(3), np.zeros(3),
        calib["K_r"], calib["dist_r"],
    )
    return pts2d.reshape(-1, 2), pts_orig_right


def render_mesh(img, verts2d, verts3d_camera, faces,
                base_color=(255, 110, 30), edge_color=(220, 80, 20),
                alpha=0.55, draw_edges=True):
    """Painter's algorithm + Lambertian. verts3d_camera must be in the rendering
    camera's frame so back-to-front sorting is correct."""
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


def annotate_label(img, hand, color, depth_cm=None, residual_mm=None, kp_used=None):
    x0, y0, _, _ = hand["bbox"].astype(int)
    label = hand["handed"]
    if depth_cm is not None:
        label += f"  z={depth_cm:5.1f}cm"
    if residual_mm is not None:
        label += f"  fit={residual_mm:.1f}mm"
    if kp_used is not None:
        label += f"  kp={kp_used}/21"
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
    ap.add_argument("--alpha", type=float, default=0.55, help="mesh overlay opacity")
    ap.add_argument("--no-mesh", action="store_true",
                    help="skip mesh raster; skeleton + bbox + label only (debug)")
    ap.add_argument("--no-edges", action="store_true", help="skip drawing triangle edges")
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

    out_video = PROJECT_ROOT / f"outputs/{today_pretty()} - phase3 ar overlay [{args.tag}].mp4"
    out_npz = PROJECT_ROOT / f"outputs/{today_pretty()} - phase3 fused [{args.tag}].npz"

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
    verts_3d_world = np.full((n, NSLOT, 778, 3), np.nan, dtype=np.float32)
    kp_3d_world = np.full((n, NSLOT, 21, 3), np.nan, dtype=np.float32)
    kp_valid_arr = np.zeros((n, NSLOT, 21), dtype=bool)
    placement_s = np.full((n, NSLOT), np.nan, dtype=np.float32)
    placement_R = np.full((n, NSLOT, 3, 3), np.nan, dtype=np.float32)
    placement_t = np.full((n, NSLOT, 3), np.nan, dtype=np.float32)
    residual_m = np.full((n, NSLOT), np.nan, dtype=np.float32)
    reproj_l_arr = np.full((n, NSLOT), np.nan, dtype=np.float32)
    reproj_r_arr = np.full((n, NSLOT), np.nan, dtype=np.float32)
    handedness = np.full((n, NSLOT), "", dtype=object)

    n_pairs = 0
    n_fused = 0
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

        # Sort by left wrist X for slot stability across frames.
        def sort_key(p):
            if p[0] is not None:
                return p[0]["kp2d_full"][WRIST, 0]
            return p[1]["kp2d_full"][WRIST, 0] + 1e9
        pairs.sort(key=sort_key)

        for slot, (h_l, h_r) in enumerate(pairs[:NSLOT]):
            if h_l is None or h_r is None:
                # Unmatched - skeleton only, no fusion.
                gray = (60, 60, 60)
                if h_l is not None:
                    draw_skeleton(fl, h_l["kp2d_full"], gray)
                    annotate_label(fl, h_l, gray)
                if h_r is not None:
                    draw_skeleton(fr, h_r["kp2d_full"], gray)
                    annotate_label(fr, h_r, gray)
                continue
            n_pairs += 1

            tf0 = time.time()
            # 3.1
            kp3d_w, valid = triangulate_full_keypoints(h_l, h_r, calib)
            # 3.2
            fused = fuse_to_world(h_l, kp3d_w, valid)
            fusion_time += time.time() - tf0

            color = [(0, 200, 255), (255, 100, 0)][slot]

            kp_3d_world[i, slot] = np.where(valid[:, None], kp3d_w, np.nan)
            kp_valid_arr[i, slot] = valid
            handedness[i, slot] = h_l["handed"]

            if fused is None:
                # Pair matched but not enough valid keypoints to fit. Show skeleton.
                draw_skeleton(fl, h_l["kp2d_full"], color)
                draw_skeleton(fr, h_r["kp2d_full"], color)
                annotate_label(fl, h_l, color, kp_used=int(valid.sum()))
                annotate_label(fr, h_r, color, kp_used=int(valid.sum()))
                continue

            n_fused += 1
            verts_3d_world[i, slot] = fused["world_verts"]
            placement_s[i, slot] = fused["s"]
            placement_R[i, slot] = fused["R"]
            placement_t[i, slot] = fused["t"]
            residual_m[i, slot] = fused["residual_m"]

            # 3.3
            tr0 = time.time()
            v2_l, v3_l = project_to_left(fused["world_verts"], calib)
            v2_r, v3_r = project_to_right(fused["world_verts"], calib)
            kp2_l_proj, _ = project_to_left(fused["world_kp"], calib)
            kp2_r_proj, _ = project_to_right(fused["world_kp"], calib)

            reproj_l = np.linalg.norm(kp2_l_proj - h_l["kp2d_full"], axis=1)
            reproj_r = np.linalg.norm(kp2_r_proj - h_r["kp2d_full"], axis=1)
            reproj_l_arr[i, slot] = float(np.median(reproj_l[valid])) if valid.any() else float("nan")
            reproj_r_arr[i, slot] = float(np.median(reproj_r[valid])) if valid.any() else float("nan")

            if not args.no_mesh:
                fl = render_mesh(fl, v2_l, v3_l, faces,
                                 alpha=args.alpha, draw_edges=not args.no_edges)
                fr = render_mesh(fr, v2_r, v3_r, faces,
                                 alpha=args.alpha, draw_edges=not args.no_edges)
            draw_skeleton(fl, h_l["kp2d_full"], color)
            draw_skeleton(fr, h_r["kp2d_full"], color)
            wrist_z_cm = float(fused["t"][2] * 100.0)
            annotate_label(fl, h_l, color, depth_cm=wrist_z_cm,
                           residual_mm=fused["residual_m"] * 1000.0, kp_used=int(valid.sum()))
            annotate_label(fr, h_r, color, depth_cm=wrist_z_cm,
                           residual_mm=fused["residual_m"] * 1000.0, kp_used=int(valid.sum()))
            render_time += time.time() - tr0

        sbs = np.hstack([fl, fr])
        writer.write(sbs)
        if i % 5 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (n - i - 1) / rate if rate else 0
            sys.stdout.write(f"\rframe {i+1}/{n}  pairs={n_pairs} fused={n_fused}  "
                             f"{rate:.2f} fps  eta {eta:.0f}s   ")
            sys.stdout.flush()
    print()

    cap_l.release()
    cap_r.release()
    writer.release()

    np.savez(
        out_npz,
        verts_3d_world=verts_3d_world,
        kp_3d_world=kp_3d_world,
        kp_valid=kp_valid_arr,
        placement_s=placement_s,
        placement_R=placement_R,
        placement_t=placement_t,
        umeyama_residual_m=residual_m,
        reproj_err_left_px=reproj_l_arr,
        reproj_err_right_px=reproj_r_arr,
        handedness=np.array(handedness.tolist()),
        fps=np.array(fps),
        image_size=np.array([w, h]),
        calib_path=str(calib_path),
    )

    # Summary stats
    print(f"\npairs detected: {n_pairs}, fused: {n_fused}")
    valid_res = residual_m[~np.isnan(residual_m)] * 1000.0
    if valid_res.size:
        print(f"umeyama residual (mm):  median {np.median(valid_res):.1f}, "
              f"5-95 [{np.percentile(valid_res,5):.1f}, {np.percentile(valid_res,95):.1f}]")
    valid_l = reproj_l_arr[~np.isnan(reproj_l_arr)]
    valid_r = reproj_r_arr[~np.isnan(reproj_r_arr)]
    if valid_l.size:
        print(f"reproj err left (px):   median {np.median(valid_l):.1f}, "
              f"5-95 [{np.percentile(valid_l,5):.1f}, {np.percentile(valid_l,95):.1f}]")
    if valid_r.size:
        print(f"reproj err right (px):  median {np.median(valid_r):.1f}, "
              f"5-95 [{np.percentile(valid_r,5):.1f}, {np.percentile(valid_r,95):.1f}]")
    valid_z = placement_t[..., 2][~np.isnan(placement_t[..., 2])] * 100.0
    if valid_z.size:
        print(f"wrist depth (cm):       median {np.median(valid_z):.1f}, "
              f"5-95 [{np.percentile(valid_z,5):.1f}, {np.percentile(valid_z,95):.1f}]")
    valid_s = placement_s[~np.isnan(placement_s)]
    if valid_s.size:
        print(f"placement scale s:      median {np.median(valid_s):.3f}, "
              f"5-95 [{np.percentile(valid_s,5):.3f}, {np.percentile(valid_s,95):.3f}]")
    print(f"inference: {inference_time:.1f}s   fusion: {fusion_time:.1f}s   "
          f"render: {render_time:.1f}s   wall: {time.time()-t0:.1f}s")
    print(f"video -> {out_video.name}")
    print(f"npz   -> {out_npz.name}")

    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(out_video)], check=False)


if __name__ == "__main__":
    main()
