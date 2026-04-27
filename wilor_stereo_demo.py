"""Stage 8 / Phase 2 + minimal Phase 3: WiLoR per view + stereo wrist triangulation.

This is the smallest demo that *actually uses* both cameras with WiLoR. We run
WiLoR independently on each view's raw frames, then triangulate the wrist
landmark using the existing stereo calibration to recover the **metric**
wrist depth - which WiLoR alone cannot give us (it's monocular and
scale-ambiguous).

Why wrist only (for now):
  Full Phase 3 fuses *all 21* landmarks plus the mesh scale across views.
  That's planned for the next iteration. The wrist alone is enough to
  *demonstrate* that stereo gives us metric depth on top of WiLoR's mesh.

Outputs:
  outputs/<date> - wilor stereo demo.mp4    side-by-side annotated video
  outputs/<date> - wilor stereo demo.npz    per-frame data dump

Run:
  .venv-hamer/bin/python wilor_stereo_demo.py            # default 15-s clip
  .venv-hamer/bin/python wilor_stereo_demo.py --long     # 60-s clip
"""

import argparse
import os
import sys
import time
import types
from pathlib import Path


# --- Workaround 1: pyrender on macOS ------------------------------------------
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

# --- Path setup (same logic as wilor_sanity.py; see comments there) -----------
PROJECT_ROOT = Path(__file__).resolve().parent
WILOR_DIR = PROJECT_ROOT / "wilor"
DEFAULT_CALIB = PROJECT_ROOT / "outputs/27th April 2026 - stereo calibration.npz"

if not WILOR_DIR.is_dir():
    sys.exit("missing wilor/ - run Phase 0 (env + WiLoR clone) first; see PLAN.md")

sys.path.insert(0, str(PROJECT_ROOT))
from dated import today_pretty   # noqa: E402

sys.path = [p for p in sys.path if Path(p).resolve() != PROJECT_ROOT.resolve()]
sys.path.insert(0, str(WILOR_DIR))
os.chdir(WILOR_DIR)

from wilor.models import load_wilor                                   # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset                # noqa: E402

# --- Constants ----------------------------------------------------------------
WRIST = 0       # MANO/MediaPipe landmark index for wrist
ROW_TOL_PX = 30 # max wrist y-difference (rectified) to count as a stereo match
Z_MIN_M = 0.05  # 5 cm: hands genuinely get this close on a head-mount when manipulating
Z_MAX_M = 2.00

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# --- Helpers ------------------------------------------------------------------
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
        "R1": c["R1"], "R2": c["R2"], "P1": c["P1"], "P2": c["P2"],
        "baseline_m": float(c["baseline_m"]),
    }


def raw_to_rectified(pts_xy, K, dist, R, P):
    """Map raw image pixel coords to rectified pixel coords."""
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, dist, R=R, P=P)
    return out.reshape(-1, 2)


def detect_and_regress(detector, model, img, device, batch_size=8):
    """YOLO detect (CPU) -> WiLoR regress (device). Returns list of dicts."""
    detections = detector(img, device="cpu", conf=0.3, verbose=False)[0]
    if len(detections) == 0:
        return []
    bboxes, is_right = [], []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())
    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    dataset = ViTDetDataset(model.cfg if hasattr(model, "cfg") else _MODEL_CFG,
                            img, boxes, right, rescale_factor=2.0, fp16=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)
    hands = []
    idx = 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.no_grad():
            out = model(batch)
        kp2d = out["pred_keypoints_2d"].detach().cpu().numpy()       # (B,21,2) crop coords
        verts = out["pred_vertices"].detach().cpu().numpy()           # (B,778,3) MANO local
        for n in range(batch["img"].shape[0]):
            is_right = batch["right"][n].item() > 0.5
            handed = "right" if is_right else "left"
            box_center = batch["box_center"][n].cpu().numpy()
            box_size = batch["box_size"][n].cpu().item()

            # ViTDetDataset flips left-hand crops horizontally before
            # inference, so pred_keypoints_2d come out in flipped-crop
            # coords. Un-flip X to map back to the original image.
            kp_crop = kp2d[n].copy()
            v = verts[n].copy()
            if not is_right:
                kp_crop[:, 0] *= -1
                v[:, 0] *= -1   # mirror MANO mesh X to match left-hand anatomy

            kp_full = np.empty_like(kp_crop)
            kp_full[:, 0] = box_center[0] + kp_crop[:, 0] * box_size
            kp_full[:, 1] = box_center[1] + kp_crop[:, 1] * box_size
            hands.append({
                "kp2d_full": kp_full,
                "verts": v,
                "handed": handed,
                "bbox": boxes[idx],
            })
            idx += 1
    return hands


def match_stereo(hands_l, hands_r, calib):
    """For each left hand, project its wrist into rectified space, then find
    the right hand whose rectified wrist y is closest within ROW_TOL_PX.
    Returns list of (left_hand, right_hand, paired_wrist_3d_metric or None)."""
    if not hands_l or not hands_r:
        return []
    wrist_l_rect = np.array([
        raw_to_rectified(h["kp2d_full"][WRIST:WRIST+1],
                         calib["K_l"], calib["dist_l"], calib["R1"], calib["P1"])[0]
        for h in hands_l
    ])
    wrist_r_rect = np.array([
        raw_to_rectified(h["kp2d_full"][WRIST:WRIST+1],
                         calib["K_r"], calib["dist_r"], calib["R2"], calib["P2"])[0]
        for h in hands_r
    ])
    pairs = []
    used_r = set()
    for li, h_l in enumerate(hands_l):
        ly = wrist_l_rect[li, 1]
        best_ri, best_dy = None, ROW_TOL_PX
        for ri in range(len(hands_r)):
            if ri in used_r:
                continue
            dy = abs(wrist_r_rect[ri, 1] - ly)
            if dy < best_dy:
                best_ri, best_dy = ri, dy
        if best_ri is None:
            pairs.append((h_l, None, None))
            continue
        used_r.add(best_ri)
        pts4 = cv2.triangulatePoints(
            calib["P1"], calib["P2"],
            wrist_l_rect[li].reshape(2, 1).astype(np.float64),
            wrist_r_rect[best_ri].reshape(2, 1).astype(np.float64),
        )
        wrist3d = (pts4[:3] / pts4[3]).flatten()
        if not (Z_MIN_M <= wrist3d[2] <= Z_MAX_M):
            pairs.append((h_l, hands_r[best_ri], None))
        else:
            pairs.append((h_l, hands_r[best_ri], wrist3d))
    for ri in range(len(hands_r)):
        if ri not in used_r:
            pairs.append((None, hands_r[ri], None))
    return pairs


def draw_skeleton(img, kp_xy, color):
    pts = [(int(x), int(y)) for x, y in kp_xy]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 5, color, 1)


def annotate(img, hand, color, depth_cm=None, paired=True):
    draw_skeleton(img, hand["kp2d_full"], color)
    x0, y0, x1, y1 = hand["bbox"].astype(int)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    label = hand["handed"]
    if depth_cm is not None:
        label += f"  z={depth_cm:5.1f} cm (stereo)"
    elif paired:
        label += "  z= ?  (stereo paired but rejected)"
    else:
        label += "  z= ?  (no stereo match)"
    cv2.putText(img, label, (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", action="store_true",
                        help="default to the 60-s 'first minute' clip instead of the 15-s grease clip")
    parser.add_argument("--clip-left", type=str, default=None,
                        help="override left clip path (cam1)")
    parser.add_argument("--clip-right", type=str, default=None,
                        help="override right clip path (cam0)")
    parser.add_argument("--calib", type=str, default=None,
                        help="override stereo calibration .npz path")
    parser.add_argument("--tag", type=str, default=None,
                        help="output filename tag; default is derived from the clip preset")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after N frames (useful for quick previews)")
    args = parser.parse_args()

    # Defaults from preset flags
    if args.long:
        clip_l = PROJECT_ROOT / "inputs/25th April 2026 - cam1 first minute.mp4"
        clip_r = PROJECT_ROOT / "inputs/25th April 2026 - cam0 first minute.mp4"
        out_tag = "60s first minute"
    else:
        clip_l = PROJECT_ROOT / "inputs/25th April 2026 - cam1 clip 1m40-1m55.mp4"
        clip_r = PROJECT_ROOT / "inputs/25th April 2026 - cam0 clip 1m40-1m55.mp4"
        out_tag = "15s grease"

    # CLI paths are interpreted relative to PROJECT_ROOT (not cwd, which by
    # this point is wilor/ - we chdir'd into it for picamera2 config paths).
    def from_root(p):
        pp = Path(p)
        return pp if pp.is_absolute() else (PROJECT_ROOT / pp).resolve()

    if args.clip_left:
        clip_l = from_root(args.clip_left)
    if args.clip_right:
        clip_r = from_root(args.clip_right)
    if args.tag:
        out_tag = args.tag

    calib_path = from_root(args.calib) if args.calib else DEFAULT_CALIB
    if not calib_path.is_file():
        sys.exit(f"calibration not found: {calib_path}")
    if not clip_l.is_file() or not clip_r.is_file():
        sys.exit(f"clip not found: {clip_l} or {clip_r}")

    out_video = PROJECT_ROOT / f"outputs/{today_pretty()} - wilor stereo demo {out_tag}.mp4"
    out_npz = PROJECT_ROOT / f"outputs/{today_pretty()} - wilor stereo demo {out_tag}.npz"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}   clip: {clip_l.name} / {clip_r.name}")
    print(f"calibration: {calib_path.name}")

    calib = load_calib(calib_path)
    print(f"calibration baseline {calib['baseline_m']*1000:.1f} mm")

    print("loading WiLoR ...")
    t0 = time.time()
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    model = model.to(device).eval()
    global _MODEL_CFG
    _MODEL_CFG = model_cfg
    print(f"loaded in {time.time()-t0:.1f}s")

    detector = YOLO("./pretrained_models/detector.pt")

    cap_l = cv2.VideoCapture(str(clip_l))
    cap_r = cv2.VideoCapture(str(clip_r))
    fps = cap_l.get(cv2.CAP_PROP_FPS)
    n = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT),
                cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.max_frames:
        n = min(n, args.max_frames)
    w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{n} frames @ {fps:.1f} fps, {w}x{h} per camera")

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w * 2, h))

    # Per-frame dump (NaN-padded for missing data)
    wrist_3d = np.full((n, 2, 3), np.nan, dtype=np.float32)        # (frames, slot, xyz) metric
    kp2d_l   = np.full((n, 2, 21, 2), np.nan, dtype=np.float32)
    kp2d_r   = np.full((n, 2, 21, 2), np.nan, dtype=np.float32)
    handed_per_frame = np.full((n, 2), "", dtype=object)

    n_l_only, n_r_only, n_paired, n_paired_metric = 0, 0, 0, 0
    t0 = time.time()
    inference_time = 0.0
    for i in range(n):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break

        ti0 = time.time()
        hands_l = detect_and_regress(detector, model, fl, device)
        hands_r = detect_and_regress(detector, model, fr, device)
        inference_time += time.time() - ti0

        if hands_l and not hands_r: n_l_only += 1
        if hands_r and not hands_l: n_r_only += 1

        pairs = match_stereo(hands_l, hands_r, calib)

        # Sort pairs by left wrist X for slot stability (left/right hand goes
        # to the slot whose wrist x is smallest in the left view).
        def sort_key(p):
            if p[0] is not None:
                return p[0]["kp2d_full"][WRIST, 0]
            return p[1]["kp2d_full"][WRIST, 0] + 1e9  # unmatched right hands sort last
        pairs.sort(key=sort_key)

        for slot, (h_l, h_r, w3d) in enumerate(pairs[:2]):
            if h_l is not None and h_r is not None:
                n_paired += 1
                if w3d is not None:
                    n_paired_metric += 1
            color = [(0, 200, 255), (255, 100, 0)][slot]
            depth_cm = w3d[2] * 100.0 if w3d is not None else None
            paired = (h_l is not None) and (h_r is not None)
            if h_l is not None:
                annotate(fl, h_l, color, depth_cm=depth_cm, paired=paired)
                kp2d_l[i, slot] = h_l["kp2d_full"]
                handed_per_frame[i, slot] = h_l["handed"]
            if h_r is not None:
                annotate(fr, h_r, color, depth_cm=depth_cm, paired=paired)
                kp2d_r[i, slot] = h_r["kp2d_full"]
                if not handed_per_frame[i, slot]:
                    handed_per_frame[i, slot] = h_r["handed"]
            if w3d is not None:
                wrist_3d[i, slot] = w3d

        sbs = np.hstack([fl, fr])
        writer.write(sbs)
        if i % 10 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (n - i - 1) / rate if rate else 0
            sys.stdout.write(f"\rframe {i+1}/{n}  paired-metric {n_paired_metric}  "
                             f"{rate:.2f} fps  eta {eta:.0f}s   ")
            sys.stdout.flush()
    print()

    cap_l.release()
    cap_r.release()
    writer.release()

    np.savez(
        out_npz,
        wrist_3d_metric=wrist_3d,
        keypoints_2d_left=kp2d_l,
        keypoints_2d_right=kp2d_r,
        handedness=np.array(handed_per_frame.tolist()),
        fps=np.array(fps),
        image_size=np.array([w, h]),
    )

    print(f"\ntotal frames: {n}")
    print(f"left-only / right-only / paired / paired-with-metric:  "
          f"{n_l_only} / {n_r_only} / {n_paired} / {n_paired_metric}")
    valid = wrist_3d[..., 2][~np.isnan(wrist_3d[..., 2])] * 100.0
    if valid.size:
        print(f"wrist depth (cm): median {np.median(valid):.1f}, "
              f"5-95 [{np.percentile(valid,5):.1f}, {np.percentile(valid,95):.1f}], "
              f"min {valid.min():.1f}, max {valid.max():.1f}")
    print(f"inference time: {inference_time:.1f}s ({inference_time/max(n,1):.2f}s/frame)")
    print(f"wall time:      {time.time()-t0:.1f}s")
    print(f"video -> {out_video.name}")
    print(f"npz   -> {out_npz.name}")

    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(out_video)], check=False)


if __name__ == "__main__":
    main()
