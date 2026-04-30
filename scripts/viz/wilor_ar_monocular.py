"""Stage 8 / AR mesh overlay: project WiLoR's MANO mesh onto each view as a
translucent 'digital glove' so we can visually verify mesh accuracy.

Pipeline per frame, per view:
  1. YOLO detects hand bboxes (CPU; ultralytics MPS bug for Pose models).
  2. WiLoR regresses each crop on MPS -> per hand:
       pred_vertices       (778, 3)  MANO local frame
       pred_keypoints_3d   (21, 3)   same frame
       pred_keypoints_2d   (21, 2)   crop coords (in [-0.5, 0.5])
  3. Fit weak-perspective (s, tx, ty) by least-squares from
     pred_keypoints_3d.xy -> pred_keypoints_2d. WiLoR uses one camera for
     both keypoints and vertices, so the same affine projects vertices to
     crop coords. Then map crop -> image via (box_center + crop * box_size).
  4. Painter's algorithm: sort the 1538 MANO faces back-to-front by mean 3D Z,
     fill each with Lambert-shaded blue, alpha-blend the overlay onto the
     frame. Front-facing triangles end up over-painting back-facing ones.
  5. Left-hand inference is run on a horizontally-flipped crop by WiLoR's
     ViTDetDataset, so we mirror X on the network's keypoints AND vertices
     AND 3D keypoints before projection (same fix as wilor_stereo_demo.py).

Output:
  outputs/<date> - wilor ar overlay [<tag>].mp4    SBS (or single view if --left-only)

Run (.venv-hamer Python only — WiLoR isn't installed in .venv):
  .venv-hamer/bin/python scripts/wilor_ar_overlay.py
  .venv-hamer/bin/python scripts/wilor_ar_overlay.py --max-frames 20  # quick smoke
  .venv-hamer/bin/python scripts/wilor_ar_overlay.py --left-only      # half the work
  .venv-hamer/bin/python scripts/wilor_ar_overlay.py --no-edges --alpha 0.7
"""

import argparse
import sys
import time
from pathlib import Path

# _lib bootstrap — must come BEFORE importing torch / wilor.* so wilor_setup's
# pyrender stubs and torch.load patch take effect.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import wilor_setup  # noqa: F401, E402  side-effects: stubs + torch.load patch + sys.path + chdir
from wilor_setup import PROJECT_ROOT, WILOR_DIR, get_mano_faces           # noqa: E402

import torch                                                             # noqa: E402
import cv2                                                               # noqa: E402
import numpy as np                                                       # noqa: E402
from ultralytics import YOLO                                             # noqa: E402

from dated import today_pretty                                           # noqa: E402
from device import (pick_device, configure_perf, to_device_safe,         # noqa: E402
                    autocast_ctx)

from wilor.models import load_wilor                                      # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset                  # noqa: E402

# --- Defaults: the wide 10-s interaction clip --------------------------------
DEFAULT_LEFT = PROJECT_ROOT / "inputs/27th April 2026 wide - cam1 interaction first 10s.mp4"
DEFAULT_RIGHT = PROJECT_ROOT / "inputs/27th April 2026 wide - cam0 interaction first 10s.mp4"
DEFAULT_TAG = "wide first 10s interaction"


# --- Helpers ------------------------------------------------------------------
def detect_and_regress(detector, model, model_cfg, img, device, cfg, batch_size=8):
    """YOLO -> WiLoR. Returns list of dicts with all the fields needed for projection.

    `cfg` is the perf config from configure_perf(device): controls YOLO device,
    fp16 input, dataloader workers/pinned memory, and ViT autocast dtype.
    """
    detections = detector(img, device=cfg["yolo_device"], conf=0.3, verbose=False)[0]
    if len(detections) == 0:
        return []
    bboxes, is_right_list = [], []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())
    boxes = np.stack(bboxes)
    right = np.stack(is_right_list)
    dataset = ViTDetDataset(model_cfg, img, boxes, right, rescale_factor=2.0, fp16=cfg["fp16"])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
    )
    hands = []
    idx = 0
    for batch in loader:
        batch = to_device_safe(batch, device)
        with torch.inference_mode(), autocast_ctx(device, cfg["autocast_dtype"]):
            out = model(batch)
        kp2d_arr = out["pred_keypoints_2d"].detach().cpu().numpy()      # (B,21,2) crop coords
        verts_arr = out["pred_vertices"].detach().cpu().numpy()          # (B,778,3)
        kp3d_arr = out["pred_keypoints_3d"].detach().cpu().numpy()       # (B,21,3)
        for n in range(batch["img"].shape[0]):
            is_r = batch["right"][n].item() > 0.5
            handed = "right" if is_r else "left"
            box_center = batch["box_center"][n].cpu().numpy()
            box_size = batch["box_size"][n].cpu().item()

            kp_crop = kp2d_arr[n].copy()
            v = verts_arr[n].copy()
            kp3 = kp3d_arr[n].copy()
            # ViTDetDataset flips left-hand crops; un-flip X on every X-bearing tensor.
            if not is_r:
                kp_crop[:, 0] *= -1
                v[:, 0] *= -1
                kp3[:, 0] *= -1

            kp_full = np.empty_like(kp_crop)
            kp_full[:, 0] = box_center[0] + kp_crop[:, 0] * box_size
            kp_full[:, 1] = box_center[1] + kp_crop[:, 1] * box_size

            hands.append({
                "kp2d_full": kp_full,
                "kp2d_crop": kp_crop,
                "kp3d": kp3,
                "verts": v,
                "handed": handed,
                "bbox": boxes[idx],
                "box_center": box_center,
                "box_size": box_size,
            })
            idx += 1
    return hands


def project_vertices(hand):
    """Fit weak-perspective (s, tx, ty) from kp3d.xy -> kp2d_crop, apply to verts.
    Returns image-space vertex 2D positions and the (possibly mirrored) 3D verts."""
    kp3 = hand["kp3d"]            # (21, 3)
    kp_crop = hand["kp2d_crop"]   # (21, 2)
    verts = hand["verts"]         # (778, 3)

    n = kp3.shape[0]
    A = np.zeros((2 * n, 3), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    A[0::2, 0] = kp3[:, 0]; A[0::2, 1] = 1.0; b[0::2] = kp_crop[:, 0]
    A[1::2, 0] = kp3[:, 1]; A[1::2, 2] = 1.0; b[1::2] = kp_crop[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    s, tx, ty = sol

    verts_crop = np.column_stack([s * verts[:, 0] + tx, s * verts[:, 1] + ty])
    bc = hand["box_center"]; bs = hand["box_size"]
    verts_full = np.column_stack([
        bc[0] + verts_crop[:, 0] * bs,
        bc[1] + verts_crop[:, 1] * bs,
    ])
    return verts_full.astype(np.float32), verts


def render_mesh(img, verts2d, verts3d, faces,
                base_color=(255, 110, 30), edge_color=(220, 80, 20),
                alpha=0.55, draw_edges=True):
    """Painter's algorithm + Lambertian shading. base_color / edge_color are BGR."""
    overlay = img.copy()
    pts2 = verts2d[faces].astype(np.int32)               # (Nfaces, 3, 2)
    tri_3d = verts3d[faces]                              # (Nfaces, 3, 3)
    face_z = tri_3d.mean(axis=1)[:, 2]
    order = np.argsort(-face_z)                           # back -> front

    # Bulk Lambertian: dot each face normal with a fixed light direction.
    e1 = tri_3d[:, 1] - tri_3d[:, 0]
    e2 = tri_3d[:, 2] - tri_3d[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    n_hat = normals / norms
    light = np.array([0.0, -0.4, -1.0])
    light /= np.linalg.norm(light)
    shade = np.clip(np.abs(n_hat @ light), 0.30, 1.0)    # never fully black
    base = np.array(base_color, dtype=np.float32)

    h, w = img.shape[:2]
    drawn = 0
    for fi in order:
        p = pts2[fi]
        # Cheap reject: triangle entirely off-screen.
        if p[:, 0].max() < 0 or p[:, 1].max() < 0 or p[:, 0].min() >= w or p[:, 1].min() >= h:
            continue
        col = (base * shade[fi]).astype(np.uint8)
        cv2.fillConvexPoly(overlay, p, (int(col[0]), int(col[1]), int(col[2])), lineType=cv2.LINE_AA)
        if draw_edges:
            cv2.polylines(overlay, [p], True, edge_color, 1, cv2.LINE_AA)
        drawn += 1

    out = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    return out, drawn


def annotate_hand(img, hand, depth_cm=None):
    """Light label so we know the script identified hand chirality correctly."""
    x0, y0, _, _ = hand["bbox"].astype(int)
    label = hand["handed"]
    cv2.putText(img, label, (x0, max(0, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-left", default=str(DEFAULT_LEFT))
    ap.add_argument("--clip-right", default=str(DEFAULT_RIGHT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--left-only", action="store_true",
                    help="render the left view only (halves WiLoR cost)")
    ap.add_argument("--alpha", type=float, default=0.55, help="mesh overlay opacity")
    ap.add_argument("--no-edges", action="store_true", help="skip drawing triangle edges")
    args = ap.parse_args()

    def from_root(p):
        pp = Path(p)
        return pp if pp.is_absolute() else (PROJECT_ROOT / pp).resolve()

    clip_l = from_root(args.clip_left)
    clip_r = from_root(args.clip_right)
    if not clip_l.is_file():
        sys.exit(f"clip not found: {clip_l}")
    if not args.left_only and not clip_r.is_file():
        sys.exit(f"clip not found: {clip_r}")

    out_video = PROJECT_ROOT / f"outputs/{today_pretty()} - wilor ar overlay [{args.tag}].mp4"

    device = pick_device()
    cfg = configure_perf(device)
    print(f"device: {device}  yolo: {cfg['yolo_device']}  fp16: {cfg['fp16']}  "
          f"workers: {cfg['num_workers']}  pin: {cfg['pin_memory']}  "
          f"autocast: {cfg['autocast_dtype']}")
    print(f"left:   {clip_l.name}")
    if not args.left_only:
        print(f"right:  {clip_r.name}")

    print("loading WiLoR ...")
    t0 = time.time()
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    model = model.to(device).eval()
    print(f"loaded in {time.time()-t0:.1f}s")

    detector = YOLO("./pretrained_models/detector.pt")
    faces = get_mano_faces(model)
    print(f"MANO faces: {faces.shape}")

    cap_l = cv2.VideoCapture(str(clip_l))
    cap_r = None if args.left_only else cv2.VideoCapture(str(clip_r))
    fps = cap_l.get(cv2.CAP_PROP_FPS)
    n = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    if cap_r is not None:
        n = min(n, int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.max_frames:
        n = min(n, args.max_frames)
    w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w if args.left_only else w * 2
    print(f"{n} frames @ {fps:.1f} fps, {w}x{h} per camera, output {out_w}x{h}")

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, h))

    n_l_hands = 0
    n_r_hands = 0
    inference_time = 0.0
    render_time = 0.0
    t0 = time.time()
    for i in range(n):
        ok_l, fl = cap_l.read()
        if not ok_l:
            break
        if cap_r is not None:
            ok_r, fr = cap_r.read()
            if not ok_r:
                break

        ti0 = time.time()
        hands_l = detect_and_regress(detector, model, model_cfg, fl, device, cfg)
        if cap_r is not None:
            hands_r = detect_and_regress(detector, model, model_cfg, fr, device, cfg)
        inference_time += time.time() - ti0

        tr0 = time.time()
        for h in hands_l:
            v2, v3 = project_vertices(h)
            fl, _ = render_mesh(fl, v2, v3, faces, alpha=args.alpha, draw_edges=not args.no_edges)
            annotate_hand(fl, h)
        n_l_hands += len(hands_l)
        if cap_r is not None:
            for h in hands_r:
                v2, v3 = project_vertices(h)
                fr, _ = render_mesh(fr, v2, v3, faces, alpha=args.alpha, draw_edges=not args.no_edges)
                annotate_hand(fr, h)
            n_r_hands += len(hands_r)
        render_time += time.time() - tr0

        frame = fl if cap_r is None else np.hstack([fl, fr])
        writer.write(frame)

        if i % 5 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (n - i - 1) / rate if rate else 0
            sys.stdout.write(f"\rframe {i+1}/{n}  L={n_l_hands} R={n_r_hands}  "
                             f"{rate:.2f} fps  eta {eta:.0f}s   ")
            sys.stdout.flush()
    print()

    cap_l.release()
    if cap_r is not None:
        cap_r.release()
    writer.release()

    print(f"hands rendered: left {n_l_hands}, right {n_r_hands}")
    print(f"inference: {inference_time:.1f}s   render: {render_time:.1f}s   "
          f"wall: {time.time()-t0:.1f}s")
    print(f"wrote {out_video.name}")
    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(out_video)], check=False)


if __name__ == "__main__":
    main()
