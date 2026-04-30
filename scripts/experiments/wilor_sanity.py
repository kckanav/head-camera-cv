"""Stage 8 / Phase 1: WiLoR sanity check on a real Pi-Cam frame.

What this validates end-to-end:
  - The .venv-hamer environment can load WiLoR (ckpt + MANO model).
  - YOLO detection runs and finds a hand on a real frame from our rig.
  - WiLoR's MANO regressor produces a sensible mesh (extent ~hand-sized).
  - 21 reprojected 2D keypoints overlay onto the actual hand in the frame.
  - Inference runs on MPS (Apple Metal) on this M2 host.

This script lives at the project root so it survives a re-clone of wilor/.
The wilor/ dir is gitignored (large weights, separate license), so anything
inside it can be lost; this entry point is what we keep.

Run:
    .venv-hamer/bin/python wilor_sanity.py

Notes on the workarounds applied here (kept intentionally inline — they are
specific to: WiLoR @ rolpotamias/main, ultralytics 8.1.34, torch 2.11):
  * pyrender doesn't load on macOS without OSMesa, so we stub the three
    renderer-using submodules. Inference works; only visualisation is gated.
  * PyTorch 2.6 flipped torch.load's default to weights_only=True. The YOLO
    detector ckpt has pickled class refs and can't load under that default;
    we monkey-patch torch.load back to weights_only=False (we trust both
    checkpoint sources from the upstream HuggingFace mirror).
  * The YOLO ckpt was pickled with `dill`, which is not in WiLoR's
    requirements.txt — we install it separately.

If any of these stop being needed (upstream pinning a newer ultralytics,
PyTorch reverting the default, etc.), the workaround can be deleted.
"""

import sys
import time
from pathlib import Path

# _lib bootstrap — must come BEFORE importing torch / wilor.* so wilor_setup's
# pyrender stubs and torch.load patch take effect.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import wilor_setup  # noqa: F401, E402  side-effects: stubs + torch.load patch + sys.path + chdir
from wilor_setup import PROJECT_ROOT  # noqa: E402

import torch                                                             # noqa: E402
import cv2                                                               # noqa: E402
import numpy as np                                                       # noqa: E402
from ultralytics import YOLO                                             # noqa: E402

from dated import today_pretty                                           # noqa: E402
from device import pick_device, configure_perf, to_device_safe as to_device  # noqa: E402

from wilor.models import load_wilor                                      # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset                  # noqa: E402

INPUT_IMG = PROJECT_ROOT / "inputs/24th April 2026 - photo cam0.jpg"
if not INPUT_IMG.is_file():
    sys.exit(f"missing {INPUT_IMG}: input photo not found")

OUT_OVERLAY = PROJECT_ROOT / f"outputs/{today_pretty()} - wilor sanity overlay.jpg"
OUT_OBJ = PROJECT_ROOT / f"outputs/{today_pretty()} - wilor sanity hand0.obj"


def save_obj(path: Path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def main():
    device = pick_device()
    cfg = configure_perf(device)
    print(f"device: {device}  yolo: {cfg['yolo_device']}")

    print("loading WiLoR ...")
    t0 = time.time()
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    model = model.to(device).eval()
    print(f"loaded in {time.time()-t0:.1f}s")

    print("loading YOLO detector ...")
    detector = YOLO("./pretrained_models/detector.pt")

    img = cv2.imread(str(INPUT_IMG))
    h, w = img.shape[:2]
    print(f"image: {INPUT_IMG.name}  ({w}x{h})")

    print("running YOLO ...")
    t0 = time.time()
    # ultralytics has a known MPS bug for Pose models, so YOLO runs on CPU
    # while WiLoR (the heavy ViT) runs on MPS. See ultralytics issue #4031.
    detections = detector(img, device=cfg["yolo_device"], conf=0.3, verbose=False)[0]
    bboxes, is_right = [], []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())
    print(f"YOLO: {len(bboxes)} hands detected in {time.time()-t0:.2f}s")
    if not bboxes:
        sys.exit("no hands detected; aborting sanity check")

    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    dataset = ViTDetDataset(model_cfg, img, boxes, right, rescale_factor=2.0, fp16=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print("running WiLoR ...")
    t0 = time.time()
    overlay = img.copy()
    saved_first_obj = False
    for batch_idx, batch in enumerate(loader):
        batch = to_device(batch, device)
        with torch.no_grad():
            out = model(batch)
        verts = out["pred_vertices"].detach().cpu().numpy()        # (B, 778, 3)
        joints2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # (B, 21, 2) in [-0.5, 0.5]
        for n in range(batch["img"].shape[0]):
            hand_idx = batch_idx * 16 + n
            is_right = batch["right"][n].item() > 0.5
            handed = "right" if is_right else "left"

            # ViTDetDataset flips left-hand crops horizontally before
            # inference, so both pred_vertices and pred_keypoints_2d come
            # out in flipped-crop coords. Un-flip X to match the original
            # image / left-hand anatomy.
            v = verts[n].copy()
            kp_crop = joints2d[n].copy()
            if not is_right:
                v[:, 0] *= -1
                kp_crop[:, 0] *= -1

            if not saved_first_obj:
                save_obj(OUT_OBJ, v, model.mano.faces)
                saved_first_obj = True
                print(f"  hand {hand_idx} ({handed}): mesh -> {OUT_OBJ.name}")

            extent_cm = (v.max(0) - v.min(0)).max() * 100.0
            print(f"  hand {hand_idx} ({handed}): mesh extent ~{extent_cm:.1f} cm "
                  f"(MANO local frame, monocular scale)")

            box_center = batch["box_center"][n].cpu().numpy()
            box_size = batch["box_size"][n].cpu().item()
            kp = kp_crop  # already un-flipped above
            kp[:, 0] = box_center[0] + kp[:, 0] * box_size
            kp[:, 1] = box_center[1] + kp[:, 1] * box_size
            color = (0, 200, 255) if handed == "right" else (255, 100, 0)
            for x, y in kp.astype(int):
                cv2.circle(overlay, (int(x), int(y)), 6, color, -1)
                cv2.circle(overlay, (int(x), int(y)), 7, (255, 255, 255), 1)
            x0, y0, x1, y1 = boxes[hand_idx].astype(int)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
            cv2.putText(overlay, handed, (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imwrite(str(OUT_OVERLAY), overlay)
    print(f"\nWiLoR ran in {time.time()-t0:.1f}s")
    print(f"overlay -> {OUT_OVERLAY}")
    print(f"mesh    -> {OUT_OBJ}")

    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(OUT_OVERLAY)], check=False)


if __name__ == "__main__":
    main()
