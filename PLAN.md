# Plan: dexterous hand capture via WiLoR + stereo

> **Pick-up-where-we-left-off (read this first if starting a new session):**
>
> - Repo state: stages 1, 3, 4 of the main pipeline are committed (MediaPipe per-camera, stereo calibration, sparse 3D triangulation). See `README.md` for the full picture.
> - Stage 8 status: **Phase 0 done** (env + WiLoR clone + checkpoints + MANO models linked) and **Phase 1 done** (sanity check passes — WiLoR runs on MPS, mesh extent ~16 cm, handedness correct on a real Pi-Cam frame).
> - **Next concrete step: Phase 2** — write `wilor_perview.py` to run WiLoR on the raw frames of a clip from `inputs/` and dump per-frame MANO params to `outputs/<date> - cam{0,1} mano raw.npz`.
> - Verify the env still works at any time:
>     ```bash
>     .venv-hamer/bin/python wilor_sanity.py
>     ```
>   Should print "device: mps", detect 1 hand, and pop open the overlay in Preview.
> - The WiLoR repo lives at `wilor/` (gitignored — large weights, separate license). If you re-clone it, follow the Phase 0 commands further down. The MANO models live at `wilor/models/MANO_{RIGHT,LEFT}.pkl` (you uploaded them) and are symlinked from `wilor/mano_data/` where the WiLoR config expects them.
> - Sanity script entry point: `wilor_sanity.py` (project root, tracked).
> - Outputs from the last sanity run: `outputs/<date> - wilor sanity overlay.jpg` and `outputs/<date> - wilor sanity hand0.obj`.

---

Status as of `27 April 2026`: stages 1, 3, 4 of the README ladder are done
(MediaPipe per-camera, stereo calibration, sparse 3D triangulation). This
plan covers the next push — replacing MediaPipe with a MANO-producing model
(WiLoR primary, HaMeR fallback) so the captured dataset is rich enough to
retarget to *any* gripper.

## Why

The captured dataset is for **robot-manipulation imitation learning**. The
robot may eventually use a parallel-jaw gripper, a 3-finger gripper, or an
anthropomorphic hand — the dataset must be **richer than any specific
gripper** so retargeting code can project it onto whatever robot is being
trained.

Concretely that means we need, per-frame, per-hand:

- **MANO joint angles** — `(16, 3)` axis-angle, the canonical retargeting target
- **MANO mesh vertices** — `(778, 3)` metric, needed for hand-object contact
- **Wrist 6DoF** — translation + rotation in the world frame, from stereo
- **Hand-presence flag** — per slot, whether the estimate is trustworthy

MediaPipe gives 21 sparse keypoints per hand; downstream code would have to
fit MANO to those points anyway, with extra error. So we go directly to a
MANO-producing model.

## Model choice: WiLoR primary, HaMeR fallback

| | WiLoR | HaMeR |
|---|---|---|
| Source | https://github.com/rolpotamias/WiLoR (CVPR 2025) | https://github.com/geopavlakos/hamer (CVPR 2024) |
| Detector | Built-in fully-convolutional, no external dep | detectron2-based |
| MPS-compatibility on M2 | Likely fine (no detectron2) | detectron2 has no MPS support — would force CPU detection |
| Reported speed | Detection >130 fps (M / S variants); reconstruction is ViT-based | ~6 fps total on consumer GPU |
| Output | MANO params + mesh | MANO params + mesh |
| Robotics-literature usage | Newer, less | More established (DexCap, Bunny-VisionPro reference HaMeR) |

WiLoR wins on **ease of setup on this Mac** and on **inference speed**. Both
produce the same MANO output, so swap-cost is low if WiLoR has issues on our
egocentric-top-down distribution.

## Why our stereo gives us an unusual advantage

Both WiLoR and HaMeR are **monocular** by default — they predict pose +
shape but **absolute scale is ambiguous** (a small hand close vs a big hand
far look identical to a single camera). With our calibrated stereo we can
fix scale: triangulate stable landmarks (wrist, palm-base, middle-finger
MCP) from the two views, then rescale the monocular MANO output to match.
This is a stronger setup than the default literature pipeline gets.

## Phased plan (each phase independently testable)

### Phase 0 — environment & weights
- New venv: `.venv-hamer/` (separate from `.venv` to avoid PyTorch ↔ MediaPipe
  dep conflicts). Name kept generic so it's reusable if we swap WiLoR for
  HaMeR later.
- Clone repo: `git clone https://github.com/rolpotamias/WiLoR wilor/` into
  the project root (subdir, gitignored).
- Install: torch (with MPS), the wilor package, smplx, and whatever the
  upstream `setup.py` / `pyproject.toml` lists.
- Download WiLoR checkpoints via the upstream script.
- **User-required step**: register at `mano.is.tue.mpg.de`, download
  `MANO_RIGHT.pkl`, place at `wilor/mano_data/MANO_RIGHT.pkl` (path per
  upstream README).
- Validate: WiLoR demo runs on a stock sample image, produces a mesh.

### Phase 1 — single-frame, single-view sanity *(done)*
- Entry point: `wilor_sanity.py` at the project root.
- Runs YOLO (CPU — known MPS bug for Pose models) + WiLoR (MPS) on
  `inputs/24th April 2026 - photo cam0.jpg`. Outputs an OpenCV overlay
  (`outputs/<date> - wilor sanity overlay.jpg`) and the MANO mesh as a
  plain `.obj` (`outputs/<date> - wilor sanity hand0.obj`).
- **Result on the test capture (4608×2592):**
  - YOLO detected 1 hand correctly, labelled "right" (correct vs. MediaPipe
    which couldn't reliably get handedness on egocentric data — promising
    sign for our use case).
  - WiLoR mesh extent ~16.3 cm in MANO local frame — anatomically plausible.
  - 21 reprojected 2D keypoints land on the actual hand in the overlay.
  - Inference: YOLO 0.4 s on CPU; WiLoR 6.3 s on MPS for the *first* call
    (kernel compilation overhead). Sustained per-frame cost in Phase 2 will
    be much lower — we'll measure properly there.
- Pyrender doesn't load on macOS without OSMesa, so the sanity script stubs
  the three renderer-using submodules of `wilor.utils`. Inference works;
  visualisation will need a separate path in Phase 4 (probably trimesh or
  matplotlib).

### Phase 2 — per-view, per-clip pipeline
- New script: `wilor_perview.py`.
- Loop over **raw** frames (not rectified) — WiLoR was trained on natural
  imagery, no need to feed it warped pixels. Skipping rectification at this
  stage also dodges the FOV-loss issue we hit with MediaPipe.
- Per frame, per view: WiLoR detection + MANO regression for up to 2 hands.
- Output per view: `outputs/<date> - cam{0,1} mano raw.npz`
  - `mano_pose[N, 2, 16, 3]` axis-angle joint rotations
  - `mano_betas[N, 2, 10]` shape parameters
  - `vertices[N, 2, 778, 3]` in WiLoR's local (un-scaled) frame
  - `keypoints_2d[N, 2, 21, 2]` reprojected pixel coords for matching/debugging
  - `confidence[N, 2]` detector score
  - `bbox[N, 2, 4]` for bookkeeping

### Phase 3 — stereo fusion
- New script: `wilor_stereo.py`.
- Per frame:
  1. **Hand correspondence across views** — match L/R detections by epipolar
     constraint: project palm-centroid through the rectification map, require
     matched-row alignment within ~20 px (same approach as
     `triangulate.py`).
  2. **Scale fusion** — triangulate the wrist + palm-base + middle-MCP
     keypoints via the calibration's `P1, P2`. Compute scale factor
     `s = mean(|triangulated_pair_distance| / |WiLoR_pair_distance|)` so
     the monocular mesh is rescaled to metric ground truth from stereo.
  3. **Pose fusion** — average the two views' joint angles in axis-angle
     space (or weight by confidence). Wrist 6DoF taken from the metric
     stereo triangulation, not from monocular regression.
- Output: `outputs/<date> - mano stereo fused.npz` — per-frame MANO at
  metric scale.

### Phase 4 — visualization & validation
- Side-by-side rectified video with mesh overlay (extends `triangulate.py`'s
  output style).
- Trajectory plots: wrist 6DoF over time, per-finger flexion over time
  (extends `inspect_3d.py`).
- Sanity bounds: mesh volume in physically plausible range; joint angles
  inside MANO's training distribution.
- Cross-check: wrist trajectory should agree with the existing
  MediaPipe-stereo `.npz` from stage 4 to within a few mm. Disagreement is a
  flag.

### Phase 5 — robot-training export
- Final dataset format (`outputs/<date> - manipulation dataset.npz` or `.h5`):
  - `mano_pose[N, 2, 16, 3]`
  - `mano_betas[N, 2, 10]`
  - `vertices[N, 2, 778, 3]` metric, world frame
  - `wrist_6dof[N, 2, 6]` translation + axis-angle rotation
  - `hand_present[N, 2]` boolean
  - `timestamps[N]`
  - `intrinsics`, `extrinsics` (copied from calibration `.npz` for self-containment)
- Schema documented inline so retargeting code can consume without
  reverse-engineering.

## Where existing code fits

| Existing | Role going forward |
|---|---|
| `dualstream.py` | Pi-side capture, unchanged |
| `make_calibration_board.py`, `calibrate.py` | Provide the calibration `.npz` that fixes stereo scale — unchanged |
| `triangulate.py` (MediaPipe stereo) | Kept as **fast preview path** (~37 fps) and as a **regression baseline** — wrist trajectories should agree with WiLoR-stereo |
| `inspect_3d.py` | Extended in phase 4 for MANO trajectory plots |
| `process.py` (stitching) | No longer used in main pipeline; reference only |

## Risks & open questions

- **MPS coverage**: PyTorch MPS supports most ops needed for transformer
  inference, but WiLoR may use ops that fall back to CPU. Plan: try, accept
  warnings, only act if inference is unusably slow.
- **Egocentric extreme angles**: top-down hand poses on a head-mount may sit
  at the edge of WiLoR's training distribution. Phase 1 will tell us.
- **Detection misses**: even single-stage, WiLoR has a confidence floor. We
  compare detection rates against MediaPipe's 99% / 93% as the regression
  baseline.
- **Scale fusion correctness**: assumes monocular shape is right *up to
  scale*. If WiLoR's shape predictions are noisy frame-to-frame the scale
  factor will jitter. Mitigation: low-pass-filter the scale over short
  windows, or fit a single shape per identity offline.

## Decisions to make as we go

- WiLoR repo location: `wilor/` subdir (gitignored) vs sibling dir.
  **Default: subdir.**
- Data format: `.npz` (simple) vs `.h5` (better for big datasets).
  **Default: `.npz` until total dataset > ~1 GB.**
- Whether to keep MediaPipe pipeline running alongside WiLoR for every clip,
  or only as needed. **Default: alongside, since it's cheap and useful as
  regression check.**

## Quick-reference setup commands (Phase 0)

```bash
# from /Users/kanavgupta/Desktop/cameramount
brew install python@3.10                # macOS only had 3.9.6; WiLoR wants 3.10
python3.10 -m venv .venv-hamer
.venv-hamer/bin/pip install --upgrade pip wheel

# --recursive picks up upstream submodules
git clone --recursive https://github.com/rolpotamias/WiLoR wilor

# WiLoR's requirements.txt pins torch+cu117. We use the default PyTorch index
# instead, which gives a build with MPS (Apple Metal) support on macOS.
.venv-hamer/bin/pip install torch torchvision

# Cython is needed at build time for xtcocotools' C extension.
.venv-hamer/bin/pip install Cython

# chumpy's setup.py imports `pip` directly, which fails inside pip's isolated
# build env. Install it separately with --no-build-isolation so it can see the
# venv's pip. Then install the rest of the requirements (also with
# --no-build-isolation so subsequent C-extension builds see numpy).
.venv-hamer/bin/pip install --no-build-isolation \
  "chumpy @ git+https://github.com/mattloper/chumpy"
grep -v "^chumpy" wilor/requirements.txt > /tmp/wilor_reqs.txt
.venv-hamer/bin/pip install --no-build-isolation -r /tmp/wilor_reqs.txt

# Pretrained checkpoints (51 MB detector + 2.4 GB MANO regressor)
curl -L -o wilor/pretrained_models/detector.pt \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
curl -L -o wilor/pretrained_models/wilor_final.ckpt \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt

# MANO model (academic license — must be done by user):
#   1. Register at https://mano.is.tue.mpg.de/
#   2. Download MANO_RIGHT.pkl (the right-hand model file)
#   3. Place at wilor/mano_data/MANO_RIGHT.pkl
#      (smplx looks for "MANO_{RIGHT,LEFT}.pkl" inside the configured dir;
#       WiLoR config sets MANO.MODEL_PATH = './mano_data/'.)
```

### Install gotchas encountered (worth keeping a note of)

- **`chumpy` git install fails under build isolation** — its `setup.py` does
  `import pip`, which is not present in the isolated build env. Workaround:
  `pip install --no-build-isolation "chumpy @ git+..."`.
- **`xtcocotools` build needs Cython + numpy at build time** — install Cython
  first, then run the rest of the install with `--no-build-isolation` so the
  C extension build can see the venv's numpy.
- **PyTorch CUDA pin in `requirements.txt`** — the upstream pins `cu117`.
  On Apple Silicon you want the default-index torch (MPS-enabled). Install
  `torch torchvision` first from the default index, then install the rest
  of the requirements (which won't try to re-resolve torch).
- **`dill` not in requirements** — the YOLO `detector.pt` was pickled with
  `dill`, but WiLoR's `requirements.txt` doesn't list it. Install separately:
  `pip install dill`.
- **PyTorch 2.6 broke YOLO checkpoint loading** — `torch.load`'s default
  flipped to `weights_only=True`. The YOLO ckpt has pickled class refs and
  can't load under that default. The sanity script monkey-patches
  `torch.load` to `weights_only=False` (we trust both checkpoint sources from
  upstream HuggingFace). When ultralytics > 8.3 lands, this won't be needed.
- **Pyrender doesn't load on macOS** — needs EGL or OSMesa, neither of which
  is on macOS by default. The sanity script stubs the three renderer-using
  submodules of `wilor.utils` so inference works. Real 3D viz comes later.
- **YOLO MPS bug for Pose models** — ultralytics issue #4031 documents that
  Pose-task YOLO models on MPS are broken / extremely slow (NMS time-limit
  warnings, 11 s vs 0.4 s on CPU on this rig). Run the detector on CPU and
  WiLoR on MPS — the detector is fast on CPU anyway.
- **MPS doesn't support `float64`** — the WiLoR dataset emits some float64
  tensors (`box_size` etc.). The sanity script casts those to `float32`
  before the `.to(device)` call. If we keep MPS for Phase 2, that cast goes
  in the per-clip pipeline too.
- **WiLoR repo layout vs sys.path** — `cameramount/wilor/` and
  `cameramount/wilor/wilor/` both have no `__init__.py`. If `PROJECT_ROOT`
  (cameramount/) is on sys.path, Python merges them into a single namespace
  package and `wilor.models` resolves ambiguously (`cameramount/wilor/models/`
  is where the user uploaded MANO files, which isn't a Python package).
  Sanity script removes PROJECT_ROOT from sys.path and inserts WILOR_DIR
  alone.

## Glossary (so future-you doesn't have to look these up)

- **MANO** — Embodied Hands parametric model. 16 joints × 3 axis-angle
  rotations for pose, 10 betas for shape. Outputs a 778-vertex mesh.
- **Axis-angle** — 3-vector encoding rotation; direction is rotation axis,
  magnitude is angle. Compact alternative to rotation matrices.
- **Rectification** — virtual rotation of stereo views so corresponding
  points lie on the same image row. Computed once in `calibrate.py`,
  reused via `cv2.remap`.
- **Epipolar constraint** — for any 3D point seen by both cameras, its image
  in the right view lies on the *epipolar line* of the left view's point.
  After rectification, all epipolar lines are horizontal, so the constraint
  becomes "same row".
- **Triangulation** — given a point's pixel coordinates in two calibrated
  views, recover its 3D position. `cv2.triangulatePoints(P1, P2, ...)`.
