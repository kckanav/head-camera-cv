# Plan: dexterous hand capture via WiLoR + stereo

> **Pick-up-where-we-left-off (read this first if starting a new session):**
>
> - Repo state: stages 1, 3, 4 of the main pipeline are committed (MediaPipe per-camera, stereo calibration, sparse 3D triangulation). See `README.md` for the full picture.
> - Stage 8 status: **Phases 0, 1, 2, and minimal Phase 3 done.** **Per-view AR overlay (`scripts/wilor_ar_overlay.py`) done** — visually validates the monocular mesh quality. **Full Phase 3 in progress**: see "Phase 3 (full) — stereo MANO mesh fusion" further down.
> - **Phase 3 sub-status**: 3.1 (full keypoint triangulation), 3.2 (Umeyama fusion), and 3.3 (fused-mesh AR overlay) are the minimum viable; 3.4 (LM refinement) and 3.5 (β refinement) are upgrades. The new entry point is `scripts/wilor_phase3.py`.
> - Verify the env still works at any time (run from repo root):
>     ```bash
>     .venv-hamer/bin/python scripts/wilor_sanity.py        # single image, fast
>     .venv-hamer/bin/python scripts/wilor_stereo_demo.py   # 15-s clip, ~7 min on MPS
>     .venv-hamer/bin/python scripts/wilor_ar_overlay.py    # per-view monocular AR overlay (10-s wide clip)
>     .venv-hamer/bin/python scripts/wilor_phase3.py        # stereo-fused mesh AR overlay (10-s wide clip)
>     ```
>   `wilor_sanity.py` should print "device: mps", detect 1 hand, pop open the overlay.
>   `wilor_stereo_demo.py` runs through 451 frames at ~1 fps on warm MPS and pops open the side-by-side annotated video at the end.
> - The WiLoR repo lives at `wilor/` (gitignored — large weights, separate license). If you re-clone it, follow the Phase 0 commands further down. The MANO models live at `wilor/models/MANO_{RIGHT,LEFT}.pkl` (you uploaded them) and are symlinked from `wilor/mano_data/` where the WiLoR config expects them. **Note:** WiLoR only ever loads `MANO_RIGHT.pkl` and uses the standard mirror trick for left hands — `MANO_LEFT.pkl` is symlinked but unused.
> - Tracked entry points (under `scripts/`): `wilor_sanity.py`, `wilor_stereo_demo.py`, `wilor_ar_overlay.py`, `wilor_phase3.py`.
> - Last run results live at `outputs/<date> - wilor sanity *`, `outputs/<date> - wilor stereo demo *`, `outputs/<date> - wilor ar overlay *`, `outputs/<date> - phase3 fused *`.

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
- Entry point: `scripts/wilor_sanity.py`.
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

### Phase 2 + minimal Phase 3 — per-view WiLoR with stereo wrist triangulation *(done as one combined demo)*
- Entry point: `scripts/wilor_stereo_demo.py`. Default clip is the 15-s grease
  clip with the wide-FOV calibration; `--long` switches to the 60-s narrow
  clip preset. CLI args `--clip-left/--clip-right/--calib/--tag/--max-frames`
  override paths and limits — relative paths resolve against the repo root.
- Per frame pair:
  1. YOLO detect on the raw left + right frame (CPU; ultralytics MPS bug).
  2. WiLoR forward pass per detected hand per view (MPS, ~0.9 s warm).
  3. Project each hand's wrist 2D into rectified space via
     `cv2.undistortPoints(..., R=R1|R2, P=P1|P2)`.
  4. Match left/right hand detections by rectified wrist-row proximity
     (`ROW_TOL_PX = 30`).
  5. Triangulate matched wrists with `cv2.triangulatePoints(P1, P2, ...)` →
     metric wrist 3D in the left rectified camera frame.
  6. Sanity-filter wrist Z to `[5 cm, 200 cm]`. (5 cm not 10 — on a head-mount
     hands genuinely get this close when manipulating something near the face.)
  7. Annotate the SBS frame with the WiLoR skeleton, bbox, handedness label,
     and the metric `z=XX.X cm` if pairing succeeded.
- Output:
  - `outputs/<date> - wilor stereo demo {tag}.mp4` — SBS annotated video.
  - `outputs/<date> - wilor stereo demo {tag}.npz` — per-frame
    `wrist_3d_metric[N, 2, 3]`, `keypoints_2d_left/right[N, 2, 21, 2]`,
    `handedness[N, 2]`.

**Result on the 15-s grease clip (451 frames, 7.5 min wall time):**
- Inference 0.98 s/frame on warm MPS.
- Wrist depth (cm): median 34.1, 5–95 [6.5, 40.1], min 5.1, max 69.7.
- Median lines up well with what MediaPipe-stereo reported on similar
  footage (~33 cm), giving us a regression baseline.

**Why this is "minimal" Phase 3 and what's still ahead:**
- Only the wrist landmark is triangulated. The full Phase 3 (next section)
  triangulates all 21 landmarks and fits the entire MANO mesh into the
  shared world frame.

### Phase 3 (full) — stereo MANO mesh fusion *(in progress)*

**Goal.** For each frame and each detected hand, produce *one* MANO mesh in
the rectified-left camera frame at metric scale, consistent with both
views' WiLoR predictions. Replaces the two independent monocular meshes
that `wilor_ar_overlay.py` renders.

**The core unknowns per matched hand per frame: a similarity transform
(s, R, t) — 7 DOF.** That places WiLoR's MANO output (in its own root-
centered local frame) into the rectified-left camera frame. WiLoR's pose θ
and shape β stay as predicted for the first iteration; β refinement is
Phase 3.5.

**Algorithm — two steps, the second is optional:**

1. **Umeyama (closed-form)** — triangulate all 21 keypoints to get 21 metric
   3D points; solve `Σᵢ wᵢ ‖triangulated_3d[i] − (s·R·pred_kp3d_local[i] + t)‖²`
   via SVD (Umeyama 1991, weighted variant). Closed form, no iteration.
2. **Levenberg–Marquardt (refinement, Phase 3.4)** — minimize 2D reprojection
   error across both views over `(s, axis_angle_R, t)` with a robust loss.
   Initialize from step 1.

**Phased implementation. Each substep is independently testable.**

| Substep | What | Status |
|---|---|---|
| 3.1 | Triangulate all 21 keypoints per matched hand. Per-keypoint validity = (rectified-row agreement < 8 px) & (Z plausible). | in progress |
| 3.2 | Weighted Umeyama. Down-weight fingertips (1 px disparity → big depth error at 41 mm baseline). Apply `(s, R, t)` to all 778 vertices. | in progress |
| 3.3 | Project the world-frame mesh into both unrectified views via `R1.T` (undo rectification) + `cv2.projectPoints` with `K_l/K_r, dist_l/dist_r`. Render with the same painter's-algorithm code as `wilor_ar_overlay.py`. | in progress |
| 3.4 | LM refinement of `(s, R, t)` against 2D reprojection error in both views. `scipy.optimize.least_squares` with `loss="soft_l1"`. | deferred |
| 3.5 | Joint solve over `(s, β, R, t)` — adds shape refinement. Stereo identifies β; monocular β is ill-posed. | deferred |
| 3.6 | Sequence smoothing. Belongs in Phase 4 if jitter is bothersome. | deferred |

**File plan.** Single new entry point: `scripts/wilor_phase3.py`. Reuses the
WiLoR detect+regress logic from `wilor_stereo_demo.py` and the painter's-
algorithm renderer from `wilor_ar_overlay.py` (copied for self-containment;
shared module is a refactor for later). Outputs:

- `outputs/<date> - phase3 fused [<tag>].npz` — schema below.
- `outputs/<date> - phase3 ar overlay [<tag>].mp4` — SBS, fused mesh
  rendered into both **unrectified** frames so it's directly comparable to
  the per-view monocular `wilor_ar_overlay.py` output.

**Output schema (`.npz`):**

```
verts_3d_world      (N, 2, 778, 3)  float32 — world (= rectified-left) frame, NaN where missing
kp_3d_world         (N, 2,  21, 3)  float32 — triangulated, per-keypoint NaN where invalid
kp_valid            (N, 2,  21)     bool    — row-tolerance + Z-bound mask
placement_s         (N, 2)          float32 — Umeyama scale
placement_R         (N, 2,  3, 3)   float32 — Umeyama rotation
placement_t         (N, 2,  3)      float32 — Umeyama translation
umeyama_residual_m  (N, 2)          float32 — mean residual after fit, metres
reproj_err_left_px  (N, 2)          float32 — median 2D error of fused mesh keypoints in left
reproj_err_right_px (N, 2)          float32 — same for right
handedness          (N, 2)                  — strings
fps, image_size, calib_path                   self-contained for downstream code
```

**Validation (what makes 3.1–3.3 a "success"):**

1. **Reprojection error.** Median < 5 px after Umeyama in both views; > 10 px
   means coordinate-frame mismatch or wrong-hand pairing.
2. **Wrist agreement.** Fused mesh's wrist position matches the directly-
   triangulated wrist (minimal Phase 3) to within 1 mm — same point, two
   ways.
3. **Visual.** Fused-mesh AR overlay video shows fingertips landing on the
   same physical fingertip in both views (which is *not* reliable in
   `wilor_ar_overlay.py`'s independent per-view meshes).

**Risks and mitigations:**

- *Fingertip triangulation noise* — weighted Umeyama (palm joints heavier).
- *Cross-view chirality flip* (left view sees "right hand", right view sees
  "left hand" for the same physical hand) — known issue, document for now,
  fix by relabeling chirality based on rectified wrist X position.
- *Outlier hand pairings* — if Umeyama residual > 2 cm, fall back to wrist-
  only fusion (the minimal Phase 3 result) and flag the frame.
- *WiLoR's `pred_keypoints_3d` frame convention* — assumed root-centered
  MANO local. Will verify at runtime by checking `kp3d[0] ≈ 0`.

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
| `scripts/dualstream.py` | Pi-side capture, unchanged |
| `scripts/make_calibration_board.py`, `scripts/calibrate.py` | Provide the calibration `.npz` that fixes stereo scale — unchanged |
| `scripts/triangulate.py` (MediaPipe stereo) | Kept as **fast preview path** (~37 fps) and as a **regression baseline** — wrist trajectories should agree with WiLoR-stereo |
| `scripts/wilor_stereo_demo.py` | The **minimal Phase 3** reference (wrist-only stereo). Stays as the regression baseline for the new fused pipeline |
| `scripts/wilor_ar_overlay.py` | The **per-view monocular** AR overlay. Stays as the regression baseline for visual mesh consistency |
| `scripts/wilor_phase3.py` | New for the **full Phase 3**: triangulates all 21 landmarks, fuses MANO mesh via Umeyama, renders the same mesh into both views |
| `scripts/inspect_3d.py` | Extended in phase 4 for MANO trajectory plots |
| `scripts/process.py` (stitching) | No longer used in main pipeline; reference only |

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
  **Default: subdir.** *(decided)*
- Data format: `.npz` (simple) vs `.h5` (better for big datasets).
  **Default: `.npz` until total dataset > ~1 GB.**
- Whether to keep MediaPipe pipeline running alongside WiLoR for every clip,
  or only as needed. **Default: alongside, since it's cheap and useful as
  regression check.**

### Open after Phase 2 + minimal Phase 3

- (a) **Proper Phase 3 — full mesh scale fusion.** *In progress, see the
  "Phase 3 (full)" section above.* Triangulate all 21 landmarks, fit
  similarity transform via Umeyama, render same mesh into both views.
- (b) **Better hand-to-hand matching.** Replace the greedy single-landmark
  match with a multi-landmark Hungarian assignment (use 3–4 stable
  landmarks per hand, not just the wrist). Specifically helps the
  "stereo paired but rejected" cases where two hands are gripping the
  same object and their wrists are at near-identical Y in rectified space.
  Becomes more relevant once Phase 3 is producing reliable per-frame meshes.
- (c) **Run on the 60-s "first minute" clip.** Same pipeline, just longer
  footage with much more varied interactions. ~30 min wall time. Good
  for evaluating the demo; doesn't change quality.

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
  `cameramount/wilor/wilor/` both have no `__init__.py`. If the repo root
  (cameramount/) is on sys.path, Python merges them into a single namespace
  package and `wilor.models` resolves ambiguously (`cameramount/wilor/models/`
  is where the user uploaded MANO files, which isn't a Python package).
  After the `scripts/` move, the repo root is no longer on sys.path
  implicitly; the wilor scripts just insert WILOR_DIR.
- **Left-hand 2D keypoints come out in flipped-crop coords** —
  `wilor/wilor/datasets/vitdet_dataset.py:62` flips the input crop
  horizontally for left hands (`flip = right == 0`). The network's
  `pred_keypoints_2d` and `pred_vertices` therefore come out in flipped
  coords. To map back to the original image / left-hand anatomy, multiply
  the X coordinate by `-1` for left hands BEFORE applying
  `box_center + box_size`. Without this, the rendered skeleton sits at
  mirrored X positions inside the bbox — looks vaguely hand-shaped (because
  hands are roughly symmetric) but anatomically wrong (thumb and pinky
  swapped), and biases the triangulated wrist X toward the bbox center.
  Bug found and fixed during the stereo demo run; see `wilor_stereo_demo.py`
  and `wilor_sanity.py`.
- **YOLO/WiLoR handedness labels are unreliable on egocentric footage** —
  same finding as MediaPipe (documented in README's "Other known
  limitations"). The same physical hand can be labelled "right" in one view
  and "left" in the other. Doesn't affect 3D positions; only the labels.
  Cheap fix: relabel based on which side of the frame the wrist sits on.
  Not blocking for Phase 3.

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
