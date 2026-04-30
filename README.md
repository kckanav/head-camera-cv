# Egocentric hand-and-object tracking on a head-mounted dual Pi Cam rig

## What we're building

A wearable computer-vision pipeline that captures the wearer's hands from their
own point of view and analyses **how they interact with objects** — finger
positions, arm pose, grip events, object orientation, and (eventually) the full
3D geometry of the interaction. Pure viewing first, neural-network analysis
later.

The key design choice is having **two cameras** instead of one. A single
egocentric camera can detect hands but has to *guess* depth from a flat image;
two cameras let us *measure* depth by triangulation. Most published egocentric
methods are monocular and have to work around that — we won't have to.

## Hardware

- 2× **Raspberry Pi Camera Module 3** (IMX708 sensor, 12 MP) mounted on a
  custom 3D-printed head bracket, tilted downward toward the wearer's hands.
- **Designed 24° toe-out** between the cameras (12° per side off-centre).
  Stereo calibration **recovered 18.9°** — print tolerances on the bracket.
  All math now uses the calibrated value, not the design value.
- **Recovered baseline**: 41.1 mm camera separation.
- **Raspberry Pi 5** for capture. Two sensor modes are exposed by the
  live-preview UI (`scripts/pi/dualstream.py`); both record at **1280×720
  H.264 at a hard-locked 30 fps**:
  - **2304×1296 binned full-FOV mode** — ~100° HFOV. **Current default**
    for the wide-FOV captures used by the WiLoR pipeline. Sensor caps at
    ~56 fps so 30 fps is comfortable.
  - **1536×864 centre-cropped mode** — ~75° HFOV. Older default; still
    used for the narrow-FOV reference clips and calibration.
  The 4608×2592 full-resolution mode is also offered by the UI but caps
  at ~14 fps so it can't sustain 30 fps recording — not used.
- Switching sensor mode changes intrinsics, so a fresh calibration is
  required after any switch. We keep both narrow-FOV (`""`) and wide-FOV
  (`" wide"`) calibrations side-by-side in `outputs/`.
- CAD source in `cad/dual_picam3_holder.scad`; v2 STL/3MF is the current revision.

## Pipeline plan (staged ladder)

The plan is to grow capability one stage at a time, validating each on real
footage before adding the next. Stages are roughly in order of "what unlocks
the most when you do it next":

| # | Stage | Purpose | Status |
|---|-------|---------|--------|
| 1 | Per-camera hand keypoints (MediaPipe) | Prove fingers are trackable on real footage | **Done** |
| 2 | Arm / body pose | Wrist-elbow-shoulder context | Not started |
| 3 | Stereo calibration + rectification | Unlock metric depth for everything below | **Done** |
| 4 | Sparse 3D fingertip triangulation | Per-finger positions in centimetres | **Done** |
| 5 | Object segmentation / tracking (SAM 2) | Know which pixels belong to the object | **Next** |
| 6 | Grip event detection (heuristic first) | "Is the hand currently holding it?" | Not started |
| 7 | Object 6-DoF pose | Orientation, not just position | Optional / later |
| 8 | Dexterous hand capture (WiLoR + stereo) | Full MANO mesh + joint angles, robot-retargeting-ready | **In progress** — Phases 0-3 done, canonical pipeline at `scripts/pipeline/08_wilor_canonical.py`. Phase 4 (table-frame anchoring) and 5 (robot-training export) are next. See [PLAN.md](PLAN.md). |

## What we've achieved so far

### Stitching pipeline (`scripts/viz/stitch_panorama.py`)

Earlier the goal was to combine the two camera streams into one image. Started
as a Colab snippet (Lowe-less SIFT, Colab-only display, wrong-direction
homography); rewrote it to run locally with Lowe's ratio test, RANSAC, and
correct canvas math.

A homography is estimated **once** from a representative mid-clip frame and
reused for every frame — the cameras are rigidly mounted, so per-frame
estimation would be wasted compute. Output: `outputs/<date> - stitched
panorama.mp4` (~300 MB, gitignored; sample frame committed).

Important finding: **stitching is the wrong tool for our actual goal**.
A homography collapses both views onto a single plane, which destroys exactly
the parallax that gives depth perception. The panorama is useful as a quick
debug view, but for the real pipeline we want the two views *preserved*, not
fused. This is what motivates the stereo-calibration step over a fancier
stitcher.

### Hand detection sanity check (`scripts/pipeline/01_per_cam_sanity.py`)

Ran **MediaPipe HandLandmarker (Tasks API)** on a 15-second clip from each
camera (`1:40 – 1:55` of the 4-minute test recording).

Results on `25th April 2026` test clips:

| Camera | Frames | Hand-detected | Rate |
|--------|--------|---------------|------|
| cam0 | 451 | 449 | **99.6 %** |
| cam1 | 451 | 420 | **93.1 %** |

Inference runs at ~37 fps on M2 CPU — comfortably faster than real-time, so it
should work on the Pi 5 too.

#### Observations from the annotated outputs

- **Drop rate is ~2–3 % overall**, concentrated in moments where the hand is
  partially out of frame or only a few fingers (e.g. thumb + index + middle)
  are visible. The detector seems to either drop the hand or hallucinate a
  plausible-but-wrong skeleton in those frames. This is the expected failure
  mode for any 2D keypoint detector on partial views, and is one of the things
  stereo + 3D will help with.
- **Strong success case**: when the hand is wholly visible — even with heavy
  inter-finger occlusion (both hands gripping a tube of grease, fingers wrapped
  around) — keypoints stay locked at 0.94–0.99 confidence on both cameras.

### Stereo calibration (`scripts/calibration/make_charuco_board.py`, `scripts/calibration/calibrate_stereo.py`)

Stage 3 needed two pieces:

**1. A calibration target.** `make_charuco_board.py` generates a 9×6
ChArUco board PDF (30 mm squares, 22 mm 5×5 ArUco markers, A4 landscape @
600 DPI) with a 100 mm scale bar in the margin to verify print scale with a
ruler.

**2. The calibration itself.** `calibrate_stereo.py` takes synchronised left/right
videos of the board moved by hand in front of a tripod-mounted rig (~40 s of
footage at 30 fps), then:

- Samples one frame pair every 0.5 s (~80 pairs).
- Detects ChArUco corners independently in each view.
- Per-camera intrinsics with `cv2.calibrateCamera`.
- Stereo extrinsics with `cv2.stereoCalibrate(..., CALIB_FIX_INTRINSIC)`.
- Rectification with `cv2.stereoRectify(..., alpha=1)`. `alpha=1` preserves
  the full source FOV (with black padding where rectification has no source
  pixel) — important on this rig because hands frequently sit near the top
  edge of the frame; the more-aggressive `alpha=0` cropped them off.
- Saves intrinsics, extrinsics, rectification rotations, projection matrices,
  `Q`, and `cv2.remap` lookup tables to `outputs/<date> - stereo
  calibration.npz`. Renders a side-by-side rectified sample with green
  scanlines as a visual check (corresponding points must lie on the same
  scanline).

Result on `27th April 2026` calibration capture (71/79 usable pairs):

| Metric | Result | Bar |
|---|---|---|
| cam1 intrinsic rms | 0.24 px | < 0.5 = good |
| cam0 intrinsic rms | 0.27 px | < 0.5 = good |
| Stereo rms | 0.26 px | < 1.0 = good |
| Recovered toe-out | **18.9°** | designed 24° |
| Recovered baseline | **41.1 mm** | matches the bracket |
| Recovered focal length | fx ≈ 803 px | matches Pi Cam 3 ~75° HFOV |

Calibration is well below the conservative quality threshold; the rectified
sanity image (`outputs/<date> - rectified pair sanity.jpg`) confirms scanline
alignment across both views.

Implied **depth precision** at hand-interaction range (`ΔZ = Z² · Δd / (f · B)`,
½-px matching):

| Distance | Disparity | Depth precision |
|---|---|---|
| 30 cm | 110 px | ~1.4 mm |
| 50 cm | 66 px | ~3.8 mm |
| 80 cm | 41 px | ~10 mm |

Single-digit-millimetre depth resolution at hand-interaction range — plenty
for grip aperture, fingertip positions, and "is the hand touching the object"
heuristics.

### Sparse 3D hand triangulation (`scripts/pipeline/04_triangulate_mp.py`, `scripts/viz/inspect_3d_mp.py`)

Stage 4 puts everything together:

1. Load the stereo calibration `.npz`.
2. Rectify each frame pair with the precomputed `cv2.remap` tables.
3. Run MediaPipe HandLandmarker independently on each rectified view.
4. Pair left/right hand detections by **wrist-row proximity** — after
   rectification the epipolar constraint is "same image row", so a true
   match must agree in y. ±18 px tolerance.
5. Triangulate the 21 landmarks per paired hand with
   `cv2.triangulatePoints(P1, P2, ...)` → 3D positions in metres in the
   left-rectified camera frame.
6. **Sanity-filter** any pair whose triangulated wrist Z falls outside
   `[10 cm, 200 cm]` — catches wrong-hand pairings the row check lets through.
7. Sort hand slots by left-view wrist X for frame-to-frame stability.
8. Save per-frame `(N_frames, 2, 21, 3)` 3D landmarks (NaN-padded) to
   `outputs/<date> - stereo hand 3d.npz`.
9. Write a side-by-side rectified video annotated with skeletons, wrist
   depth (cm), and pinch (thumb-index) aperture (mm).

`inspect_3d_mp.py` is a small companion that loads the npz and plots wrist
depth and pinch over time, one curve per hand slot.

Result on the first 60 s of the test recording (alpha=1 rectification):

| Metric | Result |
|---|---|
| Hands detected (cam1) | 1689 / 1801 = 94 % |
| Hands detected (cam0) | 1582 / 1801 = 88 % |
| Stereo-paired (after Z filter) | 1100 / 1801 = 61 % |
| Median wrist depth | 33.3 cm |
| Wrist depth 5–95 pct | 21.6 – 47.0 cm |
| Median pinch aperture | 36 mm |
| Inference rate | ~19 fps on M2 CPU (two MediaPipe instances) |

The 5–95 percentile wrist-depth band lies entirely within the expected
hand-interaction range. The trajectory plot shows clean steady-state
clustering at ~32 cm during periods of close manipulation, with both hands
tracked simultaneously.

**Known residual issue.** A single fingertip can occasionally triangulate
to a wildly wrong Z even when the wrist is solid — MediaPipe's per-finger
keypoints are noisier than the wrist, and a 1 px disparity error at that
landmark gives a large depth error. Visible as occasional pinch outliers
> 150 mm in the trajectory plot. Per-landmark Z-clamping or short-window
temporal smoothing would clean it up. Not blocking.

### Stage 8 in progress — WiLoR + stereo dexterous hand pipeline (`scripts/pipeline/08_wilor_canonical.py`)

The sparse 21-keypoint MediaPipe output is enough for *trajectories* but not
for retargeting to arbitrary robot grippers. To capture data that can drive
a parallel-jaw, 3-finger, or anthropomorphic robot hand from the same
recording, we moved to **WiLoR** ([CVPR 2025](https://github.com/rolpotamias/WiLoR)) —
an end-to-end ViT model that produces full **MANO** parameters (16 joint
rotations + 778-vertex mesh) per hand. Stereo gives us an unusual
advantage: WiLoR is monocular and has scale ambiguity, but our calibrated
stereo can fix the scale precisely.

Phases 0–3 are all done:

- **Phase 0–1** (env + sanity): WiLoR runs on Apple MPS on this M2 host
  (with the YOLO detector on CPU per ultralytics issue #4031). Sanity
  check on a real Pi-Cam frame: hand detected, mesh extent ~16 cm, 2D
  keypoints overlay cleanly on the actual hand.
  Entry point: `scripts/experiments/wilor_sanity.py`.
- **Phase 2** (per-view WiLoR + stereo wrist triangulation): metric wrist
  depth annotations on a side-by-side video. Median wrist depth ~34 cm,
  in line with what MediaPipe-stereo reported on similar footage
  (regression baseline holds).
  Entry point: `scripts/experiments/wilor_wrist_stereo.py`.
- **Phase 3 (full Umeyama)** — *reference experiment*: triangulate all 21
  keypoints, fit a 7-DOF similarity transform, project the same world-
  frame mesh into both views. 99.3% fuse rate, but median 2D reprojection
  error blew up to 10 px. **Worse than monocular WiLoR.** Kept for the
  negative-result record.
  Entry point: `scripts/experiments/wilor_phase3_umeyama.py`.
- **Phase 3 (anchored)** — *canonical pipeline*: per view, `cv2.solvePnP`
  fits 6-DOF pose to WiLoR's 2D keypoints; stereo wrist depth supplies
  a 1-DOF metric scale anchor `k = Z_stereo / t_pnp[2]`. Uniform 3D
  scaling preserves perspective 2D projection (`(X,Y,Z) → (fX/Z, fY/Z)`
  is invariant under `(kX, kY, kZ)`), so the rendered overlay matches
  the per-view monocular baseline (median PnP residual ~2 px) while
  the underlying mesh is metric. The scale anchor uses stereo only for
  the one thing it can give that monocular cannot.
  Entry point: `scripts/pipeline/08_wilor_canonical.py`. Has full
  CUDA/MPS device portability via `scripts/_lib/device.py` plus a
  `--bench` flag for per-stage perf tracing.

Discovered and fixed along the way:

- **Left-hand keypoint mirroring**: WiLoR's `ViTDetDataset` flips left-hand
  input crops, so `pred_keypoints_2d` and `pred_vertices` come out in
  flipped-crop coords. Multiplying X by −1 before mapping back to image
  coords fixes anatomically-flipped renders.
- **Centralised WiLoR runtime workarounds** in `scripts/_lib/wilor_setup.py`:
  pyrender stub (no EGL on macOS), `torch.load weights_only=False` patch
  (PyTorch 2.6 broke YOLO ckpt loading), and the wilor sys.path / chdir
  setup. Every wilor script now has a single `import wilor_setup` line
  instead of 60+ lines of boilerplate.

Full plan, install gotchas, and Phases 4–5 (visualisation + table-frame
robot-training export) are in [`PLAN.md`](PLAN.md).

### Other known limitations

- **Handedness labels are unreliable** on egocentric video. MediaPipe was
  trained primarily on selfie/forward-facing data, so its `Left` / `Right`
  classification can flip on top-down head-mount input. The keypoint
  *positions* themselves are fine; we side-step handedness by pairing hands
  across views via the rectified-row epipolar constraint instead. Cheap fix
  if we ever need it: relabel based on which side of the frame the wrist
  sits on. Proper fix: swap in **HaMeR** or **WiLoR** (both trained on
  egocentric datasets like Ego4D and EPIC-Kitchens) — that's stage 8.
- **Parallax artefacts in the stitched panorama** — close-foreground hands
  sit at a different depth than background and can't be aligned by a single
  planar homography. Inherent to the geometry, not a bug. Going away
  because we're not stitching anymore — the calibrated stereo pipeline
  preserves both views.

## Repo layout

```
README.md / PLAN.md / CLAUDE.md   Docs (read CLAUDE.md before contributing).

scripts/                          All pipeline entry points (run from repo root).
  _lib/                           Shared imports (sys.path-injected by each script).
    dated.py                      today_pretty() helper.
    device.py                     pick_device / configure_perf — CUDA/MPS/CPU portability.
    wilor_setup.py                pyrender stubs, torch.load patch, wilor sys.path/chdir.
  pi/
    dualstream.py                 Pi-side dual capture + live HTML preview UI.
  calibration/
    make_charuco_board.py         ChArUco PDF generator (stereo calibration target).
    make_aruco_marker.py          ArUco PDF generator (table-frame anchor for stage 8 phase 4-5).
    calibrate_stereo.py           Stereo calibration from board videos (stage 3).
  pipeline/                       The canonical pipeline, in run order.
    01_per_cam_sanity.py          MediaPipe HandLandmarker per-camera sanity (stage 1).
    04_triangulate_mp.py          Sparse 3D hand triangulation via MediaPipe + stereo (stage 4).
    08_wilor_canonical.py         WiLoR + stereo metric depth (stage 8). --bench for perf trace.
  viz/                            Visualisation + debug viewers.
    play_stereo.py                SBS rectified playback with on-the-fly --alpha.
    depth_dense.py                Dense stereo depth (overlay / turbo / both).
    inspect_3d_mp.py              Trajectory plot of triangulated MediaPipe hand data.
    stitch_panorama.py            Old SIFT/RANSAC panorama stitcher (debug-only).
    wilor_ar_monocular.py         Per-view monocular AR overlay (visual baseline).
  experiments/                    Reference scripts kept for the record.
    wilor_sanity.py               Single-image WiLoR sanity check (stage 8 phase 1).
    wilor_wrist_stereo.py         Per-view WiLoR + stereo wrist-only triangulation (phase 2).
    wilor_phase3_umeyama.py       Phase 3 (full Umeyama) — superseded; kept as negative-result record.

cad/                              Head-mount CAD (.scad source + .stl / .3mf exports).
inputs/                           Tracked test material — filenames dated by capture.
outputs/                          Tracked generated artefacts — filenames dated by generation.

raw/                              (gitignored) Raw camera mp4 + h264 streams from the Pi.
models/                           (gitignored) Re-downloadable model weights (e.g. hand_landmarker.task).
wilor/                            (gitignored) Cloned WiLoR repo — large weights, separate license.
.venv/                            (gitignored) Python 3.9 venv for stages 1-4.
.venv-hamer/                      (gitignored) Python 3.10 venv for stage 8.
```

Run scripts from the repo root so the relative paths in `inputs/` /
`outputs/` / `models/` resolve correctly:
`.venv/bin/python scripts/calibration/calibrate_stereo.py` or
`.venv-hamer/bin/python scripts/pipeline/08_wilor_canonical.py`.

Filenames in `inputs/` and `outputs/` look like `27th April 2026 - stereo
hands annotated.mp4`. Re-running a script on a new day produces a new
dated artefact rather than overwriting yesterday's, so progress is visually
browsable from `ls outputs/`.

**Not in git** (over GitHub's 100 MB limit, redownloadable, or
license-restricted; kept locally only): everything in `raw/`, `models/`,
`wilor/`, and the full stitched panorama / 60-s WiLoR demo renders in
`outputs/`. Switch to **git-lfs** if any of these need to be tracked.

## What's next

Stages 3 and 4 are done — we now have calibrated stereo and a working sparse
3D hand pipeline. Two reasonable directions:

**Stage 5 — object segmentation (SAM 2).** Identify which pixels belong to
the held object so we can ask "which finger is touching what part of it?".
Combines naturally with the existing 3D hand keypoints to produce per-frame
hand–object contact features.

**Refine stage 4.** Per-landmark Z-clamping and short-window temporal
smoothing to clean up the fingertip outliers; identity tracking so hand
slots stay consistent when hands physically cross; running MediaPipe on the
*raw* frames and projecting the keypoints through the rectification map
afterward, to claw back the small detection-rate gap from running on
rectified frames.

## Commit history (running log)

- `c9810b2` — initial object detection from both different cameras
- `5301032` — organise repo into inputs/ and outputs/ with readable dated filenames
- `02fef54` — README summarising goals, architecture, status, and next step
- `e63edb9` — hand detection on first minute of each camera
- `e008189` — generate ChArUco calibration board sized for the Pi Cam 3 rig
- `fa40c85` — stereo calibration (stage 3): script, capture footage, recovered geometry
- `402af8c` — sparse 3D hand triangulation (st	age 4): pipeline, results, trajectory plot
- `19c7ecd` — README + CLAUDE.md update for stages 3 and 4
- `9f17555` — stage 8 phase 0+1 (WiLoR setup + sanity)
- `5254d62` — stage 8 phase 2 + minimal phase 3 (per-view WiLoR + stereo wrist triangulation, with left-hand mirror bug fix)
- `d22400f` — stage 8 wide-FOV: dualstream sensor-mode UI, recalibration, demo CLI args, 60-s perf trace
- `582a596` — restructure root into scripts/ + cad/ + raw/ + models/ for navigability
