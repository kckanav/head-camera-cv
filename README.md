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

- 2× **Raspberry Pi Camera Module 3** mounted on a custom 3D-printed head
  bracket, tilted downward toward the wearer's hands.
- **Designed 24° toe-out** between the cameras (12° per side off-centre).
  Stereo calibration **recovered 18.9°** — print tolerances on the bracket.
  All math now uses the calibrated value, not the design value.
- **Recovered baseline**: 41.1 mm camera separation.
- **Raspberry Pi 5** for capture; recording at 1280×720, 30 fps, H.264.
- CAD source in `dual_picam3_holder.scad`; v2 STL/3MF is the current revision.

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
| 8 | Higher-fidelity hand model (HaMeR / WiLoR) | Full MANO mesh per phalanx if MediaPipe limits us | Defer until needed |

## What we've achieved so far

### Stitching pipeline (`process.py`)

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

### Hand detection sanity check (`hands.py`)

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

### Stereo calibration (`make_calibration_board.py`, `calibrate.py`)

Stage 3 needed two pieces:

**1. A calibration target.** `make_calibration_board.py` generates a 9×6
ChArUco board PDF (30 mm squares, 22 mm 5×5 ArUco markers, A4 landscape @
600 DPI) with a 100 mm scale bar in the margin to verify print scale with a
ruler.

**2. The calibration itself.** `calibrate.py` takes synchronised left/right
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

### Sparse 3D hand triangulation (`triangulate.py`, `inspect_3d.py`)

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

`inspect_3d.py` is a small companion that loads the npz and plots wrist
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
process.py                   Stitching pipeline (panorama video).
hands.py                     MediaPipe HandLandmarker sanity-check (stage 1).
make_calibration_board.py    ChArUco PDF generator (stage 3 input).
calibrate.py                 Stereo calibration from board videos (stage 3).
triangulate.py               Sparse 3D hand triangulation (stage 4).
inspect_3d.py                Trajectory plot of triangulated hand data.
dated.py                     today_pretty() helper for dated filenames.
inputs/                      Tracked test material — filenames dated by capture.
outputs/                     Tracked generated artefacts — filenames dated by generation.
*.scad / *.stl / *.3mf       Head-mount CAD.
```

Filenames look like `27th April 2026 - stereo hands annotated.mp4`. Re-running
a script on a new day produces a new dated artefact rather than overwriting
yesterday's, so progress is visually browsable from `ls`.

**Not in git** (over GitHub's 100 MB limit; kept locally only):
`cam0.mp4`, `cam1.mp4` (144 MB raw 4-min recordings each), the full
~300 MB stitched panorama, raw `.h264` streams, and the redownloadable
MediaPipe `.task` model. Switch to **git-lfs** if any of these need to be
tracked.

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
- this branch — stereo calibration (stage 3), sparse 3D hand triangulation (stage 4)
