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
- **24° toe-out** between the two cameras (12° per side off-centre). This
  geometry gives a wide combined FOV with overlap in the central region where
  manipulation happens.
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
| 3 | **Stereo calibration + rectification** | Unlock metric depth for everything below | **Next** |
| 4 | 3D fingertip triangulation | Per-finger positions in centimetres | Pending stage 3 |
| 5 | Object segmentation / tracking (SAM 2) | Know which pixels belong to the object | Not started |
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

### Observations from the annotated outputs

- **Drop rate is ~2–3 % overall**, concentrated in moments where the hand is
  partially out of frame or only a few fingers (e.g. thumb + index + middle)
  are visible. The detector seems to either drop the hand or hallucinate a
  plausible-but-wrong skeleton in those frames. This is the expected failure
  mode for any 2D keypoint detector on partial views, and is one of the things
  stereo + 3D will help with.
- **Strong success case**: when the hand is wholly visible — even with heavy
  inter-finger occlusion (both hands gripping a tube of grease, fingers wrapped
  around) — keypoints stay locked at 0.94–0.99 confidence on both cameras.
- This is good enough to call stage 1 a **proof of concept**: yes, fingers are
  individually trackable from each camera during close-proximity object
  interaction.

### Known limitations to keep in mind

- **Handedness labels are unreliable** on egocentric video. MediaPipe was
  trained primarily on selfie/forward-facing data, so its `Left` / `Right`
  classification can flip on top-down head-mount input. The keypoint *positions*
  themselves are fine. We don't need correct L/R labels for any of stages 3–7,
  so this is parked for now. Cheap fix: relabel based on which side of the
  frame the wrist sits on. Proper fix: swap in **HaMeR** or **WiLoR** (both
  trained on egocentric datasets like Ego4D and EPIC-Kitchens) — that's stage 8.
- **Parallax artefacts in the stitched panorama** — close-foreground hands sit
  at a different depth than background and can't be aligned by a single planar
  homography. Inherent to the geometry, not a bug. Going away once we move to
  stereo + depth.

## Repo layout

```
inputs/    Tracked test material — filenames dated by capture date.
outputs/   Tracked generated artefacts — filenames dated by generation date.
process.py Stitching pipeline.
hands.py   MediaPipe HandLandmarker sanity-check pipeline.
dated.py   today_pretty() helper for human-readable dated filenames.
*.scad/.stl/.3mf    Head-mount CAD.
```

Filenames look like `25th April 2026 - cam0 hands annotated.mp4`. Re-running a
script on a new day produces a new dated artefact rather than overwriting
yesterday's, so progress is visually browsable from `ls`.

**Not in git** (over GitHub's 100 MB limit; kept locally only):
`cam0.mp4`, `cam1.mp4` (144 MB raw 4-min recordings each), the full
~300 MB stitched panorama, raw `.h264` streams, and the redownloadable
MediaPipe `.task` model. Switch to **git-lfs** if any of these need to be
tracked.

## What's next

**Stage 3 — stereo calibration.** This is the single highest-leverage step
left. Procedure:

1. Print a checkerboard or ChArUco target (A4 print, mounted flat).
2. Capture ~30 synchronised image pairs of the target at varied poses (tilted,
   close, far, off-centre — fill the FOV).
3. Run `cv2.stereoCalibrate` to recover:
   - Per-camera intrinsics (focal length, principal point, distortion).
   - Extrinsics between the two cameras (rotation + translation; should
     come out near 24° as a sanity check).
4. Compute rectification maps with `cv2.stereoRectify` so the two streams can
   be aligned row-by-row.

Once calibrated, every later stage gets easier: the existing 2D MediaPipe
keypoints can be triangulated to 3D fingertip positions, dense depth maps
become available for object analysis, and the parallax problems in the
stitched panorama go away.

Time estimate: ~30 minutes of capture + ~50 lines of Python, one-time.

After calibration we'll come back to this list and pick stage 4 (triangulate
the keypoints) or stage 5 (object segmentation) depending on what the
calibrated data shows.

## Commit history (running log)

- `c9810b2` — initial object detection from both different cameras
- `5301032` — organise repo into inputs/ and outputs/ with readable dated filenames
