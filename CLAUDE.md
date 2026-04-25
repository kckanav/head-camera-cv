# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Hardware + CV prototype for an egocentric hand-tracking rig:

- **Rig**: head-mounted dual-camera mount, two Raspberry Pi Camera Module 3 sensors on a Raspberry Pi 5, tilted downward toward the wearer's hands. The cameras are toed out **24°** from each other (12° per side off-center) — this is the assumed geometry for any stereo/stitching math.
- **Goal**: capture the wearer's hand movements and hand-object interactions, then run a CV pipeline (currently: image stitching; longer term: hand pose / interaction detection).
- **Captured frames**: `photo_cam0.jpg` (left) and `photo_cam1.jpg` (right) are reference captures from the actual rig and are the canonical inputs while developing the stitching code locally. `test_cam1.jpg` is an older single-camera test capture.
- **CAD**: `dual_picam3_holder.scad` (OpenSCAD source) and exported `.stl` / `.3mf` files (`_v2` is the current revision). The CAD only matters when the physical geometry is in question — usually it isn't.

## Repo layout (only the non-obvious bits)

- `process.py` — stitching pipeline (panorama video).
- `hands.py` — MediaPipe HandLandmarker sanity-check pipeline.
- `dated.py` — tiny helper: `today_pretty()` returns a string like `25th April 2026`. All scripts use this so generated artifacts land in `outputs/` with human-readable dated filenames; re-running the same script on a new day produces a new file rather than overwriting.
- `inputs/` — tracked test material (reference photos and short clips). Filenames are dated with the **capture** date.
- `outputs/` — tracked generated artifacts. Filenames are dated with the **generation** date.
- `.venv/` — local Python 3.9 virtualenv. `opencv-python`, `numpy`, `mediapipe` already installed. Use it directly; don't recreate.

## Files NOT in git (too large)

- `cam0.mp4`, `cam1.mp4` — full 4-min recordings, ~144 MB each (over GitHub's 100 MB limit). Kept locally in repo root; `process.py` reads them from there. Switch the repo to git-lfs if you need to track them.
- `*.h264` — raw streams from the Pi, same content as the mp4s.
- `outputs/*stitched panorama.mp4` — the full stitched video is ~300 MB. We commit a sample frame instead; regenerate the video locally with `process.py`.
- `hand_landmarker.task` — MediaPipe model, re-downloadable from Google's mediapipe-models bucket.

## Running the stitching pipeline

```bash
.venv/bin/python proecss.py
```

The script as committed was authored in Google Colab and imports `from google.colab.patches import cv2_imshow`, which **will not run locally**. To run on macOS/Linux, replace the Colab-only display call with a regular `cv2.imshow(...)` (and drop the `google.colab` import). The user has explicitly asked for this to work locally, so when editing this file it's expected to make that swap.

## Pipeline architecture (`proecss.py`)

Single linear script, no functions/classes:

1. Load both frames with `cv2.imread`.
2. **SIFT** keypoints + descriptors on each.
3. **BFMatcher** with `NORM_L2` + `crossCheck=True`, sorted by distance.
4. Visualize top 50 matches via `cv2.drawMatches`.
5. Estimate **homography** `img1 → img2` with `cv2.findHomography(..., cv2.RANSAC, 5.0)`.
6. Warp `img2` onto a `(w1 + w2) × h1` canvas with `cv2.warpPerspective`, then paste `img1` into the top-left. Final "blend" is just an overwrite — no feathering or multi-band blending yet.

Things to keep in mind when modifying:

- The 24° toe-out means the overlap region between cam0 and cam1 is narrow and near the centerline. Match counts will be lower than for a typical panorama; tune `crossCheck` / Lowe ratio / RANSAC threshold accordingly rather than assuming the matcher is broken.
- Homography assumes a roughly planar scene or pure rotation. For close-up hands/objects (which is the actual use case), a homography is a known-imperfect model — expect parallax artifacts and don't treat ghosting as a code bug.
- Current code uses **all** matches (not a ratio-test filtered subset) for homography estimation; only the top 50 are used for *visualization*. RANSAC is what's filtering outliers.
