# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Hardware + CV prototype for an egocentric hand-tracking rig:

- **Rig**: head-mounted dual-camera mount, two Raspberry Pi Camera Module 3 sensors on a Raspberry Pi 5, tilted downward toward the wearer's hands. The cameras were *designed* to toe out **24°** from each other, but stereo calibration recovered **18.9°** (print tolerances on the bracket). The recovered baseline is **41.1 mm**. **Use the calibrated geometry from `outputs/<date> - stereo calibration.npz`, not the design value, for any stereo math.**
- **Goal**: capture the wearer's hand movements and hand-object interactions, then run a CV pipeline. Stages 1 (per-camera MediaPipe), 3 (stereo calibration), and 4 (sparse 3D triangulation of MediaPipe keypoints) are done. Stage 5 (SAM 2 object segmentation) is the planned next step. Full ladder is in the README.
- **Captured frames**: `inputs/24th April 2026 - photo cam0.jpg` (right) and `inputs/24th April 2026 - photo cam1.jpg` (left) are reference single-frame captures from the actual rig. **Convention used everywhere in this repo: cam1 = left, cam0 = right.**
- **CAD**: `dual_picam3_holder.scad` (OpenSCAD source) and exported `.stl` / `.3mf` files (`_v2` is the current revision). The CAD only matters when the physical geometry is in question — usually it isn't.

## Repo layout (only the non-obvious bits)

- `process.py` — stitching pipeline (panorama video). See architecture notes below.
- `hands.py` — MediaPipe HandLandmarker sanity-check pipeline (stage 1).
- `make_calibration_board.py` — generates the print-ready ChArUco board PDF used as the calibration target.
- `calibrate.py` — stereo calibration from videos of the board (stage 3). Writes intrinsics, extrinsics, rectification maps, and `Q` to an `.npz`.
- `triangulate.py` — sparse 3D hand triangulation (stage 4). Loads the calibration, rectifies both streams, runs MediaPipe per view, pairs hands by epipolar (rectified-row) proximity, triangulates 21 landmarks per matched hand. Writes an annotated SBS video and a per-frame `(N, 2, 21, 3)` landmarks `.npz`.
- `inspect_3d.py` — quick matplotlib visualisation of the triangulated `.npz`: wrist depth + pinch aperture over time.
- `dated.py` — tiny helper: `today_pretty()` returns a string like `27th April 2026`. All scripts use this so generated artifacts land in `outputs/` with human-readable dated filenames; re-running the same script on a new day produces a new file rather than overwriting.
- `inputs/` — tracked test material (reference photos, short clips, calibration footage). Filenames are dated with the **capture** date.
- `outputs/` — tracked generated artifacts. Filenames are dated with the **generation** date.
- `.venv/` — local Python 3.9 virtualenv. `opencv-python`, `numpy`, `mediapipe`, `matplotlib`, `Pillow` already installed. Use it directly; don't recreate.

## Files NOT in git (too large or redownloadable)

- `cam0.mp4`, `cam1.mp4` — full 4-min recordings, ~144 MB each (over GitHub's 100 MB limit). Kept locally in repo root; `process.py` reads them from there. Switch the repo to git-lfs if you need to track them.
- `*.h264` — raw streams from the Pi, same content as the mp4s. Mux to mp4 with `ffmpeg -framerate 30 -i <file>.h264 -c copy <file>.mp4` if you need a container with proper timing (raw h264 has no framerate metadata, so `ffprobe` defaults to 60).
- `outputs/*stitched panorama.mp4` — the full stitched video is ~300 MB. We commit a sample frame instead; regenerate the video locally with `process.py`.
- `hand_landmarker.task` — MediaPipe model, re-downloadable from Google's mediapipe-models bucket.

## Running things

```bash
.venv/bin/python process.py           # stitching pipeline (panorama)
.venv/bin/python hands.py             # per-camera hand detection sanity check
.venv/bin/python make_calibration_board.py   # regenerate the ChArUco PDF
.venv/bin/python calibrate.py         # stereo calibration from board videos in inputs/
.venv/bin/python triangulate.py       # sparse 3D hand triangulation; writes SBS video + .npz
.venv/bin/python inspect_3d.py        # plot wrist depth + pinch over time from the .npz
```

`triangulate.py` and `inspect_3d.py` `subprocess.run(["open", ...])` their output on macOS so the result pops open after the run.

## Pipeline architecture

### `process.py` — stitching

Single linear script. Estimates **one** homography from a representative
mid-clip frame and reuses it for every frame, since the cameras are rigidly
mounted. Per-frame pipeline:

1. Load both frames with `cv2.imread`.
2. **SIFT** keypoints + descriptors on each.
3. **BFMatcher** with `NORM_L2`, `knnMatch(k=2)`, **Lowe's ratio test** at 0.75.
4. Estimate homography `cam0 → cam1` with `cv2.findHomography(..., cv2.RANSAC, 4.0)`.
5. Warp `cam0` onto a canvas sized to fit both views, then paste `cam1` in.
   "Blend" is just an overwrite — no feathering or multi-band blending.

Things to keep in mind when modifying:

- The cameras toe out ~19° (calibrated), so the overlap region between cam0 and cam1 is narrow and near the centerline. Match counts will be lower than for a typical panorama; tune the Lowe ratio / RANSAC threshold accordingly rather than assuming the matcher is broken.
- Homography assumes a roughly planar scene or pure rotation. For close-up hands/objects (which is the actual use case), it's a known-imperfect model — expect parallax artefacts and don't treat ghosting as a code bug. **The real pipeline doesn't stitch — it triangulates.** Stitching is kept as a debug view.

### `calibrate.py` — stereo calibration

Reads `inputs/<date> - cam{0,1} calibration.mp4` (synchronised left/right videos of the ChArUco board moved by hand in front of a tripod-mounted rig). Per-camera intrinsics with `cv2.calibrateCamera`, then stereo extrinsics with `cv2.stereoCalibrate(..., CALIB_FIX_INTRINSIC)`, then `cv2.stereoRectify(..., alpha=1)`.

**`alpha=1` is intentional**: it preserves the full source FOV in the rectified output (with black padding where rectification has no source pixel). `alpha=0` cropped to the largest all-valid rectangle and clipped the top of the frame — exactly where hands held high need to be visible. Don't switch back without good reason.

Outputs everything (K_l/K_r, dist_l/dist_r, R, T, E, F, R1/R2, P1/P2, Q, remap tables, plus the rms numbers and recovered angle/baseline) to `outputs/<date> - stereo calibration.npz`. Renders a side-by-side rectified sample with green scanlines as a visual check — corresponding points must lie on the same scanline.

### `triangulate.py` — sparse 3D hand keypoints

Per frame pair:

1. `cv2.remap` both frames using the precomputed maps from the calibration `.npz`.
2. Run MediaPipe HandLandmarker on each rectified view (two separate landmarker instances — VIDEO mode wants monotonic timestamps, so they can't share state).
3. Pair left/right hand detections by **wrist-row proximity** — the epipolar constraint after rectification is "same image row". Tolerance is ±18 px.
4. `cv2.triangulatePoints(P1, P2, lpts.T, rpts.T)` per matched hand → `(21, 3)` metric 3D landmarks.
5. **Sanity-filter**: drop any pair whose triangulated wrist Z is outside `[0.10, 2.00]` m. This catches wrong-hand pairings (when both hands have similar Y) that the row check lets through.
6. Sort kept hands by left-view wrist X for slot stability across frames.
7. Annotate both rectified views with the skeleton, wrist depth (cm), and thumb-index pinch aperture (mm), write side-by-side to `outputs/<date> - stereo hands annotated.mp4`.
8. Save NaN-padded `(N_frames, 2, 21, 3)` landmarks (plus 2D coords) to `outputs/<date> - stereo hand 3d.npz`.

Known residual: per-fingertip Z is noisier than the wrist; a 1-pixel disparity error on a fingertip gives a large depth error. The wrist Z is solid; pinch aperture has occasional outliers > 150 mm. Per-landmark Z-clamping or short temporal smoothing would clean it up.
