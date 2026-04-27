# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Hardware + CV prototype for an egocentric hand-tracking rig:

- **Rig**: head-mounted dual-camera mount, two Raspberry Pi Camera Module 3 sensors on a Raspberry Pi 5, tilted downward toward the wearer's hands. The cameras were *designed* to toe out **24°** from each other, but stereo calibration recovered **18.9°** (print tolerances on the bracket). The recovered baseline is **41.1 mm**. **Use the calibrated geometry from `outputs/<date> - stereo calibration.npz`, not the design value, for any stereo math.**
- **Goal**: capture the wearer's hand movements and hand-object interactions, then run a CV pipeline. Stages 1 (per-camera MediaPipe), 3 (stereo calibration), and 4 (sparse 3D triangulation of MediaPipe keypoints) are done. **Stage 8 (dexterous hand capture via WiLoR + stereo) is now in progress** — Phase 0 (env + weights) and Phase 1 (sanity check) are done. The plan's full state is in `PLAN.md`; read that first when working on stage 8. Full ladder is in `README.md`.
- **Captured frames**: `inputs/24th April 2026 - photo cam0.jpg` (right) and `inputs/24th April 2026 - photo cam1.jpg` (left) are reference single-frame captures from the actual rig. **Convention used everywhere in this repo: cam1 = left, cam0 = right.**
- **CAD**: `cad/dual_picam3_holder.scad` (OpenSCAD source) and exported `.stl` / `.3mf` files (`_v2` is the current revision). The CAD only matters when the physical geometry is in question — usually it isn't.

## Repo layout (only the non-obvious bits)

All entry points live in `scripts/`. Everything is run from the **repo root** so the relative `inputs/` / `outputs/` / `models/` paths inside the scripts resolve correctly.

- `scripts/process.py` — stitching pipeline (panorama video). See architecture notes below.
- `scripts/hands.py` — MediaPipe HandLandmarker sanity-check pipeline (stage 1).
- `scripts/make_calibration_board.py` — generates the print-ready ChArUco board PDF used as the calibration target.
- `scripts/calibrate.py` — stereo calibration from videos of the board (stage 3). Writes intrinsics, extrinsics, rectification maps, and `Q` to an `.npz`. Has a `TAG` constant near the top to keep narrow-FOV (`""`) and wide-FOV (`" wide"`) calibrations side-by-side in `outputs/`.
- `scripts/triangulate.py` — sparse 3D hand triangulation (stage 4). Loads the calibration, rectifies both streams, runs MediaPipe per view, pairs hands by epipolar (rectified-row) proximity, triangulates 21 landmarks per matched hand. Writes an annotated SBS video and a per-frame `(N, 2, 21, 3)` landmarks `.npz`.
- `scripts/inspect_3d.py` — quick matplotlib visualisation of the triangulated `.npz`: wrist depth + pinch aperture over time.
- `scripts/wilor_sanity.py` — stage 8 / Phase 1 sanity check. Loads WiLoR + YOLO + MANO, runs them on `inputs/24th April 2026 - photo cam0.jpg`, writes a 2D-keypoint overlay (`outputs/<date> - wilor sanity overlay.jpg`) and the MANO mesh as `.obj`. The script lives in `scripts/` (not inside `wilor/`) so it survives a re-clone of the gitignored WiLoR repo. Has several inline workarounds documented in its docstring (pyrender stub, `torch.load` patch, MPS float64 cast, namespace-package fix via WILOR_DIR-only on sys.path). Run via `.venv-hamer/bin/python scripts/wilor_sanity.py`.
- `scripts/wilor_stereo_demo.py` — stage 8 / Phase 2 + minimal Phase 3. Loads the stereo calibration `.npz`, runs WiLoR on each view's raw frames (MPS for ViT, CPU for YOLO), pairs hands across views by rectified-row proximity of the wrist, triangulates the wrist with `cv2.triangulatePoints(P1, P2, ...)` for metric depth, writes a side-by-side annotated video plus a per-frame `.npz`. Default calibration is the wide-FOV one; CLI args `--clip-left/--clip-right/--calib/--tag/--max-frames` override the presets, and a `from_root()` helper resolves user-supplied relative paths against the repo root (the script `chdir`s into `wilor/` for picamera2 config paths). Pass `--long` for the 60-s narrow clip preset. Only the **wrist** is triangulated here; full mesh scale fusion is the next chunk of Phase 3.
- `scripts/dualstream.py` — runs on the Pi 5. Discovers IMX708 sensor modes via `picamera2.sensor_modes`, exposes them as buttons in a live HTML preview UI, and locks capture to 30 fps via `FrameDurationLimits`. Default mode is 2304×1296 (full sensor, ~100° HFOV); the older 1536×864 cropped mode is ~75°. Lives on the Pi but is checked into the repo.
- `scripts/dated.py` — tiny helper: `today_pretty()` returns a string like `27th April 2026`. All scripts import this so generated artifacts land in `outputs/` with human-readable dated filenames; re-running the same script on a new day produces a new file rather than overwriting.
- `PLAN.md` — concrete plan for stage 8. Has a "pick-up-where-we-left-off" header at the top so a fresh session can resume. Update this whenever a phase finishes or a decision changes.
- `cad/` — OpenSCAD source + `.stl` / `.3mf` exports of the head bracket.
- `inputs/` — tracked test material (reference photos, short clips, calibration footage). Filenames are dated with the **capture** date.
- `outputs/` — tracked generated artifacts. Filenames are dated with the **generation** date.
- `raw/` — **gitignored**. Local-only raw camera recordings (`cam{0,1}.mp4`, `cam{0,1}_*.h264`). Don't add tracked content here.
- `models/` — **gitignored**. Re-downloadable model weights (currently just `hand_landmarker.task`). `hands.py` and `triangulate.py` reference it as `models/hand_landmarker.task`.
- `.venv/` — local Python 3.9 virtualenv used by stages 1–4. `opencv-python`, `numpy`, `mediapipe`, `matplotlib`, `Pillow` already installed. Use it directly; don't recreate.
- `.venv-hamer/` — separate Python 3.10 virtualenv used by stage 8. Has `torch` (with MPS), the WiLoR package's deps, plus `dill` and `Cython` (not in WiLoR's `requirements.txt`; needed for the YOLO ckpt and `xtcocotools` build respectively). Don't recreate; if you do, follow the Phase 0 commands in `PLAN.md`.
- `wilor/` — cloned WiLoR repo, **gitignored** (large weights, separate license). Contains `pretrained_models/{detector.pt, wilor_final.ckpt}` (downloaded from HuggingFace), `models/MANO_{RIGHT,LEFT}.pkl` (uploaded by the user under their MANO academic license), and `mano_data/MANO_{RIGHT,LEFT}.pkl` symlinked from the above (WiLoR's config expects them at `mano_data/`).

## Files NOT in git (too large, redownloadable, or license-restricted)

- `raw/cam0.mp4`, `raw/cam1.mp4` — full 4-min recordings, ~144 MB each (over GitHub's 100 MB limit). `process.py` expects them at the repo root, so update its paths if you regenerate them. Switch the repo to git-lfs if you need to track them.
- `raw/*.h264` — raw streams from the Pi, same content as the mp4s. Mux to mp4 with `ffmpeg -framerate 30 -i <file>.h264 -c copy <file>.mp4` if you need a container with proper timing (raw h264 has no framerate metadata, so `ffprobe` defaults to 60).
- `outputs/*stitched panorama.mp4` — the full stitched video is ~300 MB. We commit a sample frame instead; regenerate the video locally with `process.py`.
- `outputs/*wilor stereo demo 60s*.mp4` — the 60-s WiLoR demo render is ~150 MB. The `.npz` and the `wilor 60s perf trace.png` are tracked; re-render the video locally if needed.
- `models/hand_landmarker.task` — MediaPipe model, re-downloadable from Google's mediapipe-models bucket.

## Running things

All commands run from the **repo root**:

```bash
.venv/bin/python scripts/process.py           # stitching pipeline (panorama)
.venv/bin/python scripts/hands.py             # per-camera hand detection sanity check
.venv/bin/python scripts/make_calibration_board.py   # regenerate the ChArUco PDF
.venv/bin/python scripts/calibrate.py         # stereo calibration from board videos in inputs/
.venv/bin/python scripts/triangulate.py       # sparse 3D hand triangulation; writes SBS video + .npz
.venv/bin/python scripts/inspect_3d.py        # plot wrist depth + pinch over time from the .npz
.venv-hamer/bin/python scripts/wilor_sanity.py        # stage 8 sanity check (uses the .venv-hamer Python)
.venv-hamer/bin/python scripts/wilor_stereo_demo.py   # stage 8 stereo demo, defaults to 15-s grease clip + wide calib
```

`triangulate.py`, `inspect_3d.py`, `wilor_sanity.py`, and `wilor_stereo_demo.py` `subprocess.run(["open", ...])` their output on macOS so the result pops open after the run.

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

### `wilor_sanity.py` — stage 8 phase 1 sanity check

Uses the `.venv-hamer` Python 3.10 venv. Loads YOLO + WiLoR + MANO from
the gitignored `wilor/` directory, runs them on a real Pi-Cam frame, and
writes an OpenCV overlay + `.obj` mesh. Several inline workarounds are
documented in the script's docstring — they target this exact combination
of WiLoR @ rolpotamias/main + ultralytics 8.1.34 + torch 2.11 on macOS:

- Stub three pyrender-using submodules of `wilor.utils` (no EGL on macOS).
- Monkey-patch `torch.load` default to `weights_only=False` (PyTorch 2.6
  flipped this; YOLO ckpt has pickled class refs).
- Run YOLO on CPU (ultralytics MPS bug for Pose models, issue #4031).
- Cast `float64` tensors to `float32` before `.to('mps')` (MPS doesn't
  support `float64`).
- Add only `wilor/` to `sys.path` (not the repo root) so Python doesn't
  treat `cameramount/wilor/` as a namespace package — that would merge
  the user's MANO uploads at `cameramount/wilor/models/` with the real
  package's `cameramount/wilor/wilor/models/` and `import wilor.models`
  would resolve ambiguously. Now that scripts live under `scripts/`, the
  repo root is no longer implicitly on sys.path, so this just means
  inserting `WILOR_DIR` directly.
- For left-hand detections, multiply the X coord of `pred_keypoints_2d` and
  `pred_vertices` by `-1` before mapping back to image coords or saving the
  mesh. WiLoR's `ViTDetDataset` flips left-hand input crops, so the network
  output is in flipped-crop space — without un-flipping, the rendered
  skeleton is mirrored inside the bbox (looks roughly hand-shaped, but
  thumb/pinky are swapped, and the wrist X is biased toward the bbox
  centre).

If any of these stop being needed (upstream pinning newer ultralytics, etc.),
delete the corresponding workaround in `wilor_sanity.py` and `wilor_stereo_demo.py`
*and* this note.
