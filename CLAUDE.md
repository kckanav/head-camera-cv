# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Hardware + CV prototype for an egocentric hand-tracking rig:

- **Rig**: head-mounted dual-camera mount, two Raspberry Pi Camera Module 3 sensors on a Raspberry Pi 5, tilted downward toward the wearer's hands. The cameras were *designed* to toe out **24°** from each other, but stereo calibration recovered **18.9°** (print tolerances on the bracket). The recovered baseline is **41.1 mm**. **Use the calibrated geometry from `outputs/<date> - stereo calibration.npz`, not the design value, for any stereo math.**
- **Goal**: capture the wearer's hand movements and hand-object interactions, then run a CV pipeline. Stages 1 (per-camera MediaPipe), 3 (stereo calibration), and 4 (sparse 3D triangulation of MediaPipe keypoints) are done. **Stage 8 (dexterous hand capture via WiLoR + stereo) is now in progress** — Phase 0 (env + weights) and Phase 1 (sanity check) are done. The plan's full state is in `PLAN.md`; read that first when working on stage 8. Full ladder is in `README.md`.
- **Captured frames**: `inputs/24th April 2026 - photo cam0.jpg` (right) and `inputs/24th April 2026 - photo cam1.jpg` (left) are reference single-frame captures from the actual rig. **Convention used everywhere in this repo: cam1 = left, cam0 = right.**
- **CAD**: `cad/dual_picam3_holder.scad` (OpenSCAD source) and exported `.stl` / `.3mf` files (`_v2` is the current revision). The CAD only matters when the physical geometry is in question — usually it isn't.

## Repo layout (only the non-obvious bits)

All entry points live under `scripts/`, segregated by role. Everything is run from the **repo root** so the relative `inputs/` / `outputs/` / `models/` paths resolve correctly.

```
scripts/
├── _lib/             shared imports (sys.path-injected, not a package)
├── pi/               runs on the Pi 5 only
├── calibration/      one-shot setup utilities
├── pipeline/         canonical pipeline, in run order (numbered)
├── viz/              visualisation + debug viewers
└── experiments/      historical / reference scripts (not on the canonical path)
```

### `scripts/_lib/` — shared imports

Each script that needs these does `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))` near the top, then imports from these modules by bare name (no package prefix).

- `_lib/dated.py` — `today_pretty()` returns a string like `27th April 2026`. Every script that writes dated artefacts imports this.
- `_lib/device.py` — **device portability layer.** `pick_device()` returns CUDA → MPS → CPU. `configure_perf(device)` returns a config dict (`yolo_device`, `fp16`, `num_workers`, `pin_memory`, `autocast_dtype`) that the wilor scripts plumb into ultralytics, ViTDetDataset, DataLoader, and `torch.autocast`. Same script behaves correctly on Mac (MPS+CPU YOLO, no fp16) and on a CUDA box (CUDA YOLO, fp16 ViT, TF32, pinned-memory dataloader). `to_device_safe(obj, device)` handles MPS's no-fp64 quirk and uses `non_blocking=True` only on CUDA. `autocast_ctx(device, dtype)` is a no-op on non-CUDA. `cuda_sync(device)` is a no-op on non-CUDA — wrap timing blocks with it for accurate CUDA numbers.
- `_lib/wilor_setup.py` — **side-effect-on-import** module that sets up the WiLoR runtime: stubs three pyrender submodules (no EGL on macOS), patches `torch.load` to default `weights_only=False` (PyTorch 2.6 ckpt-loading break), inserts `<repo>/wilor` on `sys.path`, and `chdir`s into `wilor/` so WiLoR's relative `./pretrained_models/` paths resolve. Exports `PROJECT_ROOT`, `WILOR_DIR`, and `get_mano_faces(model)`. Every wilor script does `import wilor_setup` early — once, before any `import torch` / `from wilor.*`.

### `scripts/pi/` — runs on the Pi 5

- `pi/dualstream.py` — discovers IMX708 sensor modes via `picamera2.sensor_modes`, exposes them as buttons in a live HTML preview UI, locks capture to 30 fps via `FrameDurationLimits`. Default mode is 2304×1296 (full sensor, ~100° HFOV); the older 1536×864 cropped mode is ~75°. Lives on the Pi but is checked into the repo.

### `scripts/calibration/` — one-shot setup

- `calibration/make_charuco_board.py` — generates the print-ready ChArUco board PDF used as the stereo-calibration target.
- `calibration/make_aruco_marker.py` — generates the print-ready ArUco marker PDF for table-frame anchoring (stage 8 phase 4–5). Default is one 80×80 mm `DICT_4X4_50` ID 0 with a 100 mm scale bar for print verification.
- `calibration/calibrate_stereo.py` — stereo calibration from videos of the ChArUco board (stage 3). Writes intrinsics, extrinsics, rectification maps, and `Q` to an `.npz`. Has a `TAG` constant near the top to keep narrow-FOV (`""`) and wide-FOV (`" wide"`) calibrations side-by-side in `outputs/`.

### `scripts/pipeline/` — canonical pipeline (numbered, run in order)

- `pipeline/01_per_cam_sanity.py` — MediaPipe HandLandmarker sanity-check pipeline (stage 1).
- `pipeline/04_triangulate_mp.py` — sparse 3D hand triangulation via MediaPipe + stereo (stage 4). Loads the calibration, rectifies both streams, runs MediaPipe per view, pairs hands by epipolar (rectified-row) proximity, triangulates 21 landmarks per matched hand. Writes an annotated SBS video and a per-frame `(N, 2, 21, 3)` landmarks `.npz`. **Kept as a fast preview path / regression baseline** for the wilor pipeline.
- `pipeline/08_wilor_canonical.py` — **stage 8 canonical pipeline** (was `wilor_phase3_anchored.py`). Per detected hand, per view: `cv2.solvePnP(SQPNP)` fits (R, t) so the mesh in real camera frame projects to WiLoR's 2D keypoints, then a 1-DOF scalar `k = Z_stereo_wrist / t_pnp[2]` anchors the metric scale (uniform 3D scaling preserves perspective 2D, so the rendered overlay matches `viz/wilor_ar_monocular.py`). Saves canonical world-frame metric mesh derived from the LEFT view, plus per-view PnP placement + scale + wrist 3D + PnP residuals for diagnostics. **Has full CUDA/MPS device portability** via `_lib/device.py`, plus a `--bench` flag that prints per-stage timings (io / inference / fusion / render) with `cuda.synchronize()` bracketing for accurate CUDA numbers.

### `scripts/viz/` — visualisation + debug

- `viz/play_stereo.py` — side-by-side stereo player with on-the-fly rectification (recomputed at any `--alpha`, defaults to 0 for a clean rectilinear view rather than the saved `alpha=1` "two small spheres" look).
- `viz/depth_dense.py` — dense stereo depth via SGBM. Three render styles: `overlay` (default — blue depth mask alpha-blended on the left frame, near = solid), `side` (turbo SBS), `both`.
- `viz/inspect_3d_mp.py` — quick matplotlib visualisation of `04_triangulate_mp.py`'s `.npz`: wrist depth + pinch aperture over time.
- `viz/stitch_panorama.py` — old SIFT/RANSAC panorama stitcher. Kept as a debug view; **not** on the canonical pipeline (stitching collapses parallax, which is exactly what we want to keep). See architecture notes below.
- `viz/wilor_ar_monocular.py` — per-view monocular AR overlay (was `wilor_ar_overlay.py`). Runs WiLoR per view, fits a weak-perspective affine (`pred_kp3d.xy` → `pred_kp2d`), applies it to all 778 vertices, rasterises via painter's algorithm with Lambertian shading. Two independent meshes per frame; the stereo rig isn't doing any work on the mesh itself. **The visual baseline** that the canonical pipeline is compared against. Also has full CUDA/MPS device portability.

### `scripts/experiments/` — kept for reference

- `experiments/wilor_sanity.py` — stage 8 / Phase 1 sanity check. Loads WiLoR + YOLO + MANO, runs them on `inputs/24th April 2026 - photo cam0.jpg`, writes a 2D-keypoint overlay and the MANO mesh as `.obj`. Run via `.venv-hamer/bin/python scripts/experiments/wilor_sanity.py`.
- `experiments/wilor_wrist_stereo.py` — stage 8 / Phase 2 (was `wilor_stereo_demo.py`). Per-view WiLoR + stereo *wrist-only* triangulation. Superseded by `pipeline/08_wilor_canonical.py` which does this and also fits the full mesh.
- `experiments/wilor_phase3_umeyama.py` — Phase 3 (full Umeyama) reference experiment (was `wilor_phase3.py`). Fits a 7-DOF similarity transform via weighted Umeyama 1991. Gave one mesh in one frame but worse 2D accuracy than monocular WiLoR (median 10 px reproj). Kept for the negative-result record; the canonical pipeline uses the 1-DOF anchor instead.

### Other dirs

- `PLAN.md` — concrete plan for stage 8. Has a "pick-up-where-we-left-off" header at the top so a fresh session can resume. Update this whenever a phase finishes or a decision changes.
- `cad/` — OpenSCAD source + `.stl` / `.3mf` exports of the head bracket.
- `inputs/` — tracked test material (reference photos, short clips, calibration footage). Filenames are dated with the **capture** date.
- `outputs/` — tracked generated artefacts. Filenames are dated with the **generation** date.
- `raw/` — **gitignored**. Local-only raw camera recordings (`cam{0,1}.mp4`, `cam{0,1}_*.h264`). Don't add tracked content here.
- `models/` — **gitignored**. Re-downloadable model weights (currently just `hand_landmarker.task`). `01_per_cam_sanity.py` and `04_triangulate_mp.py` reference it as `models/hand_landmarker.task`.
- `.venv/` — local Python 3.9 virtualenv used by stages 1–4. `opencv-python`, `numpy`, `mediapipe`, `matplotlib`, `Pillow` already installed. Use it directly; don't recreate.
- `.venv-hamer/` — separate Python 3.10 virtualenv used by stage 8. Has `torch` (with MPS on Mac, CUDA on Linux), the WiLoR package's deps, plus `dill` and `Cython` (not in WiLoR's `requirements.txt`; needed for the YOLO ckpt and `xtcocotools` build respectively). Don't recreate; if you do, follow the Phase 0 commands in `PLAN.md`.
- `wilor/` — cloned WiLoR repo, **gitignored** (large weights, separate license). Contains `pretrained_models/{detector.pt, wilor_final.ckpt}`, `models/MANO_{RIGHT,LEFT}.pkl` (under user's MANO academic license), and `mano_data/MANO_{RIGHT,LEFT}.pkl` symlinked from the above (WiLoR's config expects them at `mano_data/`). Audited for hardcoded device strings — none found, the package is device-agnostic.

## Files NOT in git (too large, redownloadable, or license-restricted)

- `raw/cam0.mp4`, `raw/cam1.mp4` — full 4-min recordings, ~144 MB each (over GitHub's 100 MB limit). `viz/stitch_panorama.py` reads them from `raw/`. Switch the repo to git-lfs if you need to track them.
- `raw/*.h264` — raw streams from the Pi, same content as the mp4s. Mux to mp4 with `ffmpeg -framerate 30 -i <file>.h264 -c copy <file>.mp4` if you need a container with proper timing (raw h264 has no framerate metadata, so `ffprobe` defaults to 60).
- `outputs/*stitched panorama.mp4` — the full stitched video is ~300 MB. We commit a sample frame instead; regenerate locally with `viz/stitch_panorama.py`.
- `outputs/*wilor stereo demo 60s*.mp4` — the 60-s WiLoR demo render is ~150 MB. The `.npz` and the `wilor 60s perf trace.png` are tracked; re-render locally if needed.
- `models/hand_landmarker.task` — MediaPipe model, re-downloadable from Google's mediapipe-models bucket.

## Running things

All commands run from the **repo root**:

```bash
# stage 1 / 3 / 4 (Python 3.9 venv)
.venv/bin/python scripts/calibration/make_charuco_board.py    # regenerate the ChArUco PDF
.venv/bin/python scripts/calibration/calibrate_stereo.py      # stereo calibration from inputs/<date> - cam{0,1} calibration.mp4
.venv/bin/python scripts/calibration/make_aruco_marker.py     # ArUco marker for table anchoring (stage 8 phase 4-5)
.venv/bin/python scripts/pipeline/01_per_cam_sanity.py        # per-camera MediaPipe sanity check
.venv/bin/python scripts/pipeline/04_triangulate_mp.py        # sparse 3D hand triangulation; SBS video + .npz
.venv/bin/python scripts/viz/inspect_3d_mp.py                 # plot wrist depth + pinch over time
.venv/bin/python scripts/viz/play_stereo.py                   # SBS rectified playback (recomputes maps at --alpha)
.venv/bin/python scripts/viz/depth_dense.py                   # dense stereo depth (--style overlay/side/both)
.venv/bin/python scripts/viz/stitch_panorama.py               # debug-only panorama stitcher

# stage 8 (Python 3.10 venv)
.venv-hamer/bin/python scripts/pipeline/08_wilor_canonical.py            # canonical: PnP + stereo depth anchor
.venv-hamer/bin/python scripts/pipeline/08_wilor_canonical.py --bench    # adds per-stage perf trace
.venv-hamer/bin/python scripts/viz/wilor_ar_monocular.py                 # per-view monocular AR (visual baseline)
.venv-hamer/bin/python scripts/experiments/wilor_sanity.py               # single-image sanity (Phase 1)
.venv-hamer/bin/python scripts/experiments/wilor_wrist_stereo.py         # wrist-only stereo (Phase 2)
.venv-hamer/bin/python scripts/experiments/wilor_phase3_umeyama.py       # Phase 3 (full Umeyama) reference experiment
```

The wilor scripts, `04_triangulate_mp.py`, and `inspect_3d_mp.py` all `subprocess.run(["open", ...])` their output on macOS so the result pops open after the run.

### Device portability (CUDA / MPS / CPU)

The canonical pipeline (`pipeline/08_wilor_canonical.py`) and the visual baseline (`viz/wilor_ar_monocular.py`) use the `_lib/device.py` helpers, so the *same script* runs efficiently on:

- **macOS / Apple Silicon**: MPS for the WiLoR ViT, CPU for YOLO (ultralytics issue #4031 — Pose models break on MPS), no fp16, no autocast, no DataLoader workers.
- **Linux + NVIDIA**: CUDA for both ViT and YOLO, fp16 input + fp16 autocast for the ViT (~1.7–2× throughput on Ampere+), TF32 fp32 matmul, `cudnn.benchmark=True`, multi-worker DataLoader with pinned memory for async H2D copy.
- **CPU only**: fp32 throughout, no autocast, no workers — slow but functional.

Verify on a CUDA box with `--bench` and a few frames; expect ~3–5× lower per-frame inference time than the unconfigured baseline (YOLO-on-GPU + fp16 ViT + TF32 stack).

## Pipeline architecture

### `viz/stitch_panorama.py` — stitching

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

### `calibration/calibrate_stereo.py` — stereo calibration

Reads `inputs/<date> - cam{0,1} calibration.mp4` (synchronised left/right videos of the ChArUco board moved by hand in front of a tripod-mounted rig). Per-camera intrinsics with `cv2.calibrateCamera`, then stereo extrinsics with `cv2.stereoCalibrate(..., CALIB_FIX_INTRINSIC)`, then `cv2.stereoRectify(..., alpha=1)`.

**`alpha=1` is intentional**: it preserves the full source FOV in the rectified output (with black padding where rectification has no source pixel). `alpha=0` cropped to the largest all-valid rectangle and clipped the top of the frame — exactly where hands held high need to be visible. Don't switch back without good reason.

Outputs everything (K_l/K_r, dist_l/dist_r, R, T, E, F, R1/R2, P1/P2, Q, remap tables, plus the rms numbers and recovered angle/baseline) to `outputs/<date> - stereo calibration.npz`. Renders a side-by-side rectified sample with green scanlines as a visual check — corresponding points must lie on the same scanline.

### `pipeline/04_triangulate_mp.py` — sparse 3D hand keypoints

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

### `_lib/wilor_setup.py` — WiLoR runtime workarounds (centralised)

Every wilor script (`pipeline/08_wilor_canonical.py`, `viz/wilor_ar_monocular.py`, all of `experiments/wilor_*.py`) does `import wilor_setup` near the top, *before* any `import torch` or `from wilor.*`. That single import applies all the runtime workarounds we need on this stack (WiLoR @ rolpotamias/main + ultralytics 8.1.34 + torch 2.11 on macOS):

- Stubs the three pyrender-using submodules of `wilor.utils` (no EGL on macOS by default).
- Monkey-patches `torch.load` default to `weights_only=False` (PyTorch 2.6 flipped this; the YOLO ckpt has pickled class refs).
- Inserts `<repo>/wilor` on `sys.path` and `chdir`s into it so WiLoR's relative `./pretrained_models/` paths resolve.

Two further workarounds are *device-conditional*, applied via `_lib/device.py`:

- YOLO on CPU only when `device.type == "mps"` (ultralytics issue #4031). On CUDA the detector runs on GPU as normal.
- `float64 → float32` cast before `.to(device)` only when `device.type == "mps"` (MPS doesn't support fp64). On CUDA / CPU we leave dtypes alone.

The last one is *per-script*, not centralised:

- For left-hand detections, multiply the X coord of `pred_keypoints_2d`, `pred_vertices`, and `pred_keypoints_3d` by `-1` before mapping back to image coords or saving the mesh. WiLoR's `ViTDetDataset` flips left-hand input crops, so the network output is in flipped-crop space — without un-flipping, the rendered skeleton is mirrored inside the bbox (looks roughly hand-shaped, but thumb/pinky are swapped, and the wrist X is biased toward the bbox centre).

If any of these stop being needed (upstream pinning newer ultralytics, PyTorch reverting `weights_only`, EGL on macOS), delete the corresponding step in `_lib/wilor_setup.py` (or `_lib/device.py`) *and* this note.

### `pipeline/08_wilor_canonical.py` — canonical Phase 3 (monocular pose + stereo depth)

The cleanest way to combine WiLoR's monocular strength with our stereo
metric depth. Per matched-hand pair per frame:

1. WiLoR per view → `pred_vertices`, `pred_keypoints_3d` (MANO local
   frame, root-centered), `pred_keypoints_2d_full` (real-image pixels).
2. `cv2.solvePnP(SQPNP)` per view → (R_view, t_view) such that the mesh
   in the real camera frame projects to WiLoR's 2D keypoints. This is the
   monocular wrist depth estimate (= t_view[2]).
3. Stereo triangulate the wrist in rectified-left frame, transform into
   each real camera frame to get `Z_stereo_view`.
4. Scalar scale anchor: `k_view = Z_stereo_view / t_view[2]`. (1 DOF.)
5. Metric mesh in each real camera frame: `k_view · (R_view · verts_local + t_view)`.
   Render via `cv2.projectPoints` through (K_view, dist_view).

**Math justification.** Uniform 3D scaling preserves perspective 2D
projection, so the rendered overlay matches the per-view monocular
output (`viz/wilor_ar_monocular.py`) with PnP residuals of ~2 px median.
The 1-DOF anchor uses stereo only for the one thing it can give that
monocular cannot: absolute scale.

**Saved canonical 3D**: `verts_3d_world` is the LEFT view's metric mesh
transformed to world (rectified-left) frame. PnP placement + scale +
wrist 3D + PnP residual are saved per view for diagnostics. This `.npz`
is what Phase 5 (robot-training export) consumes.
