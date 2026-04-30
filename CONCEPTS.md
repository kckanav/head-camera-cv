# Pipeline concepts (Q&A)

Notes on **how** and **why** the egocentric hand-tracking pipeline works,
captured from a series of explainer questions during development. Read this
if you want to understand the conceptual framework — what each model does,
what units and frames the data lives in, and why the pipeline is structured
the way it is.

For **what** the scripts do, see [`CLAUDE.md`](CLAUDE.md).
For **what's planned**, see [`PLAN.md`](PLAN.md).
For the **project narrative + ladder**, see [`README.md`](README.md).

---

## Contents

1. [What CV models are we using? (full data flow + math)](#1-what-cv-models-are-we-using)
2. [What creates the blue-glove effect?](#2-what-creates-the-blue-glove-effect)
3. [MANO vs WiLoR — what is the difference?](#3-mano-vs-wilor--what-is-the-difference)
4. [How are θ and β stored, and in what reference frame?](#4-how-are-θ-and-β-stored-and-in-what-reference-frame)
5. [What is the ArUco marker for?](#5-what-is-the-aruco-marker-for)
6. [Why don't we just triangulate the table directly?](#6-why-dont-we-just-triangulate-the-table-directly)

---

## 1. What CV models are we using?

Three deep-learned models, plus a stack of OpenCV geometry. None of the
deep-learned ones live inside this repo — they're either downloaded
checkpoints or installed packages.

### The deep-learned models

**YOLOv8 hand detector** (`wilor/pretrained_models/detector.pt`)
- Input: full RGB frame.
- Output: 2D bounding boxes of hands + handedness label (left/right).
- Math: standard one-stage anchor-free detector. CNN backbone → per-pixel
  bbox + class.

**WiLoR ViT** (`wilor/pretrained_models/wilor_final.ckpt`)
- Input: a 256×256 hand crop from YOLO's bbox.
- Output: MANO pose θ (16 axis-angle joint rotations), shape β (10
  betas), per-image weak-perspective camera, plus reprojected 2D / 3D
  keypoints and the 778-vertex mesh.
- Math: ViT encoder → transformer decoder → regression heads. Trained on a
  mix of monocular hand datasets. The output is **monocular and
  scale-ambiguous** — pose and *relative* shape are right, but absolute
  size is a guess.

**MANO** (`wilor/mano_data/MANO_RIGHT.pkl`)
- Not a learned step at runtime — a fixed parametric hand model.
- Input: pose θ + shape β.
- Output: 778-vertex mesh + 21 joint positions.
- Math: linear blend skinning on a mean template mesh, plus pose-dependent
  blend shapes. Closed form, fast.

### OpenCV geometry (no deep learning, just math)

**ArUco / ChArUco detectors** — `cv2.aruco.ArucoDetector` finds quadrilateral
fiducials and returns their 4 corners with sub-pixel precision. Used for
calibration (ChArUco board) and table anchoring (single ArUco marker).

**Stereo calibration suite** (one-time, captured in our `.npz`):
- `cv2.calibrateCamera` → intrinsics K (focal length, principal point) and
  distortion coefficients per camera.
- `cv2.stereoCalibrate` → extrinsic R, T relating the two cameras:
  `X_right = R · X_left + T`.
- `cv2.stereoRectify` → rectification rotations R1, R2 + projection
  matrices P1, P2 such that, after virtually rotating each camera by
  R1/R2, corresponding image points lie on the same row.

**Stereo triangulation** — `cv2.triangulatePoints(P1, P2, x_l, x_r)` solves
for 3D X such that both rectified projections match. After rectification:
`Z = fx · baseline / (x_l − x_r)`.

**PnP** — `cv2.solvePnP` / `solvePnPGeneric`. Given 3D object points in
some local frame and their 2D image projections, recover the camera pose
`(R, t)` such that `K · (R · X_local + t)` projects to the observed 2D.
We use SQPNP for hand keypoints and IPPE_SQUARE for the marker.

**MediaPipe HandLandmarker** (legacy, only in
`scripts/pipeline/04_triangulate_mp.py`) — fast 21-keypoint detector kept
as a regression baseline against WiLoR.

### Data flow per frame

```
                                    ┌──── cam1 RAW frame (left) ────┐
                                    │                               │
[1] YOLO ──► hand bboxes  ─────────►│                               │
                                    │                               │
[2] WiLoR ViT (per crop) ──► θ, β, kp_3d_local, kp_2d_full, vertices│
                                    │                               │
[3] MANO (θ, β) ──► 778-vertex mesh in MANO local frame             │
                                    │                               │
                                    │       ┌──── cam0 RAW frame (right) ───┐
                                    │       │                                │
[1+2+3] same pipeline on right view │       │   another (R_right, t_right)   │
                                    │       │                                │
                                    ▼       ▼                                │
                          ┌────────────────────────────────────┐             │
[6] Triangulate wrist:    │ wrist 2D in cam1 + wrist 2D in cam0│             │
    rectify both, then    │ ──► cv2.triangulatePoints          │             │
    triangulate           │ ──► metric Z_wrist in WORLD frame  │             │
                          └────────────────────────────────────┘             │
                                       │                                     │
                                       ▼                                     │
[7] solvePnP(SQPNP) per view: fit (R_view, t_view) so monocular mesh         │
    projects to WiLoR's 2D keypoints. t_view[2] = monocular wrist depth.     │
                                       │                                     │
                                       ▼                                     │
[8] Scale anchor: k = Z_stereo_wrist / t_view[2]   (1 DOF per view)          │
                                       │                                     │
                                       ▼                                     │
[9] Metric mesh in cam frame: k · (R_view · vertices_local + t_view)         │
                                       │                                     │
                          ┌────────────┴───────────┐                         │
                          │                        │                         │
                   ┌──────▼─────┐         ┌────────▼──────┐                  │
                   │ project to │         │ transform via │                  │
                   │ each view  │         │ R1.T into     │                  │
                   │ (render)   │         │ WORLD frame   │                  │
                   └────────────┘         │  (stored)     │                  │
                                          └───────────────┘                  │

    ┌────────────────────────────────────────────────────────────────────────┘
    │  Separately:
    │  [4] ArUco detector finds marker corners in each view
    │  [7] solvePnP(IPPE_SQUARE) → marker pose (R_l, t_l) per view
    │  ──► T_world_to_table per frame (transforms WORLD frame → TABLE frame)
    │
    ▼
For Phase 5 export: apply T_world_to_table to all stored 3D quantities,
producing table-anchored, metric, retargeting-ready demonstrations.
```

### The two non-obvious tricks

**Trick 1 — Uniform 3D scaling preserves perspective 2D.** A point at
`(X, Y, Z)` projects to `(fX/Z, fY/Z)`. Scale the *whole 3D point* to
`(kX, kY, kZ)` and the projection is `(fkX/kZ, fkY/kZ)` = unchanged. So we
can change the metric size of the monocular mesh without breaking its 2D
fit. That's why a **1-DOF stereo correction** (just absolute scale) is
enough — pose, rotation, and finger-shape are all preserved from monocular.

**Trick 2 — Disparity → depth.** After rectification, the only thing that
varies for a 3D point is its horizontal pixel position between the two
views. The relationship is exactly `Z = fx · baseline / disparity`. So
with calibrated `fx ≈ 545 px` and `baseline = 41.9 mm`, every pixel of
disparity error at 50 cm depth costs ~1 cm in Z. That's the noise floor
we live with on this rig.

---

## 2. What creates the blue-glove effect?

The blue glove is the rendered **MANO mesh** painted onto the original
frame in 2D, with shading. Five steps:

**1. Mesh geometry — what we render.** MANO gives us 778 vertices and 1538
triangular faces forming a hand. WiLoR predicts the per-frame mesh; the
canonical pipeline scales it to metric size via the `k` anchor.

**2. Project 3D vertices to 2D.** `cv2.projectPoints(vertices_3d_camera,
K, dist)` gives us the 2D pixel coordinates for each of the 778 vertices
in the image plane. Now each triangle has both a 3D position (in camera
frame) and a 2D footprint.

**3. Paint the triangles back-to-front.** This is the **painter's
algorithm** — we sort all 1538 triangles by their **mean 3D Z** (depth),
descending, and draw the farther ones first so nearer ones overpaint
them. No z-buffer, no GPU — just sort and fill in order.

```python
face_z = triangles_3d.mean(axis=1)[:, 2]
order = np.argsort(-face_z)           # back to front
for fi in order:
    cv2.fillConvexPoly(overlay, pts2[fi], color)
```

**4. Per-triangle shading — Lambertian blue.** This is what gives it the
3D look instead of being a flat blob. For each triangle:
- Compute its 3D normal: `n = (v1 − v0) × (v2 − v0)`, normalised.
- Compute brightness:
  `shade = max(|n · light_direction|, 0.3)`.
  We use a fixed light direction `(0, −0.4, −1)` (slightly above and
  behind the camera) and clamp to 0.3 minimum so back-facing triangles
  aren't fully black.
- Multiply the base BGR colour `(255, 110, 30)` by `shade`. That base
  is the saturated cyan-blue you see.
- Draw the triangle's edge with a slightly darker BGR `(220, 80, 20)`.
  That gives the polygonal "wireframe-on-skin" look.

**5. Alpha-blend onto the original frame.**

```python
out = cv2.addWeighted(overlay, 0.55, original_frame, 0.45, 0)
```

55% mesh-colour, 45% original. That's why the hand still shows through
faintly — the glove is translucent rather than opaque.

### Why it looks "right"

Because the underlying mesh is positioned to **match WiLoR's 2D keypoints
exactly** (uniform-scaling preserves 2D projection — Trick 1 above). So
when WiLoR locks the wrist and finger positions to the actual hand pixels,
the mesh you paint over the frame is registered to within ~2 px of the
real hand outline. Combined with depth-sorted shading, your eye reads it
as a 3D blue glove sitting on the hand instead of a flat overlay.

Pure NumPy + OpenCV — no GPU rendering. ~15 ms per frame on Apple MPS
for one hand.

---

## 3. MANO vs WiLoR — what is the difference?

The short version: **MANO is a *mathematical model* of a hand. WiLoR is a
*neural network* that estimates the parameters of that model from an
image.** They live at completely different levels of the stack — one is
geometry, one is perception.

### MANO — the hand "puppet"

MANO was published by the Max Planck Institute in 2017. It's a fixed
mathematical formula that describes how a hand looks and articulates.
Think of it as a rigged 3D hand model in Blender — fixed mesh topology
and fixed skeleton. You give it inputs and it gives you a 3D mesh.

Inputs:
- **θ** (theta): 16 joints × 3 axis-angle values = 48 numbers describing
  the joint rotations (wrist, knuckles, finger curls).
- **β** (beta): 10 shape coefficients describing the *person*'s hand
  (size, finger thickness, etc.). These don't change frame-to-frame for
  the same person.

Output (deterministic, computed by a closed-form function):
- A **778-vertex mesh** (the surface).
- **21 keypoints** (anatomical landmarks: wrist, MCPs, PIPs, DIPs,
  fingertips).

That's it. MANO is the puppet — it can take any pose, but it doesn't
know how to look at the world. It's frozen on disk as `MANO_RIGHT.pkl`.

### WiLoR — the "puppeteer"

WiLoR is a neural network (a Vision Transformer) trained on lots of hand
images. Its job is to look at a hand crop and **predict θ and β** —
i.e., decide what pose the MANO puppet should be in to match the image.

- Input: a 256×256 RGB hand crop.
- Output: predicted θ and β (plus a few extras like a per-image
  weak-perspective camera).

WiLoR by itself doesn't produce a mesh — it produces *parameters*. The
mesh comes out only when you plug those parameters into MANO's formula.

### How they work together

```
   image ─► [WiLoR network] ─► θ, β ─► [MANO formula] ─► 778-vertex mesh + 21 joints
            (perception)              (geometry)
```

Inside WiLoR's checkpoint there's actually a copy of MANO baked in, so
calling `model(batch)` runs both steps in one forward pass and gives you
`pred_vertices`, `pred_keypoints_3d`, etc. directly. But conceptually
they're separate.

### What this means for our pipeline specifically

- **The skeleton drawn on the AR video** = the 21 keypoints. The
  *anatomy* (which dot is the wrist, which is the index-MCP, etc.) is
  **defined by MANO**. Their *positions in the image* come from
  **WiLoR's predictions**.
- **The blue-glove mesh** = the 778-vertex surface. The *topology*
  (which vertex connects to which) is **MANO**. Their *3D positions*
  come from **WiLoR + MANO**.

In the saved `.npz`, the most compact and most "lingua franca" thing to
keep is **θ and β**. With those, anyone can rebuild the mesh later by
calling MANO. Mesh vertices are 778 × 3 = 2334 floats per hand per frame;
θ + β is just 48 + 10 = 58 floats. Most robot-retargeting codebases
consume θ directly — they map MANO's joint angles onto their robot's
joint angles.

### Why the distinction matters

It means we have a **swappable perception layer**. If WiLoR ever becomes
obsolete (or HaMeR works better on egocentric video, or some new model
lands later), we can swap the predictor without changing anything
downstream — as long as the new model also outputs MANO parameters, all
our anchoring math, our table-frame export, our retargeting code keeps
working.

That's why our Phase 5 export should save `mano_pose` (= θ) and
`mano_betas` (= β) — not just the meshes. The meshes are derived; θ + β
are the *source of truth*.

---

## 4. How are θ and β stored, and in what reference frame?

### θ — joint rotations

Stored as a `(16, 3)` float array. **48 numbers.**

Each row is an **axis-angle vector**: a 3D vector where the *direction*
is the rotation axis and the *magnitude* (in **radians**) is the
rotation angle. So if `θ[5] = (0, 0, 1.57)`, that means "joint 5 rotates
1.57 radians (= 90°) around the +Z axis of its parent joint's frame."

There are 16 joints in MANO's tree: 1 wrist + 3 per finger × 5 fingers.
They're arranged hierarchically (wrist → MCP → PIP → DIP), so each
rotation is **relative to its parent**. The wrist's rotation is relative
to the MANO "root frame" — i.e., it tells you which way the whole hand
is facing in MANO's local universe.

Units: radians. Range: ~[−π, π] per axis.

### β — shape coefficients

Stored as a `(10,)` float array. **10 numbers.**

These are **PCA scores** in a learned basis of hand shapes. MANO was
built by scanning ~1000 real hands and running PCA on the variation. β
= (1.5, −0.3, 0.8, ...) means "this person's hand = mean hand + 1.5 ×
shape_basis_0 − 0.3 × shape_basis_1 + ...". The first component is
roughly overall size (big hand vs small hand); later ones capture
finger-length ratios, palm thickness, etc.

Units: dimensionless. Typical range: [−3, +3] (it's standardised).

### 3D vertex positions — units and frame

When MANO computes its mesh from (θ, β), the output is `(778, 3)`
floats. The units are **metres**. But — and this is the key — they're in
**MANO's local frame, not the camera frame, and not the world**.

What does that mean concretely?

- The wrist is at the origin `(0, 0, 0)` of MANO's local frame.
- The mesh extends roughly ±0.1 m around the wrist (a hand is ~10 cm
  across).
- The orientation is a "canonical hand orientation" defined by MANO —
  palm-down, fingers pointing along some axis.

So when you call `MANO(θ, β)` and look at vertex 100, you might get
something like `(0.04, −0.02, 0.07)` metres. That doesn't mean "vertex
100 is 4 cm to the right of the camera." It means "vertex 100 is 4 cm
in the +X direction from the wrist *in MANO's own coordinate system*,
when the hand is posed according to θ".

It's analogous to having a rigged 3D hand model in Blender: the mesh has
positions in the model's local space, but until you put it INTO a scene
with a transform, those positions don't correspond to any real-world
location.

### Why this matters for our pipeline

WiLoR outputs vertices in MANO's local frame. To **place that mesh in
your real scene** (above your table, anchored to the marker), we need
three additional pieces of information that MANO+WiLoR alone can't give
us:

1. **Where is the hand in 3D space?** (translation `t`)
2. **Which way is the hand facing in 3D space?** (rotation `R`)
3. **What's the absolute size?** (scale `k`)

Each of those is solved by a different step in the canonical pipeline:

| Question | Solved by | Math |
|---|---|---|
| Where in 3D? | `solvePnP` per view | Fit (R, t) so projected MANO 2D matches image 2D, plus stereo wrist for absolute Z |
| Which way facing? | Same `solvePnP` | The R part of (R, t) |
| Absolute size? | Stereo wrist depth + the `k` anchor | `k = Z_stereo_wrist / t_pnp[2]` |

Once you have (R, t, k), you transform every MANO vertex:

```
vertex_in_camera = k · (R · vertex_in_MANO_local + t)
```

Now the vertex has a 3D position in the camera's frame, in metric metres.

Then `T_world_to_table @ vertex_in_world` puts it in table frame.

So the **chain of frames** for a single vertex looks like:

```
MANO local frame (wrist-centred, MANO units)
    │  scale by k, rotate by R, translate by t   ← from solvePnP + stereo anchor
    ▼
Real left camera frame (metres, camera origin)
    │  apply R1 (stereo rectification)            ← from cv2.stereoRectify
    ▼
World frame (= rectified-left frame, metres)
    │  apply T_world_to_table                     ← from ArUco solvePnP
    ▼
Table frame (metres, marker centre = origin, Z = up)
```

That's why the saved `.npz` has so many fields — each is needed to
express where something is at a different stage of the chain.

### Pixels vs 3D positions

MANO outputs are **never** in pixels. Pixels show up only when you
*project* a 3D point through a camera intrinsic matrix `K`:

```
pixel = K · (3D point in camera frame)
```

That's the `cv2.projectPoints` step. It's the LAST thing that happens
in the canonical pipeline — only for the rendered visualisation.

For a Phase 5 export to a 3D modelling software, **you don't want pixels
at all**. You want the 3D vertex positions in **table frame, in metric
metres**. Or even more compact: you want θ + β + the wrist 6-DoF
transform that maps MANO local → table frame, which lets the consumer
rebuild everything else.

---

## 5. What is the ArUco marker for?

The marker is doing a **different job** than answering the three "where /
which-way / how-big" questions for the hand. Those three are answered
**entirely from the cameras**, not the marker:

- **Where the hand is in 3D space** → `solvePnP` per view + stereo
  triangulation of the wrist.
- **Which way the hand is facing** → `solvePnP` per view (the R part of
  the same fit).
- **Absolute size (the `k` anchor)** → stereo triangulation of the wrist
  2D landmark across both views, which gives metric Z, which lets us
  compute `k = Z_stereo / Z_monocular`.

None of these involve the marker. They use:
- The **stereo calibration** (the `.npz` from `calibrate_stereo.py`),
  which tells us the rigid geometry between the two cameras — that's
  what makes stereo triangulation possible at all.
- The **monocular WiLoR predictions** in each image.

So after this stage, we have a metric MANO mesh **in the moving
head-mounted camera's frame**, frame by frame.

### What the marker does — answers a different question

The marker answers: **"Where is the table?"**

Without the marker: your camera is on your head. As your head moves, the
camera frame moves with it. A point that's stationary on the table — say,
the corner of a coaster — has different camera-frame coordinates every
frame. If we just exported the hand mesh in camera frame, you'd see the
hand AND the entire world wobbling around in Blender as you replayed the
demo, because everything is being expressed relative to a moving camera.

For robot training, you want everything in **the robot's workspace** —
which is anchored to the table, not to the user's head. So we need a
transform that takes camera-frame points → table-frame points, **per
frame**, accounting for head motion.

The marker provides exactly this. Detected every frame, `solvePnP` against
its known size gives us `T_world_to_table[frame]` — a per-frame transform
from the moving world (rectified-left) frame into the stationary table
frame.

The marker doesn't help us figure out where the hand is. It helps us
**express** where the hand is, **relative to the stationary table** instead
of the moving camera.

### The two stereo roles, separated

There are actually **two completely separate uses of stereo** in our
pipeline, and it's worth keeping them distinct:

| Stereo role | Source | What it gives us |
|---|---|---|
| **Stereo calibration** (one-time, from the ChArUco board capture) | `<date> wide - stereo calibration.npz` | The rigid geometry between cam1 and cam0 (their R, T, intrinsics, distortion). This is what lets us triangulate ANY 2D-2D point pair into 3D. |
| **Stereo triangulation** (per frame, for the wrist) | Live, in `08_wilor_canonical.py` | The metric Z of the wrist, which anchors the absolute scale of the monocular MANO mesh via `k`. |

The marker is **separate from both of these**. It's a third role:

| Marker role | Source | What it gives us |
|---|---|---|
| **Table anchoring** (per frame) | Live, in `09_anchor_table.py` | A per-frame `T_world_to_table` transform that takes any camera-frame point and re-expresses it in stationary table coordinates. |

The full chain for a single hand vertex on a single frame is:

1. WiLoR predicts MANO θ, β → 778 vertices in **MANO local frame**
   (scale-free, wrist-centred).
2. `solvePnP` + stereo wrist triangulation → vertex in **left camera
   frame** (metric metres, but in a frame that moves with the head).
3. Stereo calibration's `R1` rotates left-camera into rectified-left =
   **world frame** (still moves with head, but parallel-rectified).
4. Marker-derived `T_world_to_table` → vertex in **table frame** (metric
   metres, stationary, Z=0 is the table surface).

The first three steps don't involve the marker at all. The marker only
enters at step 4.

---

## 6. Why don't we just triangulate the table directly?

You could absolutely triangulate the table. Two reasons we use a marker
anyway, both worth understanding.

### Reason 1 — Identifying the "same point" in both views

Stereo triangulation only works when you know that pixel `(x_l, y_l)`
in the left view and pixel `(x_r, y_r)` in the right view are **the
same physical point in 3D**. For the wrist, this is easy: WiLoR detects
"the wrist landmark" in each view independently, and we trust those two
detections refer to the same anatomical wrist.

For the table, what's the equivalent? The table is **mostly
featureless**. Most of its surface is flat colour with no texture. If you
tried to match pixels between the two views — say "this beige pixel in
the left view = which beige pixel in the right view?" — you'd have no
way to know. Every beige pixel looks like every other beige pixel. The
matching is ambiguous.

This is the classic **correspondence problem** in stereo vision. It's
why dense stereo (the `viz/depth_dense.py` script) gives crisp depth on
textured regions but noisy/missing depth on flat surfaces. If you ran
SGBM on a clip of just your bare table, you'd get a depth map full of
speckle and holes precisely on the table.

The marker solves this brutally well: the high-contrast black-and-white
pattern is uniquely identifiable, and the corner detector finds the same
four corners reliably in both views with sub-pixel precision.

### Reason 2 — A point isn't enough; you need a frame

Even if we DID triangulate the table successfully (let's say you put a
textured tablecloth on it and stereo SGBM works perfectly), we'd recover
the **table plane** — its height and orientation, but **not its in-plane
axes**.

A plane has 3 DOF: one for height, two for tilt. A coordinate frame has
6 DOF: 3 for translation, 3 for rotation. So the missing piece is the
**rotation about the table normal** — i.e., which direction along the
table is "X" and which is "Y"?

Without that, you'd have:
- table-Z (height) for any point. ✓
- table-X and table-Y. ✗

Robot retargeting needs all 6 DOF. "The mug is at table position (12 cm,
5 cm)" requires knowing what 12 cm in X means.

The ArUco marker is **chiral** — its pattern is asymmetric, so its X and
Y axes are well-defined. Detecting the marker gives you all 6 DOF of
the table frame in one shot.

### Alternatives if you ever want to drop the marker

You're not stuck with a marker. There are real alternatives that recover
the full 6-DOF frame:

1. **Texture the table.** A patterned tablecloth or a coffee table with
   strong wood grain gives stereo enough features to triangulate the
   surface. Fit a plane to the triangulated points → get the plane. Then
   orient it using one strong feature (a logo, a corner of an object,
   the edge of the table) → get X/Y. Sensitive to lighting and surface
   texture.

2. **Track the table edges.** If the camera can see the corners or edges
   of the table, those are strong geometric features. You can
   triangulate the corners, fit a rectangle, and use the rectangle's
   sides for X and Y. Practical only if your FOV reliably includes the
   edges.

3. **Multiple markers / objects.** Put two distinguishable objects on
   the table at known positions. Each gives a 3D point; the line between
   them gives you an X axis; the table normal completes the frame. Less
   robust than one well-detected ArUco.

4. **A larger fiducial like a ChArUco board** (which you already have
   for calibration). Same idea as our marker but with more redundancy.

5. **Detect a known object** (mug with a known shape, fixed known
   pattern). Conceptually fine but requires a prior 3D model and
   pose-estimation infrastructure.

The single ArUco we're using is essentially the *minimum-overhead*
version of these. It's small (8 cm), trivially detected, gives all 6
DOF, and only needs to be visible in one view to anchor that frame for
that frame.

### The deeper answer

**Stereo can recover the table's plane, but not its frame.** The
marker's job is to define the **frame**, not to provide depth. The
marker isn't a stereo input at all — it's a 6-DOF reference object.
