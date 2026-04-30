"""Stage 8 / Phase 5: dataset export.

Combines the canonical pipeline output (08_wilor_canonical.py) with the
table anchor output (09_anchor_table.py) into a single retargeting-ready
.npz with everything in **table frame** (Z=0 = table surface, +Z up).

The canonical pipeline produces metric MANO data in **world frame** (=
rectified-left camera frame, which moves with the head). The anchor
script produces a per-frame **T_world_to_table** transform. This script
applies that transform and writes a self-contained dataset entry.

Data flow:

    canonical .npz                   anchor .npz
       │                                 │
       ├ verts_3d_world (in world)       ├ R_world_to_table (per frame)
       ├ wrist_3d_world (in world)       ├ t_world_to_table (per frame)
       ├ pose_R_left   (MANO→origLeft)   ├ R_pnp_left, t_pnp_left
       ├ pose_t_left                     ├ detected_left  (anchor presence)
       ├ mano_global_orient_R            └ ...
       ├ mano_hand_pose_R                          │
       ├ mano_betas                                │
       └ ...                                       │
                  │                                │
                  └────────────┬───────────────────┘
                               ▼
                         this script:
                  apply T_world_to_table to every 3D
                  quantity, convert MANO rotations to
                  axis-angle, embed calibration intrinsics
                               ▼
                  outputs/<date> - dataset [<tag>].npz
                  (self-contained, schema_version="1.0")

Schema (saved in the .npz):

  Per-frame, per-slot (slot 0 is the leftmost-by-X hand, slot 1 the next):
    mano_pose          (N, 2, 16, 3)    axis-angle, [global_orient | 15 hand_pose]
    mano_betas         (N, 2, 10)       shape coefficients
    vertices_table     (N, 2, 778, 3)   metric MANO mesh in TABLE frame (m)
    wrist_table        (N, 2, 3)        wrist position in table frame (m)
    wrist_R_table      (N, 2, 3, 3)     wrist orientation: takes MANO local axes
                                        to table-frame axes
    hand_present       (N, 2)           bool — slot has anchored, fused hand
    handedness         (N, 2)           "right" / "left" / ""

  Per-frame:
    T_world_to_table   (N, 4, 4)        per-frame world→table transform
    anchored           (N,)             bool — marker detected this frame
    timestamps_s       (N,)             seconds since clip start

  Constants:
    intrinsics_left    (3, 3)           K_left
    intrinsics_right   (3, 3)           K_right
    distortion_left    (5,)             dist_left
    distortion_right   (5,)             dist_right
    stereo_R           (3, 3)           extrinsic rotation cam_left → cam_right
    stereo_T           (3,)             extrinsic translation
    baseline_m                          scalar
    fps                                 scalar
    image_size         (2,)             [W, H]
    mano_faces         (1538, 3) int    MANO mesh topology

  Provenance:
    schema_version     "1.0"
    source_canonical   path string
    source_anchor      path string
    source_calibration path string
    generated_on       date string

Run:
  .venv/bin/python scripts/pipeline/10_export_dataset.py
  .venv/bin/python scripts/pipeline/10_export_dataset.py \\
      --canonical "outputs/<date> - phase3 anchored fused [foo].npz" \\
      --anchor    "outputs/<date> - table anchor [foo].npz" \\
      --tag       "foo"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "1.0"

DEFAULT_CANONICAL = (PROJECT_ROOT
    / "outputs/30th April 2026 - phase3 anchored fused [20260430 manipulation].npz")
DEFAULT_ANCHOR = (PROJECT_ROOT
    / "outputs/30th April 2026 - table anchor [20260430 manipulation].npz")
DEFAULT_TAG = "20260430 manipulation"


def from_root(p):
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def rotation_matrices_to_axis_angle(R_arr):
    """Batch-convert (..., 3, 3) rotation matrices to (..., 3) axis-angle.
    NaN-rotation matrices map to NaN axis-angle vectors. Uses cv2.Rodrigues
    per matrix (vectorised would be nicer, but ~17k matrices for a 540-frame
    clip runs well under a second)."""
    R_arr = np.asarray(R_arr)
    out_shape = R_arr.shape[:-2] + (3,)
    flat = R_arr.reshape(-1, 3, 3)
    out = np.full((flat.shape[0], 3), np.nan, dtype=np.float32)
    for i, R in enumerate(flat):
        if np.any(np.isnan(R)):
            continue
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        out[i] = rvec.flatten().astype(np.float32)
    return out.reshape(out_shape)


def transform_points(X_world, R_w_t, t_w_t):
    """Apply X_table = R · X_world + t to a (..., 3) array. R is (3, 3),
    t is (3,). Equivalent to X_world @ R.T + t for row vectors."""
    return X_world @ R_w_t.T + t_w_t


def homogeneous_4x4(R, t):
    """Pack (R, t) into a 4x4 homogeneous matrix. R: (3, 3), t: (3,)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", default=str(DEFAULT_CANONICAL))
    ap.add_argument("--anchor", default=str(DEFAULT_ANCHOR))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    args = ap.parse_args()

    canonical_path = from_root(args.canonical)
    anchor_path = from_root(args.anchor)
    if not canonical_path.is_file():
        sys.exit(f"missing canonical npz: {canonical_path}")
    if not anchor_path.is_file():
        sys.exit(f"missing anchor npz: {anchor_path}")

    print(f"canonical: {canonical_path.name}")
    print(f"anchor:    {anchor_path.name}")

    canonical = np.load(canonical_path, allow_pickle=True)
    anchor = np.load(anchor_path, allow_pickle=True)

    # Required canonical keys (with helpful errors if the canonical pipeline
    # was run from a pre-MANO-export version).
    required_canonical = (
        "verts_3d_world", "wrist_3d_world",
        "pose_R_left", "pose_t_left",
        "mano_global_orient_R", "mano_hand_pose_R", "mano_betas",
        "handedness", "fps", "image_size", "calib_path",
    )
    for key in required_canonical:
        if key not in canonical.files:
            sys.exit(
                f"canonical npz missing key {key!r}.\n"
                f"  This usually means the canonical pipeline was run before\n"
                f"  the MANO-export commit. Re-run scripts/pipeline/08_wilor_canonical.py\n"
                f"  on the same clip to regenerate it with MANO fields included."
            )
    required_anchor = ("R_world_to_table", "t_world_to_table", "detected_left")
    for key in required_anchor:
        if key not in anchor.files:
            sys.exit(f"anchor npz missing key {key!r}")

    # Load calibration for intrinsics + extrinsics (so the export is
    # self-contained and downstream code doesn't need our calibration .npz).
    calib_path = Path(str(canonical["calib_path"]))
    if not calib_path.is_absolute():
        calib_path = PROJECT_ROOT / calib_path
    if not calib_path.is_file():
        sys.exit(f"calibration referenced by canonical not found: {calib_path}")
    calib = np.load(calib_path)

    # --- Resolve frame range ---
    n_canonical = int(canonical["verts_3d_world"].shape[0])
    n_anchor = int(anchor["R_world_to_table"].shape[0])
    n = min(n_canonical, n_anchor)
    if n_canonical != n_anchor:
        print(f"[warn] canonical has {n_canonical} frames, anchor has {n_anchor}; "
              f"truncating to {n}")
    nslot = canonical["verts_3d_world"].shape[1]
    fps = float(canonical["fps"])
    print(f"frames: {n} ({n / fps:.1f}s @ {fps:.1f} fps), slots: {nslot}")

    # --- Per-frame T_world_to_table as a homogeneous 4x4 ---
    R_w_t_arr = anchor["R_world_to_table"][:n]
    t_w_t_arr = anchor["t_world_to_table"][:n]
    anchored = anchor["detected_left"][:n].astype(bool)
    T_world_to_table = np.full((n, 4, 4), np.nan, dtype=np.float32)
    for i in np.where(anchored)[0]:
        if np.any(np.isnan(R_w_t_arr[i])) or np.any(np.isnan(t_w_t_arr[i])):
            continue
        T_world_to_table[i] = homogeneous_4x4(R_w_t_arr[i], t_w_t_arr[i])

    # --- Vertices and wrist in table frame ---
    verts_world = canonical["verts_3d_world"][:n]   # (n, slot, 778, 3) m
    wrist_world = canonical["wrist_3d_world"][:n]   # (n, slot, 3)
    vertices_table = np.full(verts_world.shape, np.nan, dtype=np.float32)
    wrist_table = np.full(wrist_world.shape, np.nan, dtype=np.float32)
    for i in np.where(anchored)[0]:
        R = R_w_t_arr[i]
        t = t_w_t_arr[i]
        if np.any(np.isnan(R)) or np.any(np.isnan(t)):
            continue
        for s in range(nslot):
            v = verts_world[i, s]
            if np.any(np.isnan(v)):
                continue
            vertices_table[i, s] = transform_points(v, R, t)
            wrist_table[i, s] = transform_points(wrist_world[i, s], R, t)

    # --- Wrist orientation in table frame ---
    # MANO local axes → origLeft camera = R_pnp_left.
    # origLeft → world (rectified-left) = R1 (the rectification rotation).
    # world → table = R_w_t (from anchor).
    # Composed: R_wrist_in_table = R_w_t · R1 · R_pnp_left.
    R1 = calib["R1"]
    pose_R_left = canonical["pose_R_left"][:n]
    wrist_R_table = np.full((n, nslot, 3, 3), np.nan, dtype=np.float32)
    for i in np.where(anchored)[0]:
        R_w_t = R_w_t_arr[i]
        if np.any(np.isnan(R_w_t)):
            continue
        for s in range(nslot):
            R_pnp = pose_R_left[i, s]
            if np.any(np.isnan(R_pnp)):
                continue
            wrist_R_table[i, s] = (R_w_t @ R1 @ R_pnp).astype(np.float32)

    # --- MANO θ as axis-angle ---
    # canonical saves rotation matrices for fidelity; the standard MANO θ
    # convention is axis-angle (B, 16, 3). Pack global_orient + hand_pose.
    go_R = canonical["mano_global_orient_R"][:n]   # (n, slot, 1, 3, 3)
    hp_R = canonical["mano_hand_pose_R"][:n]        # (n, slot, 15, 3, 3)
    pose_R = np.concatenate([go_R, hp_R], axis=2)  # (n, slot, 16, 3, 3)
    mano_pose = rotation_matrices_to_axis_angle(pose_R)  # (n, slot, 16, 3)
    mano_betas = canonical["mano_betas"][:n].astype(np.float32)

    # --- hand_present mask ---
    # A slot is present if anchor was good and the canonical fused a mesh.
    hand_present = np.zeros((n, nslot), dtype=bool)
    valid_canon = ~np.isnan(canonical["wrist_3d_world"][:n, :, 0])
    hand_present = anchored[:, None] & valid_canon

    # --- Other constants ---
    handedness = canonical["handedness"][:n]
    timestamps_s = (np.arange(n) / fps).astype(np.float32)

    # MANO faces — 1538 triangles, vertex topology. Pre-extracted once from
    # MANO_RIGHT.pkl into _lib/mano_faces.npy (the pkl needs chumpy + scipy
    # to unpickle; the .npy is a plain int32 array, no deps).
    faces_npy = PROJECT_ROOT / "scripts" / "_lib" / "mano_faces.npy"
    if not faces_npy.is_file():
        sys.exit(f"missing {faces_npy} — re-run the one-time face extraction:\n"
                 f"  .venv-hamer/bin/python -c \"...wilor_setup.get_mano_faces(None)...\"")
    mano_faces = np.load(faces_npy).astype(np.int32)

    # --- Output path ---
    out_path = PROJECT_ROOT / f"outputs/{today_pretty()} - dataset [{args.tag}].npz"

    np.savez(
        out_path,
        # Per-frame per-slot
        mano_pose=mano_pose,
        mano_betas=mano_betas,
        vertices_table=vertices_table.astype(np.float32),
        wrist_table=wrist_table.astype(np.float32),
        wrist_R_table=wrist_R_table,
        hand_present=hand_present,
        handedness=np.array(handedness.tolist()),
        # Per-frame
        T_world_to_table=T_world_to_table,
        anchored=anchored,
        timestamps_s=timestamps_s,
        # Constants
        intrinsics_left=calib["K_left"].astype(np.float32),
        intrinsics_right=calib["K_right"].astype(np.float32),
        distortion_left=calib["dist_left"].astype(np.float32).flatten(),
        distortion_right=calib["dist_right"].astype(np.float32).flatten(),
        stereo_R=calib["R"].astype(np.float32),
        stereo_T=calib["T"].astype(np.float32).flatten(),
        baseline_m=np.array(float(calib["baseline_m"]), dtype=np.float32),
        fps=np.array(fps, dtype=np.float32),
        image_size=np.asarray(canonical["image_size"], dtype=np.int32),
        mano_faces=mano_faces,
        # Provenance
        schema_version=SCHEMA_VERSION,
        source_canonical=str(canonical_path),
        source_anchor=str(anchor_path),
        source_calibration=str(calib_path),
        generated_on=today_pretty(),
    )

    # --- Summary ---
    print()
    print(f"=== exported dataset → {out_path.name} ===")
    print(f"frames:           {n} ({n/fps:.1f} s @ {fps:.1f} fps)")
    print(f"anchored frames:  {anchored.sum()} ({100*anchored.sum()/max(n,1):.1f}%)")
    for s in range(nslot):
        present = hand_present[:, s].sum()
        if present == 0:
            print(f"slot {s}:           0 frames")
            continue
        wt = wrist_table[hand_present[:, s], s]
        print(f"slot {s}:           {present} frames "
              f"({100*present/max(n,1):.1f}%)")
        print(f"  wrist X (cm):   median {np.median(wt[:, 0])*100:+6.1f}  "
              f"5-95 [{np.percentile(wt[:,0],5)*100:+.1f}, "
              f"{np.percentile(wt[:,0],95)*100:+.1f}]")
        print(f"  wrist Y (cm):   median {np.median(wt[:, 1])*100:+6.1f}  "
              f"5-95 [{np.percentile(wt[:,1],5)*100:+.1f}, "
              f"{np.percentile(wt[:,1],95)*100:+.1f}]")
        print(f"  wrist Z (cm):   median {np.median(wt[:, 2])*100:+6.1f}  "
              f"5-95 [{np.percentile(wt[:,2],5)*100:+.1f}, "
              f"{np.percentile(wt[:,2],95)*100:+.1f}]")
        # Mesh extent sanity (~14-22 cm for a real hand).
        verts = vertices_table[hand_present[:, s], s]   # (n_present, 778, 3)
        per_frame_extent = (verts.max(axis=1) - verts.min(axis=1)).max(axis=1) * 100
        print(f"  mesh extent (cm): median {np.median(per_frame_extent):.1f}, "
              f"5-95 [{np.percentile(per_frame_extent,5):.1f}, "
              f"{np.percentile(per_frame_extent,95):.1f}]  (real hand ~14-22)")
    print()
    print(f"file size:        {out_path.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
