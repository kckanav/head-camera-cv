"""Stage 8 / Phase 4 quick-look: wrist trajectory in table frame.

Combines two existing per-frame outputs to visualise the canonical pipeline
*through the table-frame anchor*:

  outputs/<date> - phase3 anchored fused [<tag>].npz   (canonical pipeline)
  outputs/<date> - table anchor [<tag>].npz            (table anchor)

For each frame where both detections are present, the wrist 3D position
(in rectified-left frame) is transformed into the table frame using the
per-frame T_world_to_table the anchor saved. Then we plot:

  1. Top-down (table X vs Y): wrist trajectory on the table plane.
     Marker centre is (0, 0) by construction. The first thing to look at —
     does the trajectory live in a plausible region of the table?
  2. Side view (table X vs Z): height profile. Z is height above the
     table; should be > 0 (hand above marker plane) except when the
     wrist is right at the marker level.
  3. Z over time: same data, time-axis. Lets you spot reach / pickup /
     release events as Z dips and rises.
  4. Presence strip: per-slot wrist availability over time, plus an
     anchor-only band (frames where the marker was anchored but no hand
     pair was found). Used to sanity-check that the protocol's static /
     pan / occlusion segments line up with what you intended.

Run:
  .venv/bin/python scripts/viz/inspect_anchored.py
  .venv/bin/python scripts/viz/inspect_anchored.py \\
      --canonical "outputs/<date> - phase3 anchored fused [foo].npz" \\
      --anchor    "outputs/<date> - table anchor [foo].npz" \\
      --tag       "foo"
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CANONICAL = PROJECT_ROOT / "outputs/30th April 2026 - phase3 anchored fused [20260430 protocol-A].npz"
DEFAULT_ANCHOR = PROJECT_ROOT / "outputs/30th April 2026 - table anchor [20260430 protocol-A clean].npz"
DEFAULT_TAG = "20260430 protocol-A"

# Match the BGR colours the canonical pipeline uses for slots in the SBS video:
# slot 0 = (0, 200, 255) BGR = orange in display; slot 1 = (255, 100, 0) BGR = blue.
SLOT_COLORS = ["#ffa500", "#1f77b4"]


def from_root(p):
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def transform_world_to_table(X_world: np.ndarray, R_w_t: np.ndarray, t_w_t: np.ndarray) -> np.ndarray:
    """Apply X_table = R_w_t · X_world + t_w_t to an array of shape (..., 3).

    R_w_t is (3, 3); t_w_t is (3,). For row vectors X, the equivalent matrix
    form is X @ R_w_t.T + t_w_t.
    """
    return X_world @ R_w_t.T + t_w_t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", default=str(DEFAULT_CANONICAL),
                    help="phase3 anchored fused .npz from 08_wilor_canonical.py")
    ap.add_argument("--anchor", default=str(DEFAULT_ANCHOR),
                    help="table anchor .npz from 09_anchor_table.py")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="output filename tag")
    ap.add_argument("--no-open", action="store_true",
                    help="don't auto-open the PNG on macOS")
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

    for key in ("wrist_3d_world", "fps"):
        if key not in canonical.files:
            sys.exit(f"canonical npz missing key {key!r}")
    for key in ("R_world_to_table", "t_world_to_table", "detected_left"):
        if key not in anchor.files:
            sys.exit(f"anchor npz missing key {key!r}")

    wrist_world = canonical["wrist_3d_world"]   # (N, NSLOT, 3)
    R_w_t_arr = anchor["R_world_to_table"]       # (Nt, 3, 3)
    t_w_t_arr = anchor["t_world_to_table"]       # (Nt, 3)
    anchored = anchor["detected_left"]           # (Nt,) bool

    fps = float(canonical["fps"])
    n = int(min(wrist_world.shape[0], R_w_t_arr.shape[0]))
    n_slot = wrist_world.shape[1]
    print(f"frames: {n}  slots: {n_slot}  fps: {fps:.1f}")

    # Transform wrist into table frame, NaN-padded where missing.
    wrist_table = np.full((n, n_slot, 3), np.nan, dtype=np.float32)
    for i in range(n):
        if not anchored[i]:
            continue
        R = R_w_t_arr[i]
        t = t_w_t_arr[i]
        if np.any(np.isnan(R)) or np.any(np.isnan(t)):
            continue
        for s in range(n_slot):
            w = wrist_world[i, s]
            if np.any(np.isnan(w)):
                continue
            wrist_table[i, s] = (R @ w) + t

    valid = ~np.isnan(wrist_table[..., 0])
    valid_per_slot = valid.sum(axis=0)
    print(f"wrist + anchor present: slot 0 = {valid_per_slot[0]}, "
          f"slot 1 = {valid_per_slot[1]}")
    if valid.any():
        z_cm = wrist_table[..., 2][valid] * 100
        x_cm = wrist_table[..., 0][valid] * 100
        y_cm = wrist_table[..., 1][valid] * 100
        print(f"wrist X (cm):  median {np.median(x_cm):+.1f}  "
              f"5-95 [{np.percentile(x_cm, 5):+.1f}, {np.percentile(x_cm, 95):+.1f}]")
        print(f"wrist Y (cm):  median {np.median(y_cm):+.1f}  "
              f"5-95 [{np.percentile(y_cm, 5):+.1f}, {np.percentile(y_cm, 95):+.1f}]")
        print(f"wrist Z (cm):  median {np.median(z_cm):+.1f}  "
              f"5-95 [{np.percentile(z_cm, 5):+.1f}, {np.percentile(z_cm, 95):+.1f}]")
        n_below_table = int((z_cm < -1).sum())
        if n_below_table:
            print(f"  ⚠ {n_below_table} samples have Z < -1 cm "
                  f"(below table — penetration or tracking error)")

    # ---- plot ----
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2,
                          height_ratios=[3.0, 2.0, 1.2],
                          hspace=0.45, wspace=0.25)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_zt = fig.add_subplot(gs[1, :])
    ax_pres = fig.add_subplot(gs[2, :])

    # Top-down (X, Y) on table plane
    for s in range(n_slot):
        m = valid[:, s]
        if not m.any():
            continue
        ax_xy.plot(wrist_table[m, s, 0] * 100, wrist_table[m, s, 1] * 100,
                   ".", color=SLOT_COLORS[s], markersize=2.5,
                   label=f"slot {s} ({int(m.sum())} fr)")
    ax_xy.scatter([0], [0], color="black", s=80, marker="x", zorder=5,
                  label="marker (0, 0)")
    ax_xy.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax_xy.axvline(0, color="black", lw=0.5, alpha=0.3)
    ax_xy.set_xlabel("table X (cm)")
    ax_xy.set_ylabel("table Y (cm)")
    ax_xy.set_title("Top-down: wrist on table plane")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best", fontsize=8)

    # Side view (X, Z): height profile
    for s in range(n_slot):
        m = valid[:, s]
        if not m.any():
            continue
        ax_xz.plot(wrist_table[m, s, 0] * 100, wrist_table[m, s, 2] * 100,
                   ".", color=SLOT_COLORS[s], markersize=2.5)
    ax_xz.axhline(0, color="black", lw=1.2, alpha=0.6, label="table surface (Z=0)")
    ax_xz.set_xlabel("table X (cm)")
    ax_xz.set_ylabel("table Z (cm) — height above table")
    ax_xz.set_title("Side view (X-Z)")
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend(loc="best", fontsize=8)

    # Z vs time
    t = np.arange(n) / fps
    for s in range(n_slot):
        ax_zt.plot(t, wrist_table[:, s, 2] * 100, ".", color=SLOT_COLORS[s],
                   markersize=2, label=f"slot {s}")
    ax_zt.axhline(0, color="black", lw=1.0, alpha=0.6)
    ax_zt.set_xlabel("time (s)")
    ax_zt.set_ylabel("wrist Z above table (cm)")
    ax_zt.set_title("Wrist height vs time — Z=0 is the table surface")
    ax_zt.grid(True, alpha=0.3)
    ax_zt.legend(loc="best", fontsize=8)

    # Presence strip: per-slot wrist + anchor-only band
    band_y = {0: 0.0, 1: 1.0, "anchor_only": 2.0}
    band_h = 0.7
    dt = 1.0 / fps
    for s in range(n_slot):
        for i in np.where(valid[:, s])[0]:
            ax_pres.add_patch(mpatches.Rectangle(
                (t[i] - dt / 2, band_y[s]), dt, band_h,
                color=SLOT_COLORS[s], linewidth=0))
    anchor_only = anchored[:n] & ~valid.any(axis=1)
    for i in np.where(anchor_only)[0]:
        ax_pres.add_patch(mpatches.Rectangle(
            (t[i] - dt / 2, band_y["anchor_only"]), dt, band_h,
            color="gray", alpha=0.5, linewidth=0))
    ax_pres.set_xlim(0, t[-1] if n > 0 else 1)
    ax_pres.set_ylim(-0.2, 3.0)
    ax_pres.set_yticks([band_h / 2, 1 + band_h / 2, 2 + band_h / 2])
    ax_pres.set_yticklabels(["slot 0", "slot 1", "anchor only\n(no hand pair)"])
    ax_pres.set_xlabel("time (s)")
    ax_pres.set_title("Presence over time")
    ax_pres.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"Phase 4 inspection — {args.tag}", fontsize=12, y=0.995)

    out_path = PROJECT_ROOT / f"outputs/{today_pretty()} - phase4 inspection [{args.tag}].png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path.name}")

    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(out_path)], check=False)


if __name__ == "__main__":
    main()
