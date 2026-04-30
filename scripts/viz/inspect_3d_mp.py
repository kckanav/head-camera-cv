"""Quick visualization of the triangulated 3D hand data.

Loads outputs/<date> - stereo hand 3d.npz (produced by triangulate.py) and
plots wrist depth and thumb-index pinch distance over time, one curve per
hand slot. Useful to eyeball whether the per-frame estimates form smooth
trajectories or are jittery / dropping in/out.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty

NPZ_PATH = f"outputs/{today_pretty()} - stereo hand 3d.npz"
OUT_PNG = f"outputs/{today_pretty()} - stereo hand 3d trajectory.png"

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
COLORS = ["#ffa500", "#1f77b4"]   # match the BGR (0,200,255) and (255,100,0) used in the video


def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    L = data["landmarks_3d"]      # (N, 2, 21, 3) metres
    fps = float(data["fps"])
    n = L.shape[0]
    t = np.arange(n) / fps

    wrist_z = L[:, :, WRIST, 2] * 100.0                      # (N, 2) cm
    pinch = np.linalg.norm(L[:, :, THUMB_TIP] - L[:, :, INDEX_TIP], axis=-1) * 1000.0  # mm

    detected = (~np.isnan(wrist_z)).sum(axis=0)
    print(f"frames: {n}, duration {t[-1]:.1f}s")
    print(f"hand slot 0: {detected[0]} frames ({100*detected[0]/n:.1f}%)")
    print(f"hand slot 1: {detected[1]} frames ({100*detected[1]/n:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for slot in range(2):
        ax1.plot(t, wrist_z[:, slot], ".", color=COLORS[slot], markersize=2,
                 label=f"hand slot {slot} ({detected[slot]} frames)")
        ax2.plot(t, pinch[:, slot], ".", color=COLORS[slot], markersize=2)

    ax1.set_ylabel("wrist depth (cm)")
    ax1.set_ylim(0, 80)
    ax1.axhspan(30, 50, color="green", alpha=0.05, label="typical hand-interaction range")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("thumb-index pinch (mm)")
    ax2.set_ylim(0, 200)
    ax2.set_xlabel("time (s)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Sparse 3D hand keypoints from stereo triangulation")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"wrote {OUT_PNG}")

    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", OUT_PNG], check=False)


if __name__ == "__main__":
    main()
