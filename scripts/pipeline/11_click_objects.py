"""Stage 9 / Phase 1: interactive click UI to mark trackable objects in
frame 0 of a stereo pair. Drives the SAM2 video segmenter downstream.

You click each object once in the LEFT rectified view, type its label in
the terminal, and the script auto-recovers the matching click in the
RIGHT view by template-matching along the rectified epipolar row (so you
get one click per object, not two).

The output JSON lists per-object {label, left_xy, right_xy} for frame 0;
`12_segment_objects.py` reads it as the seed prompts for SAM2's video
predictor on each view independently.

Run on the Mac side (CPU only; the heavy GPU pass comes later):

    .venv/bin/python scripts/pipeline/11_click_objects.py
    .venv/bin/python scripts/pipeline/11_click_objects.py --frame 30
    .venv/bin/python scripts/pipeline/11_click_objects.py \\
        --cam0 inputs/...mp4 --cam1 inputs/...mp4 \\
        --calib outputs/...npz --tag "27th April interaction"

Output:
    outputs/<date> - object clicks [<tag>].json
    {
      "left_video":  "...",
      "right_video": "...",
      "calib":       "...",
      "frame":       0,
      "objects": [
        {"label": "cup",   "left_xy": [x, y], "right_xy": [x, y]},
        {"label": "plate", "left_xy": [x, y], "right_xy": [x, y]},
        ...
      ]
    }

Convention: cam1 = LEFT, cam0 = RIGHT (matches the rest of the repo).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CAM0 = (PROJECT_ROOT
    / "inputs/27th April 2026 wide - cam0 interaction first 10s.mp4")
DEFAULT_CAM1 = (PROJECT_ROOT
    / "inputs/27th April 2026 wide - cam1 interaction first 10s.mp4")
DEFAULT_CALIB = (PROJECT_ROOT
    / "outputs/30th April 2026 wide - stereo calibration.npz")
DEFAULT_TAG = "27th April interaction"

# Template-matching parameters for auto-recovering the right-view click.
# patch_px wide enough to capture local texture; search_rows accommodates
# the residual stereo misalignment after rectification (typical < 1 px,
# ±5 is generous).
PATCH_PX = 31
SEARCH_ROWS = 5


def from_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def grab_frame(path: Path, idx: int) -> np.ndarray:
    """Read a single BGR frame at `idx` from a video file."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"failed to read frame {idx} of {path}")
    return frame


def rectify(frame: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


def match_right_click(left_xy: tuple[float, float],
                      left_gray: np.ndarray,
                      right_gray: np.ndarray,
                      patch: int = PATCH_PX,
                      search_rows: int = SEARCH_ROWS,
                      ) -> tuple[float, float] | None:
    """Find the right-view click for a left-view click, by NCC template
    matching along the rectified epipolar row (±search_rows of slack to
    absorb residual rectification error).

    Returns None if the click is too close to the image edge for a full
    template patch.
    """
    x, y = int(round(left_xy[0])), int(round(left_xy[1]))
    h, w = left_gray.shape
    half = patch // 2
    if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
        return None

    template = left_gray[y - half:y + half + 1, x - half:x + half + 1]

    # Full-width strip in the right image, ±search_rows around the same y.
    strip_y0 = max(0, y - half - search_rows)
    strip_y1 = min(h, y + half + search_rows + 1)
    strip = right_gray[strip_y0:strip_y1, :]
    if strip.shape[0] < template.shape[0] or strip.shape[1] < template.shape[1]:
        return None

    res = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, top_left = cv2.minMaxLoc(res)
    rx = float(top_left[0]) + half
    ry = float(strip_y0 + top_left[1]) + half
    return (rx, ry)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cam0", default=str(DEFAULT_CAM0),
                    help="right-view video (cam0)")
    ap.add_argument("--cam1", default=str(DEFAULT_CAM1),
                    help="left-view video (cam1)")
    ap.add_argument("--calib", default=str(DEFAULT_CALIB),
                    help="stereo calibration .npz")
    ap.add_argument("--frame", type=int, default=0,
                    help="frame index used as click reference (default 0)")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="output filename tag")
    ap.add_argument("--out", default=None,
                    help="output JSON path "
                         "(default outputs/<today> - object clicks [<tag>].json)")
    args = ap.parse_args()

    cam0_path = from_root(args.cam0)
    cam1_path = from_root(args.cam1)
    calib_path = from_root(args.calib)
    for p in (cam0_path, cam1_path, calib_path):
        if not p.is_file():
            sys.exit(f"missing: {p}")

    out_path = (Path(args.out).expanduser().resolve() if args.out
                else PROJECT_ROOT
                     / f"outputs/{today_pretty()} - object clicks [{args.tag}].json")

    print(f"left  (cam1): {cam1_path.name}")
    print(f"right (cam0): {cam0_path.name}")
    print(f"calib:        {calib_path.name}")
    print(f"frame:        {args.frame}")
    print(f"output:       {out_path.relative_to(PROJECT_ROOT) if out_path.is_relative_to(PROJECT_ROOT) else out_path}")
    print()

    calib = np.load(calib_path, allow_pickle=True)
    rect_left = rectify(grab_frame(cam1_path, args.frame),
                        calib["map_left_x"], calib["map_left_y"])
    rect_right = rectify(grab_frame(cam0_path, args.frame),
                         calib["map_right_x"], calib["map_right_y"])

    rect_left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB)
    rect_right_rgb = cv2.cvtColor(rect_right, cv2.COLOR_BGR2RGB)
    rect_left_gray = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    rect_right_gray = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    objects: list[dict] = []

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(rect_left_rgb)
    axes[0].set_title(f"LEFT (rectified, frame {args.frame})\n"
                      "click on each object — type its label in the terminal")
    axes[1].imshow(rect_right_rgb)
    axes[1].set_title("RIGHT (rectified) — auto-matched via epipolar NCC")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    # Single-flag re-entrancy guard: matplotlib delivers click events
    # before input() returns, and we don't want a second click while the
    # terminal is still asking for the first click's label.
    state = {"awaiting": True}

    def on_click(event):
        if event.inaxes is not axes[0] or event.button != 1:
            return
        if not state["awaiting"]:
            return
        state["awaiting"] = False

        x, y = float(event.xdata), float(event.ydata)
        right = match_right_click((x, y), rect_left_gray, rect_right_gray)
        if right is None:
            print("  (click too close to image edge — try again)")
            state["awaiting"] = True
            return

        n = len(objects)
        color = plt.cm.tab10(n % 10)
        axes[0].plot(x, y, "+", color=color, markersize=20, mew=2.5)
        axes[0].text(x + 8, y - 8, str(n), color=color, fontsize=12, weight="bold")
        axes[1].plot(*right, "x", color=color, markersize=16, mew=2.5)
        axes[1].text(right[0] + 8, right[1] - 8, str(n),
                     color=color, fontsize=12, weight="bold")
        fig.canvas.draw_idle()
        plt.pause(0.05)   # flush draws before blocking on input()

        try:
            label = input(f"  label for object #{n} (or empty to discard): ").strip()
        except EOFError:
            label = ""
        if not label:
            print("  (discarded)")
            # Strip the markers we just drew so the discard is visible.
            for ax in axes:
                if ax.lines:
                    ax.lines[-1].remove()
                if ax.texts:
                    ax.texts[-1].remove()
            fig.canvas.draw_idle()
            state["awaiting"] = True
            return

        objects.append({
            "label": label,
            "left_xy":  [x, y],
            "right_xy": list(right),
        })
        state["awaiting"] = True
        print(f"  saved '{label}' — click another, or close the window when done.")

    fig.canvas.mpl_connect("button_press_event", on_click)

    print("Click each object in the LEFT view. Close the window when done.")
    print()
    plt.show()

    if not objects:
        sys.exit("no objects clicked — nothing to save")

    record = {
        "left_video":  str(cam1_path.relative_to(PROJECT_ROOT)
                           if cam1_path.is_relative_to(PROJECT_ROOT) else cam1_path),
        "right_video": str(cam0_path.relative_to(PROJECT_ROOT)
                           if cam0_path.is_relative_to(PROJECT_ROOT) else cam0_path),
        "calib":       str(calib_path.relative_to(PROJECT_ROOT)
                           if calib_path.is_relative_to(PROJECT_ROOT) else calib_path),
        "frame":       int(args.frame),
        "objects":     objects,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))

    print()
    print(f"=== wrote {len(objects)} objects → {out_path} ===")
    for o in objects:
        print(f"  {o['label']:<20s} "
              f"L=({o['left_xy'][0]:6.1f}, {o['left_xy'][1]:6.1f})   "
              f"R=({o['right_xy'][0]:6.1f}, {o['right_xy'][1]:6.1f})")


if __name__ == "__main__":
    main()
