"""Generate a print-ready ArUco marker PDF for table-frame anchoring (stage 8 / Phase 4-5).

Geometry rationale (capture at 2304x1296, ~100 deg HFOV wide cam, ~50 cm working distance):
  - 80 mm marker side -> ~150 px in capture at 50 cm. Comfortably above the
    detection floor (~50 px) and still ~95 px at 80 cm. Pose accuracy scales with
    apparent pixel size, so larger is strictly better - 80 mm is the comfort/space
    trade-off for table use.
  - 4x4 ArUco dict, ID 0 -> minimal grid complexity for fast detection. 50 IDs
    is overkill for a single marker, but the dict is still tiny and stable.
  - 10 mm white quiet zone on all sides of the marker (the detector needs an
    unbroken white border at least 1 module wide; 1 module = 80/(4+2) = ~13 mm
    here, so 10 mm is the floor).

Output is an A4-portrait PDF rendered at 600 DPI. Print at 100% scale (no
"fit to page"). The page includes a 100 mm scale bar - measure it with a ruler
to verify the print came out at true size, then ALSO measure the marker side
and tell the table_anchor.py script the actual measured length (1 mm error
~ 1% scale error in the table frame).
"""

import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty

MARKER_ID = 0
MARKER_MM = 80.0
DICT_ID = aruco.DICT_4X4_50

A4_W_MM, A4_H_MM = 210.0, 297.0  # portrait
PAGE_MARGIN_MM = 6.0
DPI = 600


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def generate_marker_image() -> np.ndarray:
    dictionary = aruco.getPredefinedDictionary(DICT_ID)
    side_px = mm_to_px(MARKER_MM)
    return aruco.generateImageMarker(dictionary, MARKER_ID, side_px)


def compose_page(marker_img: np.ndarray) -> np.ndarray:
    page_w_px = mm_to_px(A4_W_MM)
    page_h_px = mm_to_px(A4_H_MM)
    page = 255 * np.ones((page_h_px, page_w_px), dtype=np.uint8)

    # Marker centered horizontally, ~70 mm from top - leaves room for spec text
    # above (so it's not swallowed by the marker's quiet zone) and scale bar below.
    mx = (page_w_px - marker_img.shape[1]) // 2
    my = mm_to_px(70.0)
    page[my:my + marker_img.shape[0], mx:mx + marker_img.shape[1]] = marker_img

    # Spec line at the very top of the page.
    spec = (
        f"ArUco DICT_4X4_50 ID={MARKER_ID}  marker={MARKER_MM:.0f}mm  "
        f"PRINT AT 100% SCALE - bar below should measure 100mm"
    )
    cv2.putText(
        page, spec,
        (mm_to_px(PAGE_MARGIN_MM), mm_to_px(PAGE_MARGIN_MM + 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=DPI / 240,
        color=0,
        thickness=max(2, DPI // 240),
        lineType=cv2.LINE_AA,
    )

    # Reminder under the marker to measure the actual printed side - feeds the
    # table_anchor.py --marker-mm flag.
    reminder = "After printing, measure the marker side with a ruler and pass that exact value to table_anchor.py"
    cv2.putText(
        page, reminder,
        (mm_to_px(PAGE_MARGIN_MM), my + marker_img.shape[0] + mm_to_px(12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=DPI / 300,
        color=0,
        thickness=max(2, DPI // 300),
        lineType=cv2.LINE_AA,
    )

    # 100 mm scale bar in bottom-left margin to verify print scale with a ruler.
    bar_x0 = mm_to_px(PAGE_MARGIN_MM)
    bar_y = page_h_px - mm_to_px(PAGE_MARGIN_MM + 4)
    bar_x1 = bar_x0 + mm_to_px(100.0)
    bar_thick = mm_to_px(1.5)
    page[bar_y - bar_thick // 2:bar_y + bar_thick // 2, bar_x0:bar_x1] = 0
    tick_h = mm_to_px(2.5)
    for mm in range(0, 101, 10):
        x = bar_x0 + mm_to_px(mm)
        page[bar_y - tick_h:bar_y + tick_h, x - bar_thick // 2:x + bar_thick // 2] = 0

    cv2.putText(
        page, "100 mm",
        (bar_x1 + mm_to_px(3), bar_y + mm_to_px(2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=DPI / 240,
        color=0,
        thickness=max(2, DPI // 240),
        lineType=cv2.LINE_AA,
    )
    return page


def main():
    marker_img = generate_marker_image()
    page = compose_page(marker_img)

    out_path = f"outputs/{today_pretty()} - aruco table marker (A4 portrait).pdf"
    pil = Image.fromarray(page)
    pil.save(out_path, "PDF", resolution=DPI)
    print(f"wrote {out_path}")
    print(f"page: {A4_W_MM:.0f}x{A4_H_MM:.0f}mm @ {DPI} DPI ({page.shape[1]}x{page.shape[0]} px)")
    print(f"marker: DICT_4X4_50 ID={MARKER_ID}, {MARKER_MM:.0f}x{MARKER_MM:.0f}mm "
          f"({marker_img.shape[1]}x{marker_img.shape[0]} px)")


if __name__ == "__main__":
    main()
