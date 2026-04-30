"""Generate a print-ready ChArUco calibration board PDF for the Pi Cam 3 rig.

Geometry rationale (capture at 1280x720, ~75 deg HFOV, 30-80 cm working distance):
  - 30 mm squares -> ~42 px per square in capture at 60 cm. Comfortably above the
    sub-pixel corner-detection floor (~20 px) and still ~30 px at 80 cm.
  - 9x6 squares -> 40 inner ChArUco corners, 27 ArUco markers. Plenty of
    constraints for calibration without going A3.
  - 5x5 ArUco dict -> more error correction than 4x4; 50 IDs is plenty.
  - Marker/square ratio 22/30 = 0.73, in OpenCV's recommended range.

Output is an A4-landscape PDF rendered at 600 DPI. Print at 100% scale (no
"fit to page"). The page includes a 100 mm scale bar - measure it with a ruler
to verify the print came out at true size.
"""

import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from dated import today_pretty

SQUARES_X = 9
SQUARES_Y = 6
SQUARE_MM = 30.0
MARKER_MM = 22.0
DICT_ID = aruco.DICT_5X5_50

A4_W_MM, A4_H_MM = 297.0, 210.0  # landscape
PAGE_MARGIN_MM = 6.0  # outer margin
DPI = 600


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def generate_board_image() -> np.ndarray:
    dictionary = aruco.getPredefinedDictionary(DICT_ID)
    board = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        squareLength=SQUARE_MM / 1000.0,
        markerLength=MARKER_MM / 1000.0,
        dictionary=dictionary,
    )
    board_w_px = mm_to_px(SQUARES_X * SQUARE_MM)
    board_h_px = mm_to_px(SQUARES_Y * SQUARE_MM)
    return board.generateImage((board_w_px, board_h_px), marginSize=0)


def compose_page(board_img: np.ndarray) -> np.ndarray:
    page_w_px = mm_to_px(A4_W_MM)
    page_h_px = mm_to_px(A4_H_MM)
    page = 255 * np.ones((page_h_px, page_w_px), dtype=np.uint8)

    # Center the board on the page.
    bx = (page_w_px - board_img.shape[1]) // 2
    by = (page_h_px - board_img.shape[0]) // 2
    page[by:by + board_img.shape[0], bx:bx + board_img.shape[1]] = board_img

    # 100 mm scale bar in bottom-left margin to verify print scale with a ruler.
    bar_x0 = mm_to_px(PAGE_MARGIN_MM)
    bar_y = page_h_px - mm_to_px(PAGE_MARGIN_MM + 4)
    bar_x1 = bar_x0 + mm_to_px(100.0)
    bar_thick = mm_to_px(1.5)
    page[bar_y - bar_thick // 2:bar_y + bar_thick // 2, bar_x0:bar_x1] = 0
    # Tick marks every 10 mm.
    tick_h = mm_to_px(2.5)
    for mm in range(0, 101, 10):
        x = bar_x0 + mm_to_px(mm)
        page[bar_y - tick_h:bar_y + tick_h, x - bar_thick // 2:x + bar_thick // 2] = 0

    # Labels (rendered in OpenCV - vector text would be nicer but this is fine).
    spec = (
        f"ChArUco {SQUARES_X}x{SQUARES_Y}  square={SQUARE_MM:.0f}mm  "
        f"marker={MARKER_MM:.0f}mm  dict=DICT_5X5_50    "
        f"PRINT AT 100% SCALE - the bar below should measure 100mm"
    )
    cv2.putText(
        page, spec,
        (mm_to_px(PAGE_MARGIN_MM), mm_to_px(PAGE_MARGIN_MM + 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=DPI / 200,  # ~3.0 at 600 DPI
        color=0,
        thickness=max(2, DPI // 200),
        lineType=cv2.LINE_AA,
    )
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
    board_img = generate_board_image()
    page = compose_page(board_img)

    out_path = f"outputs/{today_pretty()} - charuco calibration board (A4 landscape).pdf"
    pil = Image.fromarray(page)
    # Saving with resolution=DPI tells the PDF that <pixels> / <DPI> = inches,
    # so the page comes out exactly A4 in physical size.
    pil.save(out_path, "PDF", resolution=DPI)
    print(f"wrote {out_path}")
    print(f"page: {A4_W_MM:.0f}x{A4_H_MM:.0f}mm @ {DPI} DPI ({page.shape[1]}x{page.shape[0]} px)")
    print(f"board: {SQUARES_X}x{SQUARES_Y} squares, {SQUARE_MM:.0f}mm each "
          f"({SQUARES_X * SQUARE_MM:.0f}x{SQUARES_Y * SQUARE_MM:.0f}mm)")


if __name__ == "__main__":
    main()
