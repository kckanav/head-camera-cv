"""MediaPipe Hands sanity check on a short clip (Tasks API)."""

import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from dated import today_pretty

MODEL_PATH = "hand_landmarker.task"
TODAY = today_pretty()

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
    (9, 13), (13, 14), (14, 15), (15, 16),# ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                              # palm base
]


def draw_hand(frame, landmarks, handed_label, w, h, color):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, color, 1)
    cv2.putText(frame, handed_label, (pts[0][0], pts[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def annotate_clip(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise FileNotFoundError(in_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    detected_frames = 0
    t0 = time.time()
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(1000 * i / fps)
        res = landmarker.detect_for_video(mp_image, ts_ms)

        if res.hand_landmarks:
            detected_frames += 1
            for lms, handed in zip(res.hand_landmarks, res.handedness):
                label = handed[0].category_name
                score = handed[0].score
                color = (0, 200, 255) if label == "Right" else (255, 100, 0)
                draw_hand(frame, lms, f"{label} {score:.2f}", w, h, color)

        writer.write(frame)
        if i % 30 == 0:
            sys.stdout.write(f"\r{in_path}: {i+1}/{n}  ")
            sys.stdout.flush()

    cap.release()
    writer.release()
    landmarker.close()
    elapsed = time.time() - t0
    pct = 100 * detected_frames / max(n, 1)
    print(f"\n{in_path} -> {out_path}: {detected_frames}/{n} frames had hands ({pct:.1f}%), {elapsed:.1f}s")


if __name__ == "__main__":
    annotate_clip(
        f"inputs/25th April 2026 - cam0 clip 1m40-1m55.mp4",
        f"outputs/{TODAY} - cam0 hands annotated.mp4",
    )
    annotate_clip(
        f"inputs/25th April 2026 - cam1 clip 1m40-1m55.mp4",
        f"outputs/{TODAY} - cam1 hands annotated.mp4",
    )
