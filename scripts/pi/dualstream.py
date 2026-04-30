"""Dual Pi Cam 3 Wide live preview + recording server, with sensor-mode switching.

Runs an HTTP server on :8080 that:
  - Streams MJPEG previews of both cameras at /cam0 and /cam1.
  - Exposes a small HTML UI with buttons to start/stop H.264 recording.
  - Lets you switch the underlying SENSOR mode (1536x864 cropped /
    2304x1296 binned full-FOV / 4608x2592 full-sensor full-FOV) so you can
    visually compare FOVs before committing to one for capture.

Why sensor mode matters: by default, picamera2 picks the smallest sensor
mode that can produce the requested main output (1280x720 here), which on
the IMX708 is the 1536x864 CENTRE-CROPPED mode - effectively ~75 deg HFOV
on the wide variant. The 2304x1296 binned mode reads the full sensor and
gives the camera's native ~102 deg HFOV at the same 1280x720 main output.

Caveats:
  - Switching mode changes FOV AND the intrinsic camera matrix, so the
    stereo calibration must be re-run after picking the final mode.
  - The 4608x2592 mode caps at ~14 fps; can't sustain 30 fps recording.
    The 2304x1296 mode caps at ~56 fps and IS full-FOV - the sweet spot.
"""

import io
import signal
import sys
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from threading import Condition, Lock
from urllib.parse import urlparse, parse_qs

from libcamera import Transform
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, MJPEGEncoder
from picamera2.outputs import FileOutput

PREVIEW_SIZE = (1280, 720)
TARGET_FPS = 30   # locked across all sensor modes that can sustain it; the
                  # 4608x2592 mode can't (caps at ~14 fps) and falls back to
                  # the mode's max.


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


# ---------------------------------------------------------------------------
# Sensor-mode discovery + camera (re)configuration
# ---------------------------------------------------------------------------
def discover_modes(cam):
    """Return list of (w, h, max_fps, full_fov) for the camera, dedup'd, sorted."""
    sensor_w = max(m["size"][0] for m in cam.sensor_modes)
    sensor_h = max(m["size"][1] for m in cam.sensor_modes)
    seen = {}
    for m in cam.sensor_modes:
        w, h = m["size"]
        crop = m.get("crop_limits", (0, 0, sensor_w, sensor_h))
        full_fov = (crop[0] == 0 and crop[1] == 0
                    and crop[2] == sensor_w and crop[3] == sensor_h)
        # If we've seen this (w, h) already, keep the entry with the higher fps.
        prev = seen.get((w, h))
        fps = m.get("fps", 0.0)
        if prev is None or fps > prev[2]:
            seen[(w, h)] = (w, h, fps, full_fov)
    return sorted(seen.values(), key=lambda x: x[0] * x[1])


def actual_fps_for_mode(raw_size):
    """Pick the recording fps: TARGET_FPS if the mode supports it, else the mode max."""
    mode_max = next((fps for (w, h, fps, _) in modes if (w, h) == raw_size), TARGET_FPS)
    return min(TARGET_FPS, int(mode_max))


def configure_for_mode(cam, raw_size):
    """Configure cam for video with the given raw sensor size + locked frame rate.

    Returns the fps the camera will actually run at (TARGET_FPS or mode max).
    """
    fps = actual_fps_for_mode(raw_size)
    frame_duration_us = int(round(1_000_000 / fps))
    cfg = cam.create_video_configuration(
        main={"size": PREVIEW_SIZE},
        lores={"size": PREVIEW_SIZE},
        raw={"size": raw_size},
        controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
        transform=Transform(hflip=True, vflip=True),
    )
    cam.configure(cfg)
    return fps


def start_preview(cam, output):
    cam.start_recording(MJPEGEncoder(), FileOutput(output))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
output0 = StreamingOutput()
output1 = StreamingOutput()
state_lock = Lock()
recording = False
rec_files = []
rec_encs = {}

picam0 = Picamera2(1)
picam1 = Picamera2(0)

modes = discover_modes(picam0)
# Default: largest mode that still hits >=30 fps. For IMX708 Wide that's the
# 2304x1296 binned full-FOV mode (~56 fps cap). 4608x2592 caps at ~14 fps so
# it can't sustain 30 fps recording.
candidates_30fps = [m for m in modes if m[2] >= 30.0]
default_mode = max(candidates_30fps, key=lambda m: m[0] * m[1]) if candidates_30fps else modes[-1]
current_raw = (default_mode[0], default_mode[1])

print("Available sensor modes (* = current default):")
for w, h, fps, full in modes:
    flag = "*" if (w, h) == current_raw else " "
    note = "FULL FOV" if full else "cropped FOV"
    will_record_at = min(TARGET_FPS, int(fps))
    print(f"  {flag} {w:4d} x {h:<4d}   max {fps:6.2f} fps   "
          f"records at {will_record_at} fps   {note}")

active_fps = configure_for_mode(picam0, current_raw)
start_preview(picam0, output0)
time.sleep(2)
configure_for_mode(picam1, current_raw)
start_preview(picam1, output1)


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Pi 5 Dual Camera</title>
<style>
  body { background:#111; color:#fff; font-family:sans-serif; margin:0; padding:0;
         display:flex; flex-direction:column; align-items:center; min-height:100vh; }
  h1 { margin:16px 0 8px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; justify-content:center;
         align-items:center; margin:6px 0; }
  button { padding:10px 14px; font-size:14px; border:none; border-radius:6px;
           cursor:pointer; font-family:inherit; }
  .rec      { background:#c00; color:#fff; } .rec:hover      { background:#e00; }
  .stop-btn { background:#555; color:#fff; } .stop-btn:hover { background:#777; }
  .mode     { background:#246; color:#fff; }
  .mode:hover { background:#357; }
  .mode.active { background:#4a8; outline:2px solid #fff; }
  #status { color:#aaf; font-family:monospace; margin-top:6px; min-height:1.2em; }
  .warn   { color:#fb6; font-size:12px; margin:4px 0; max-width:740px; text-align:center; }
  .container { display:flex; gap:16px; flex-wrap:wrap; justify-content:center; padding:12px; }
  .cam img { border-radius:6px; max-width:90vw; width:640px; }
  .cam p { color:#aaa; margin:6px 0; }
</style>
</head>
<body>
<h1>Dual Camera Stream</h1>

<div class="row">
  <strong>Sensor mode:</strong>
  __MODE_BUTTONS__
</div>
<div class="warn">
  Switching mode changes FOV and invalidates the stereo calibration.
  Re-run calibrate.py after settling on a final mode. The full 4608x2592
  mode caps at ~14 fps - use 2304x1296 (full FOV, ~56 fps cap) for 30 fps recording.
</div>

<div class="row">
  <button class="rec"      onclick="startRec()">Start Recording</button>
  <button class="stop-btn" onclick="stopRec()">Stop Recording</button>
</div>
<div id="status">current sensor mode: __CURRENT_MODE__</div>

<div class="container">
  <div class="cam"><img src="/cam0" /><p>Camera 0</p></div>
  <div class="cam"><img src="/cam1" /><p>Camera 1</p></div>
</div>

<script>
  function startRec() {
    fetch('/record/start').then(r => r.text())
      .then(t => document.getElementById('status').innerText = t);
  }
  function stopRec() {
    fetch('/record/stop').then(r => r.text())
      .then(t => document.getElementById('status').innerText = t);
  }
  function switchMode(w, h) {
    document.getElementById('status').innerText =
      'switching to ' + w + 'x' + h + ' ... live preview will reload in a moment';
    fetch('/switch_mode?w=' + w + '&h=' + h)
      .then(r => r.text())
      .then(t => {
        document.getElementById('status').innerText = t;
        setTimeout(function () { location.reload(); }, 1500);
      });
  }
</script>
</body>
</html>
"""


def render_html():
    btns = []
    for w, h, fps, full in modes:
        active = "active" if (w, h) == current_raw else ""
        note = "full FOV" if full else "cropped FOV"
        rec_fps = min(TARGET_FPS, int(fps))
        rec_note = f"records at {rec_fps} fps"
        btns.append(
            f'<button class="mode {active}" onclick="switchMode({w}, {h})">'
            f'{w} x {h} &mdash; {rec_note} &mdash; {note} '
            f'<span style="opacity:0.6">(mode max {fps:.0f} fps)</span>'
            f'</button>'
        )
    cur_w, cur_h = current_raw
    cur_full = next((full for (w, h, fps, full) in modes if (w, h) == current_raw), False)
    cur_fps = actual_fps_for_mode(current_raw)
    cur_label = (f"{cur_w} x {cur_h}, recording at {cur_fps} fps  "
                 f"({'FULL FOV' if cur_full else 'cropped FOV'})")
    return (HTML_PAGE
            .replace("__MODE_BUTTONS__", "\n        ".join(btns))
            .replace("__CURRENT_MODE__", cur_label))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class StreamingHandler(BaseHTTPRequestHandler):
    outputs = {'/cam0': output0, '/cam1': output1}
    cameras = {'picam0': picam0, 'picam1': picam1}

    def do_GET(self):
        global recording, rec_files, current_raw
        url = urlparse(self.path)

        if url.path == '/':
            content = render_html().encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
            return

        if url.path == '/record/start':
            self.send_response(200); self.send_header('Content-Type', 'text/plain'); self.end_headers()
            with state_lock:
                if recording:
                    self.wfile.write(b'already recording'); return
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                f0, f1 = f'cam0_{ts}.h264', f'cam1_{ts}.h264'
                rec_files = [f0, f1]
                enc0 = H264Encoder(); enc1 = H264Encoder()
                self.cameras['picam0'].start_encoder(enc0, FileOutput(f0), name='lores')
                self.cameras['picam1'].start_encoder(enc1, FileOutput(f1), name='lores')
                rec_encs['enc0'] = enc0; rec_encs['enc1'] = enc1
                recording = True
            self.wfile.write(f'recording started: {f0}, {f1}'.encode()); return

        if url.path == '/record/stop':
            self.send_response(200); self.send_header('Content-Type', 'text/plain'); self.end_headers()
            with state_lock:
                if not recording:
                    self.wfile.write(b'not recording'); return
                self.cameras['picam0'].stop_encoder(rec_encs['enc0'])
                self.cameras['picam1'].stop_encoder(rec_encs['enc1'])
                recording = False
            self.wfile.write(f'recording saved: {rec_files[0]}, {rec_files[1]}'.encode()); return

        if url.path == '/switch_mode':
            self.send_response(200); self.send_header('Content-Type', 'text/plain'); self.end_headers()
            qs = parse_qs(url.query)
            try:
                w = int(qs.get('w', [0])[0])
                h = int(qs.get('h', [0])[0])
            except (TypeError, ValueError, IndexError):
                self.wfile.write(b'invalid mode params'); return
            with state_lock:
                if recording:
                    self.wfile.write(b'cannot switch mode while recording - stop recording first')
                    return
                if not any((w, h) == (mw, mh) for (mw, mh, _, _) in modes):
                    self.wfile.write(f'unsupported mode {w}x{h}'.encode()); return
                try:
                    for cam_key, out in (('picam0', output0), ('picam1', output1)):
                        cam = self.cameras[cam_key]
                        cam.stop_recording()
                        configure_for_mode(cam, (w, h))
                        start_preview(cam, out)
                    current_raw = (w, h)
                except Exception as e:
                    self.wfile.write(f'switch failed: {e}'.encode()); return
            self.wfile.write(f'switched to raw {w}x{h}'.encode()); return

        if url.path in ('/cam0', '/cam1'):
            output = self.outputs[url.path]
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception:
                pass
            return

        self.send_error(404)

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


print(f"\nDual stream live at http://0.0.0.0:8080/   "
      f"(current sensor mode: {current_raw[0]}x{current_raw[1]} @ {active_fps} fps)")
print("Use the browser UI to switch sensor mode and start/stop recording.")
ThreadedHTTPServer(('0.0.0.0', 8080), StreamingHandler).serve_forever()
