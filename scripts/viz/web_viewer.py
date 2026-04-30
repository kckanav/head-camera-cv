"""Browser-based viewer for the OBJ mesh sequences (Phase 5 visualisation, step 1).

Loads `outputs/<date> - mesh sequence [<tag>]/` (the per-frame OBJs from
scripts/viz/export_obj_sequence.py) and renders them in a NiceGUI +
Three.js scene with a frame slider and play/pause controls.

Architecture cribbed from
https://github.com/Jepson2k/Waldo-Commander (services/urdf_scene/) — the
Python server holds state, NiceGUI pushes Three.js commands to the
browser, the browser does the WebGL rendering.

What's here (step 1):
  - Static scene helpers: table plane (Z=0) + ArUco marker outline at
    origin + small RGB world axes.
  - Per-frame hand mesh swapped in/out as the user scrubs the timeline.
  - Slider, play/pause, single-step buttons.

Out of scope (later commits):
  - The PAROL6 URDF (load via urchin in the same scene).
  - Retargeting visualisation alongside the human hand.
  - Multi-clip browser, source-video overlay.

Notes
-----
* Three.js defaults to Y-up. Our table frame is Z-up. Every mesh is
  pre-rotated by R_x(-90°) at STL-export time so the data displays the
  right way up in the default browser view.
* OBJ-to-STL conversion happens once at startup into a per-process
  tempdir; the per-frame STLs are served from that dir via
  `app.add_static_files('/cache', ...)`.
* Frame swapping uses `ui.scene.stl(url).delete()` + re-add, which is
  enough for scrubbing-speed (<10 fps). Real-time 30 fps playback is
  somewhat sluggish on a laptop browser; flip to vertex-buffer updates
  via raw Three.js if needed.

Run:
  .venv/bin/python scripts/viz/web_viewer.py
  .venv/bin/python scripts/viz/web_viewer.py --dir <path-to-mesh-sequence-dir>
  .venv/bin/python scripts/viz/web_viewer.py --port 8081

Then open http://localhost:8080 (or whatever --port).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from nicegui import app, ui


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "outputs/30th April 2026 - mesh sequence [20260430 manipulation]"
SLOT_COLORS = ["#ffa500", "#1f77b4"]   # orange, blue (matches inspect_anchored.py)
DEFAULT_FPS = 30

# Three.js is Y-up; table frame is Z-up. Rotate Z-up -> Y-up via R_x(-90°):
#   (X, Y, Z)_table  ->  (X, Z, -Y)_viewer
# Applied as a 4x4 to every mesh at export time.
ROT_Z_UP_TO_Y_UP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def cache_obj_sequence(seq_dir: Path, cache_dir: Path) -> tuple[list, str | None]:
    """Convert each frame's OBJ → per-slot STL files in cache_dir, applying
    the Z-up → Y-up rotation. Returns:
      frame_slots: list of (slot0_filename or None, slot1_filename or None),
        one per frame, indexing into cache_dir.
      scene_filename: filename of the static scene helper (table + marker)
        in cache_dir, or None if no scene.obj was found in seq_dir.
    """
    obj_files = sorted(seq_dir.glob("mesh_*.obj"))
    if not obj_files:
        sys.exit(f"no mesh_*.obj files in {seq_dir}")
    print(f"Caching {len(obj_files)} frame meshes → {cache_dir}/")

    frame_slots: list[tuple[str | None, str | None]] = []
    for i, obj in enumerate(obj_files):
        slots: list[str | None] = [None, None]
        try:
            scene_obj = trimesh.load(str(obj), force="scene", skip_materials=True)
        except Exception:
            frame_slots.append((None, None))
            continue
        for name, geom in scene_obj.geometry.items():
            for s in (0, 1):
                if f"slot{s}" in name:
                    geom.apply_transform(ROT_Z_UP_TO_Y_UP)
                    fname = f"frame_{i:04d}_slot{s}.stl"
                    geom.export(str(cache_dir / fname))
                    slots[s] = fname
                    break
        frame_slots.append((slots[0], slots[1]))
        if (i + 1) % 100 == 0 or i == len(obj_files) - 1:
            print(f"  {i+1}/{len(obj_files)}")

    scene_filename = None
    scene_path = seq_dir / "scene.obj"
    if scene_path.exists():
        sd = trimesh.load(str(scene_path), force="scene", skip_materials=True)
        combined = trimesh.util.concatenate(list(sd.geometry.values()))
        combined.apply_transform(ROT_Z_UP_TO_Y_UP)
        scene_filename = "scene.stl"
        combined.export(str(cache_dir / scene_filename))

    return frame_slots, scene_filename


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help="path to a mesh sequence directory (mesh_*.obj + optional scene.obj)")
    ap.add_argument("--port", type=int, default=8080, help="server port (default 8080)")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS,
                    help="playback frame rate (default 30; lower if browser is sluggish)")
    args = ap.parse_args()

    seq_dir = Path(args.dir).resolve() if Path(args.dir).is_absolute() \
        else (PROJECT_ROOT / args.dir).resolve()
    if not seq_dir.is_dir():
        sys.exit(f"missing directory: {seq_dir}")

    # Pre-convert all frames into a tempdir of STLs that NiceGUI can serve.
    cache_dir = Path(tempfile.mkdtemp(prefix="hand_viewer_"))
    frame_slots, scene_filename = cache_obj_sequence(seq_dir, cache_dir)
    n_frames = len(frame_slots)

    # Expose the cache as a static URL prefix.
    app.add_static_files("/cache", str(cache_dir))

    # ---- UI ----
    state = {"frame": 0, "playing": False}

    ui.dark_mode().enable()

    with ui.row().classes("w-full no-wrap").style("height: 95vh"):
        # Left: 3D scene
        with ui.column().classes("flex-grow h-full"):
            scene = ui.scene().classes("w-full h-full")
            # Camera: looking down at the table plane from above-and-side.
            scene.move_camera(
                x=0.6, y=0.5, z=0.6,        # in viewer (Y-up) coords
                look_at_x=0.15, look_at_y=0.05, look_at_z=0.0,
            )

            # Static scene helper (table plane + marker outline).
            if scene_filename is not None:
                scene.stl(f"/cache/{scene_filename}").material("#666666", opacity=0.5)

            # Tiny RGB world axes at origin (in viewer coords; XYZ = R G B).
            ax = 0.06
            scene.box(width=ax, height=0.003, depth=0.003).move(ax / 2, 0, 0).material("#ff3333")
            scene.box(width=0.003, height=ax, depth=0.003).move(0, ax / 2, 0).material("#33ff33")
            scene.box(width=0.003, height=0.003, depth=ax).move(0, 0, ax / 2).material("#3366ff")

            # Two slot meshes — handles to the currently-displayed STLs.
            slot_handles: list = [None, None]

            def render_frame(idx: int) -> None:
                slot0_name, slot1_name = frame_slots[idx]
                for s, name in enumerate((slot0_name, slot1_name)):
                    if slot_handles[s] is not None:
                        slot_handles[s].delete()
                        slot_handles[s] = None
                    if name is not None:
                        slot_handles[s] = scene.stl(f"/cache/{name}").material(SLOT_COLORS[s])

            render_frame(0)

        # Right: controls
        with ui.column().classes("w-80 p-4 gap-3"):
            ui.label("Hand mesh viewer").classes("text-2xl font-bold")
            ui.label(f"{n_frames} frames @ {args.fps} fps  ({n_frames / args.fps:.1f} s)") \
                .classes("text-sm text-gray-400")

            frame_label = ui.label(f"Frame 0 / {n_frames - 1}  (0.00 s)").classes("font-mono")

            def _on_slider_change(e):
                idx = int(e.value)
                state["frame"] = idx
                t = idx / args.fps
                frame_label.text = f"Frame {idx} / {n_frames - 1}  ({t:5.2f} s)"
                render_frame(idx)

            slider = ui.slider(
                min=0, max=n_frames - 1, value=0, step=1,
                on_change=_on_slider_change,
            ).props("label-always")

            with ui.row().classes("gap-2 w-full"):
                play_btn = ui.button("▶ Play")

                def toggle_play():
                    state["playing"] = not state["playing"]
                    play_btn.text = "⏸ Pause" if state["playing"] else "▶ Play"

                def step_back():
                    new = max(0, state["frame"] - 1)
                    slider.set_value(new)

                def step_forward():
                    new = min(n_frames - 1, state["frame"] + 1)
                    slider.set_value(new)

                play_btn.on("click", toggle_play)
                ui.button("⏮ −1", on_click=step_back)
                ui.button("+1 ⏭", on_click=step_forward)

            ui.separator()
            ui.label("Coordinate frame").classes("font-bold mt-2")
            ui.html(
                "• Origin: ArUco marker centre on the table.<br>"
                "• Marker outline + table plane at viewer Y=0.<br>"
                "• Table-frame +Z is up → mapped to viewer +Y.<br>"
                "• Table-frame +X / +Y stay in the table plane.<br>"
                "• Units: metres.",
                sanitize=False,
            ).classes("text-xs text-gray-300 leading-relaxed")

            ui.separator()
            ui.html(
                f"<span style='color:{SLOT_COLORS[0]}'>■</span> slot 0 "
                f"&nbsp;&nbsp;<span style='color:{SLOT_COLORS[1]}'>■</span> slot 1",
                sanitize=False,
            ).classes("text-sm")

            ui.separator()
            ui.label(f"Source: {seq_dir.name}").classes("text-xs text-gray-400 mt-2 break-all")

    # Auto-play timer.
    def _step_auto():
        if state["playing"]:
            new = (state["frame"] + 1) % n_frames
            slider.set_value(new)

    ui.timer(1.0 / args.fps, _step_auto)

    print()
    print(f"Open http://localhost:{args.port} in your browser.")
    print(f"(Press Ctrl-C to stop the server.)")
    ui.run(port=args.port, show=False, title="Hand Mesh Viewer", reload=False)


# NiceGUI re-imports the script in worker processes; the standard
# `__main__` guard misses those, so include `__mp_main__`.
if __name__ in {"__main__", "__mp_main__"}:
    main()
