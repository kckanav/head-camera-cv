"""Shared one-time setup for any script that imports the WiLoR package.

Importing this module performs three side effects, in order, BEFORE the
caller imports `wilor.*`:

  1. Stub the three pyrender-using submodules of `wilor.utils` so model
     construction works on macOS without OSMesa (we run inference-only —
     nothing actually invokes these renderers).

  2. Monkey-patch `torch.load` to default `weights_only=False`. PyTorch 2.6
     flipped the default; the YOLO ckpt has pickled class refs and won't
     load under the new default. We trust both checkpoint sources (the
     upstream HuggingFace mirror).

  3. Insert `<repo>/wilor` on sys.path and chdir into it. WiLoR's config
     has relative `./pretrained_models/...` paths; chdir is the simplest
     way to make them resolve. Putting only WILOR_DIR on sys.path (not
     PROJECT_ROOT) avoids a namespace-package collision: PROJECT_ROOT/
     wilor/models/ holds the user-uploaded MANO pkls, while PROJECT_ROOT/
     wilor/wilor/models/ is the real package's submodule. With only
     WILOR_DIR on path, `import wilor.models` resolves unambiguously.

Why side-effects-on-import: every wilor script needs all three, in this
order, before any `from wilor.* import *` statement. Centralising them
means each script has a single `import wilor_setup` line instead of 60+
duplicated bootstrap lines.

If any of these workarounds stop being needed (upstream pinning a newer
ultralytics, PyTorch reverting `weights_only`, EGL becoming available on
macOS), delete the corresponding step here AND the matching note in
CLAUDE.md.
"""

import os
import sys
import types
from pathlib import Path


# --- 1. pyrender stubs --------------------------------------------------------
class _NoopRenderer:
    def __init__(self, *a, **kw):
        pass

    def __getattr__(self, name):
        raise RuntimeError(f"renderer stubbed; nothing should call {name!r}")


def _cam_crop_to_full(*a, **kw):
    raise RuntimeError("cam_crop_to_full stubbed")


for _mod_name, _attrs in [
    ("wilor.utils.renderer",          {"Renderer": _NoopRenderer, "cam_crop_to_full": _cam_crop_to_full}),
    ("wilor.utils.mesh_renderer",     {"MeshRenderer": _NoopRenderer}),
    ("wilor.utils.skeleton_renderer", {"SkeletonRenderer": _NoopRenderer}),
]:
    _stub = types.ModuleType(_mod_name)
    for _k, _v in _attrs.items():
        setattr(_stub, _k, _v)
    sys.modules[_mod_name] = _stub


# --- 2. torch.load weights_only default ---------------------------------------
import torch  # noqa: E402

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


# --- 3. WiLoR sys.path + chdir ------------------------------------------------
# Every script that imports this module lives at scripts/<subdir>/<file>.py.
# This module lives at scripts/_lib/wilor_setup.py, so PROJECT_ROOT is parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WILOR_DIR = PROJECT_ROOT / "wilor"

if not WILOR_DIR.is_dir():
    sys.exit(f"missing {WILOR_DIR} - run Phase 0 (env + WiLoR clone) first; see PLAN.md")

sys.path.insert(0, str(WILOR_DIR))
os.chdir(WILOR_DIR)


# --- helper -------------------------------------------------------------------
def get_mano_faces(model):
    """Pull the (1538, 3) MANO face index array out of the WiLoR model.

    Falls back to loading directly from MANO_RIGHT.pkl if the model didn't
    expose its mano layer under any of the expected attributes.
    """
    import numpy as np
    for attr in ("mano", "mano_layer", "smpl", "body_model"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "faces"):
            return np.asarray(m.faces, dtype=np.int32)
    import pickle
    with open(WILOR_DIR / "mano_data" / "MANO_RIGHT.pkl", "rb") as f:
        mano_right = pickle.load(f, encoding="latin1")
    return np.asarray(mano_right["f"], dtype=np.int32)
