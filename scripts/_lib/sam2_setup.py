"""Path + checkpoint helpers for any script that imports `sam2.*`.

SAM2 is properly pip-installable (`pip install -e .` from the cloned
repo), so unlike `wilor_setup.py` this module needs no sys.path or
torch.load monkey-patching — `from sam2.build_sam import
build_sam2_video_predictor` just works once the package is installed.

What this module does:
  - Resolve the SAM2 checkpoint path (gives a copy-pasteable wget command
    in the error message if it's missing).
  - Pin the canonical model variant + hydra config name in one place so
    every script defaults to the same one (sam2.1 hiera large).

Two venvs use SAM2:
  - `.venv-sam2` (Linux / CUDA) — the heavy inference path. Built from
    PyTorch+cu121 so SAM2's optional CUDA extensions compile.
  - `.venv-hamer` (macOS / MPS) — sufficient for click-UI smoke testing
    and short clips. SAM2's CUDA extensions skip cleanly when there's no
    nvcc; the Python fallbacks are slower but functional.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM2_DIR = PROJECT_ROOT / "sam2"   # cloned facebookresearch/sam2 repo (gitignored)
SAM2_MODELS_DIR = PROJECT_ROOT / "models" / "sam2"

# SAM2.1 large — best quality (~224 M params, ~900 MB). The hiera_l yaml
# is shipped inside the sam2 package as `configs/sam2.1/sam2.1_hiera_l.yaml`
# and looked up via Hydra; we pass it as a string, not a filesystem path.
SAM2_CKPT = SAM2_MODELS_DIR / "sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def assert_checkpoint(path: Path = SAM2_CKPT) -> Path:
    """Return `path` if the file exists; otherwise raise with the exact
    wget command to download it. Saves a half-page of stack trace when
    the user just hasn't fetched the weights yet."""
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"SAM2 checkpoint missing: {path}\n"
        f"Download with:\n"
        f"  mkdir -p {path.parent}\n"
        f"  wget -P {path.parent} "
        f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{path.name}"
    )
