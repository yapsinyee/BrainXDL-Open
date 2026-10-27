"""Checkpoint save / load helpers shared across ST and MM modules.

Provides:
- Pickle-based object I/O used by the ST preprocessing pipeline.
- PyTorch checkpoint save/load with RNG state preservation.
- CPU-safe state dict stripping for device-portable checkpoints.

Sources: consolidated from
  legacy_code/1-ST/train.py  (save_obj, load_obj, check_path, loadRNG)
  legacy_code/2-MM/main.py   (to_cpu_state_dict)
"""

import os
import pickle
from typing import Any, Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Pickle-based object I/O
# ---------------------------------------------------------------------------

def save_obj(obj: Any, path: str) -> None:
    """Pickle *obj* to *path*, appending '.pkl' if absent."""
    if not path.endswith(".pkl"):
        path = path + ".pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_obj(path: str) -> Any:
    """Load and return a pickled object from *path*."""
    if not path.endswith(".pkl"):
        path = path + ".pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def check_path(path: str) -> bool:
    """Return True if *path* (with optional '.pkl' extension) exists."""
    if not path.endswith(".pkl"):
        path = path + ".pkl"
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# PyTorch checkpoint helpers
# ---------------------------------------------------------------------------

def to_cpu_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Detach and move every tensor in *state_dict* to CPU.

    Checkpoints saved via this helper can be loaded on any device without
    specifying map_location.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    return {key: tensor.detach().cpu() for key, tensor in state_dict.items()}


def save_checkpoint(path: str, payload: dict) -> None:
    """Save *payload* to *path* using torch.save.

    The caller is responsible for building the payload dict.  Common keys:
      model_state_dict, optimizer_state_dicts, scheduler_state_dicts,
      epoch, rng_state, train_losses, val_losses, test_losses.

    The parent directory is created automatically if missing.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> dict:
    """Load a torch checkpoint from *path*, mapping tensors to *device*.

    Args:
        path:   Filesystem path to the .pth / .pt checkpoint.
        device: Target device.  Defaults to CPU so the checkpoint is
                portable regardless of where it was saved.

    Returns:
        The checkpoint dict as saved by torch.save.
    """
    map_location = device if device is not None else torch.device("cpu")
    return torch.load(path, map_location=map_location, weights_only=False)


def load_rng_state(path: str) -> None:
    """Restore the PyTorch CPU RNG state from a checkpoint at *path*.

    No-ops silently if the file does not exist or contains no 'rng_state'
    key, so callers do not need to guard against missing checkpoints during
    the first training run.

    Source: extracted from legacy_code/1-ST/train.py (loadRNG).
    """
    if not os.path.exists(path):
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
