"""Device selection utilities for CUDA / MPS / CPU."""

import os

import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU.

    Also sets PYTORCH_ENABLE_MPS_FALLBACK so ops unsupported by MPS
    silently fall back to CPU rather than raising.

    Source: consolidated from legacy_code/1-ST/model.py (get_device)
    and legacy_code/1-ST/main.py (pick_device).
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = True
        return torch.device("cuda")

    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass

    return torch.device("cpu")


def pick_device(prefer_mps: bool = True) -> torch.device:
    """Return CUDA, preferred MPS, or CPU in that order.

    This is the lightweight selector extracted from the ST command-line code.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = True
        return torch.device("cuda")

    if prefer_mps:
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass

    return torch.device("cpu")
