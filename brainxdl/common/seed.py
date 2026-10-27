"""Reproducibility helpers: seed all RNGs from one call."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Call this once at the top of a training script before any data loading
    or model initialisation.

    Source: extracted from legacy_code/2-MM/main.py
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
