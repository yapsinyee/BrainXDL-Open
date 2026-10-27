"""Load frozen ST-BrainXDL checkpoints for evaluation and explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from brainxdl.common.checkpointing import load_checkpoint
from brainxdl.st.model import STBrainXDL


def load_model(
    checkpoint: Path,
    device: torch.device,
    sparse_on_cpu_override: bool | None,
) -> tuple[STBrainXDL, dict[str, Any], dict[str, float]]:
    """Load a frozen ST model using the checkpoint's recorded training settings."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    payload = load_checkpoint(str(checkpoint), device=device)
    setting = payload["training_setting"]
    setting["device"] = str(device)
    if sparse_on_cpu_override is not None:
        setting["sparse_on_cpu"] = bool(sparse_on_cpu_override)
    sparse_on_cpu = bool(setting.get("sparse_on_cpu", True))

    model = STBrainXDL(setting)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device)
    model.apply_device_split(device, sparse_on_cpu=sparse_on_cpu)
    model.eval()

    weights = {
        "kl_weight": float(setting.get("kl_weight", 1.0)),
        "bce_weight": float(setting.get("bce_weight", 1.0)),
        "mce_weight": float(setting.get("mce_weight", 1.0)),
    }
    return model, setting, weights
