"""Validation helpers for participant-grouped ST experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FOLD_DENSITY = {1: "p0p1", 2: "p0p2", 3: "p0p2", 4: "p0p1", 5: "p0p2"}
FOLDS = (1, 2, 3, 4, 5)
SPLITS = ("train", "val", "test")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from *path*."""
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(metrics_path: Path, metrics: dict[str, Any]) -> Path:
    """Resolve a fold checkpoint from recorded metadata or its checkpoint directory."""
    recorded = metrics.get("checkpoint")
    if recorded:
        recorded_path = Path(str(recorded)).expanduser()
        if recorded_path.is_file():
            return recorded_path.resolve()
    candidates = sorted((metrics_path.parent / "checkpoints").glob("*.pth"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not resolve one checkpoint for {metrics_path}; "
            f"found {len(candidates)} candidates."
        )
    return candidates[0].resolve()


def fold_inputs(args: Any, frozen: dict[str, Any], fold: int) -> dict[str, Any]:
    """Validate and resolve the frozen inputs for one ST outer fold."""
    selected = frozen.get("folds", {}).get(str(fold), frozen.get("folds", {}).get(fold))
    if not isinstance(selected, dict):
        raise ValueError(f"Fold {fold}: missing frozen selection.")
    if selected.get("outer_test_used_for_selection") is not False:
        raise ValueError(f"Fold {fold}: outer-test exclusion is not explicit.")

    metrics_path = (
        args.d1_root
        / "final"
        / f"seed_{args.seed}"
        / f"outer_{fold:02d}"
        / "metrics.json"
    )
    metrics = read_json(metrics_path)
    if int(metrics.get("seed", -1)) != args.seed or int(metrics.get("outer_fold", -1)) != fold:
        raise ValueError(f"Fold {fold}: checkpoint metrics have mismatched seed/fold metadata.")
    for field in ("variant", "encoder_cls_weight", "readout_mode", "stable_fc_skip"):
        if field in selected and metrics.get(field) != selected[field]:
            raise ValueError(f"Fold {fold}: {field} differs from the frozen selection.")
    if "best_epoch" in selected and int(metrics.get("fixed_refit_epochs", -1)) != int(
        selected["best_epoch"]
    ):
        raise ValueError(f"Fold {fold}: refit epoch budget differs from the frozen selection.")
    if not 0.0 <= float(metrics.get("validation_threshold", -1.0)) <= 1.0:
        raise ValueError(f"Fold {fold}: invalid validation-selected threshold.")
    checkpoint = resolve_checkpoint(metrics_path, metrics)

    density = FOLD_DENSITY[fold]
    fold_file = args.data_root / f"w40_s10_{density}" / f"graphs_outer{fold}_inner1.pkl"
    if not fold_file.is_file():
        raise FileNotFoundError(fold_file)
    return {
        "fold": fold,
        "selected": selected,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "checkpoint": checkpoint,
        "fold_file": fold_file,
        "density": density,
    }


def split_metadata(
    metadata: dict[str, Any], fold: int
) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    """Validate complete, observation-disjoint, participant-disjoint split metadata."""
    sample_ids = [str(value) for value in metadata.get("sample_ids", [])]
    participant_groups = [str(value) for value in metadata.get("participant_groups", [])]
    raw_indices = metadata.get("split_indices")
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Fold {fold}: sample_ids are missing or not unique.")
    if len(participant_groups) != len(sample_ids):
        raise ValueError(f"Fold {fold}: participant_groups do not match sample_ids.")
    if not isinstance(raw_indices, dict):
        raise ValueError(f"Fold {fold}: split_indices metadata is missing.")

    indices = {name: np.asarray(raw_indices[name], dtype=np.int64) for name in SPLITS}
    seen: set[int] = set()
    split_groups: dict[str, set[str]] = {}
    for name in SPLITS:
        values = indices[name]
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ValueError(f"Fold {fold} {name}: indices must be one-dimensional and unique.")
        if np.any(values < 0) or np.any(values >= len(sample_ids)):
            raise ValueError(f"Fold {fold} {name}: split index outside sample range.")
        overlap = seen.intersection(map(int, values))
        if overlap:
            raise ValueError(f"Fold {fold}: overlapping split indices: {sorted(overlap)[:10]}")
        seen.update(map(int, values))
        split_groups[name] = {participant_groups[int(index)] for index in values}

    if seen != set(range(len(sample_ids))):
        raise ValueError(f"Fold {fold}: train/val/test do not cover every observation exactly once.")
    if (
        split_groups["train"] & split_groups["val"]
        or split_groups["train"] & split_groups["test"]
        or split_groups["val"] & split_groups["test"]
    ):
        raise ValueError(f"Fold {fold}: participant leakage detected across splits.")
    return sample_ids, participant_groups, indices


def condition_name(label_mapping: dict[str, Any], label: int) -> str:
    """Return the concise condition name stored for a numeric label."""
    value = str(label_mapping.get(str(int(label)), label_mapping.get(int(label), label)))
    return value.split("(", maxsplit=1)[0].strip()
