"""Participant-grouped static PsiConnect MM benchmark utilities.

This module intentionally does not reuse the historical observation-level ISBI
split.  The ST PsiConnect manifest is the single split authority for every model.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from brainxdl.common.connectivity import knn_thresholding
from brainxdl.common.metrics import choose_binary_threshold, compute_classification_metrics
from brainxdl.common.seed import seed_everything
from brainxdl.mm.data.dataset import prepare_data
from brainxdl.mm.models.baselines import FmriGCN, LateFusionMLP
from brainxdl.mm.models.brainmgf import MMBrainXDL


EXPECTED_COHORT = {
    "n_observations": 120,
    "n_participants": 65,
    "n_control": 57,
    "n_psilocybin": 63,
    "n_paired": 55,
    "n_psilocybin_only": 8,
    "n_control_only": 2,
}
SPLITS = ("train", "val", "test")
METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "f1", "precision", "sensitivity", "specificity", "log_loss")
CLASSICAL_MODELS = ("logistic", "svm_linear", "svm_rbf")
NEURAL_MODELS = (
    "fcnn",
    "gcn",
    "gcn_eeg",
    "gat",
    "eeg_only",
    "fmri_only",
    "late_fusion",
    "concat_fusion",
    "equal_fusion",
    "adaptive_fusion",
)
AVAILABLE_MODELS = CLASSICAL_MODELS + NEURAL_MODELS
UNAVAILABLE_MODELS: dict[str, str] = {}

MODEL_INFO = {
    "logistic": ("fMRI + EEG", "Classical connectivity"),
    "svm_linear": ("fMRI + EEG", "Classical connectivity"),
    "svm_rbf": ("fMRI + EEG", "Classical connectivity"),
    "fcnn": ("fMRI + EEG", "Non-graph neural"),
    "gcn": ("fMRI", "Generic graph"),
    "gcn_eeg": ("EEG", "Generic graph"),
    "gat": ("fMRI", "Generic graph"),
    "eeg_only": ("EEG", "MM-BrainXDL ablation"),
    "fmri_only": ("fMRI", "MM-BrainXDL ablation"),
    "late_fusion": ("fMRI + EEG", "Prediction-level fusion"),
    "concat_fusion": ("fMRI + EEG", "Embedding fusion"),
    "equal_fusion": ("fMRI + EEG", "Embedding fusion"),
    "adaptive_fusion": ("fMRI + EEG", "Adaptive embedding fusion"),
}


@dataclass(frozen=True)
class BenchmarkData:
    corr_fmri: np.ndarray
    pcorr_fmri: np.ndarray
    corr_eeg: np.ndarray
    pcorr_eeg: np.ndarray
    pseudo_fmri: np.ndarray
    pseudo_eeg: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    tasks: np.ndarray
    sample_ids: np.ndarray
    sessions: np.ndarray

    def __len__(self) -> int:
        return int(len(self.labels))


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=_json_value, allow_nan=False) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    fields = list(fields)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("folds"), dict):
        raise ValueError(f"Invalid PsiConnect fold manifest: {path}")
    return manifest


def manifest_master_order(manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(manifest.get("n_samples", 0))
    if n < 1:
        raise ValueError("Manifest has no samples")
    fold = manifest["folds"].get("outer1_inner1")
    if not isinstance(fold, dict):
        raise ValueError("Manifest lacks outer1_inner1")
    sample_ids: list[str | None] = [None] * n
    groups: list[str | None] = [None] * n
    for split in SPLITS:
        section = fold.get(split, {})
        indices = section.get("indices", [])
        ids = section.get("sample_ids", [])
        participant_groups = section.get("participant_groups", [])
        if not (len(indices) == len(ids) == len(participant_groups)):
            raise ValueError(f"Manifest outer1_inner1/{split} fields differ in length")
        for index, sample_id, group in zip(indices, ids, participant_groups, strict=True):
            index = int(index)
            if index < 0 or index >= n or sample_ids[index] is not None:
                raise ValueError(f"Invalid or duplicate manifest index {index}")
            sample_ids[index] = str(sample_id)
            groups[index] = str(group)
    if any(value is None for value in sample_ids + groups):
        raise ValueError("Manifest first fold does not cover every observation")
    sessions = np.asarray([_session_from_sample_id(str(value)) for value in sample_ids], dtype=str)
    return np.asarray(sample_ids, dtype=str), np.asarray(groups, dtype=str), sessions


def _session_from_sample_id(sample_id: str) -> str:
    marker = "_ses-"
    if marker not in sample_id:
        raise ValueError(f"Cannot parse session from sample ID: {sample_id}")
    return "ses-" + sample_id.split(marker, 1)[1].split("_", 1)[0]


def _label_from_session(session: str) -> int:
    if session in {"ses-01", "ses-1"}:
        return 1
    if session in {"ses-02", "ses-2"}:
        return 0
    raise ValueError(f"Unexpected PsiConnect session: {session}")


def validate_manifest(
    manifest: dict[str, Any], *, enforce_expected_cohort: bool = True
) -> dict[str, Any]:
    sample_ids, groups, sessions = manifest_master_order(manifest)
    labels = np.asarray([_label_from_session(value) for value in sessions], dtype=int)
    n = len(labels)
    seen_test: list[int] = []
    fold_audit: dict[str, Any] = {}
    paired_groups = {
        group for group in np.unique(groups) if set(sessions[groups == group]) == {"ses-01", "ses-02"}
    }
    for fold_number in range(1, 6):
        key = f"outer{fold_number}_inner1"
        fold = manifest["folds"].get(key)
        if not isinstance(fold, dict):
            raise ValueError(f"Manifest lacks {key}")
        indices = {split: np.asarray(fold[split]["indices"], dtype=int) for split in SPLITS}
        for split in SPLITS:
            if indices[split].ndim != 1 or len(np.unique(indices[split])) != len(indices[split]):
                raise ValueError(f"{key}/{split} indices are not unique one-dimensional values")
            if np.any(indices[split] < 0) or np.any(indices[split] >= n):
                raise ValueError(f"{key}/{split} index outside sample range")
            expected_ids = sample_ids[indices[split]].astype(str)
            expected_groups = groups[indices[split]].astype(str)
            if not np.array_equal(expected_ids, np.asarray(fold[split]["sample_ids"]).astype(str)):
                raise ValueError(f"{key}/{split} sample IDs are not index-aligned")
            if not np.array_equal(expected_groups, np.asarray(fold[split]["participant_groups"]).astype(str)):
                raise ValueError(f"{key}/{split} participant IDs are not index-aligned")
        if set().union(*(set(values.tolist()) for values in indices.values())) != set(range(n)):
            raise ValueError(f"{key} does not cover every observation")
        if any(
            set(indices[left].tolist()) & set(indices[right].tolist())
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError(f"{key} has observation overlap")
        split_groups = {split: set(groups[indices[split]].tolist()) for split in SPLITS}
        overlaps = {
            "train_val": sorted(split_groups["train"] & split_groups["val"]),
            "train_test": sorted(split_groups["train"] & split_groups["test"]),
            "val_test": sorted(split_groups["val"] & split_groups["test"]),
        }
        if any(overlaps.values()):
            raise ValueError(f"Participant leakage in {key}: {overlaps}")
        for group in paired_groups:
            memberships = [split for split in SPLITS if group in split_groups[split]]
            if len(memberships) != 1:
                raise ValueError(f"Paired participant {group} is split in {key}")
        seen_test.extend(indices["test"].tolist())
        fold_audit[key] = {
            "samples": {split: int(len(indices[split])) for split in SPLITS},
            "participants": {split: int(len(split_groups[split])) for split in SPLITS},
            "participant_intersections": overlaps,
        }
    if sorted(seen_test) != list(range(n)):
        raise ValueError("Every observation must occur in exactly one outer test fold")
    patterns = {
        group: tuple(sorted(labels[groups == group].tolist())) for group in np.unique(groups)
    }
    cohort = {
        "n_observations": n,
        "n_participants": int(len(np.unique(groups))),
        "n_control": int(np.sum(labels == 0)),
        "n_psilocybin": int(np.sum(labels == 1)),
        "n_paired": sum(value == (0, 1) for value in patterns.values()),
        "n_psilocybin_only": sum(value == (1,) for value in patterns.values()),
        "n_control_only": sum(value == (0,) for value in patterns.values()),
    }
    if enforce_expected_cohort and cohort != EXPECTED_COHORT:
        raise ValueError(f"Cohort differs from the verified 120-observation cohort: {cohort}")
    return {"cohort": cohort, "folds": fold_audit, "all_checks_passed": True}


def fold_indices(manifest: dict[str, Any], fold: int) -> dict[str, np.ndarray]:
    key = f"outer{fold}_inner1"
    if key not in manifest["folds"]:
        raise ValueError(f"Missing fold {fold} in manifest")
    return {
        split: np.asarray(manifest["folds"][key][split]["indices"], dtype=int)
        for split in SPLITS
    }


def _prepare_data_args(fmri_path: Path, eeg_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="PSICONNECT",
        fmri_path=str(fmri_path),
        eeg_path=str(eeg_path),
        task="REST",
        tasks=None,
        graph_mode="static",
        window_len=50,
        window_stride=25,
    )


def prepare_benchmark_cache(
    *, fmri_path: Path, eeg_path: Path, manifest_path: Path, cache_path: Path, output_root: Path
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    audit = validate_manifest(manifest)
    sample_ids, manifest_groups, sessions = manifest_master_order(manifest)
    prepared = prepare_data(_prepare_data_args(fmri_path, eeg_path))
    if len(prepared) != 9:
        raise RuntimeError("MM loader did not return participant groups and task labels")
    corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels, pseudo_fmri, pseudo_eeg, groups, tasks = prepared
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups).astype(str)
    expected_labels = np.asarray([_label_from_session(value) for value in sessions], dtype=int)
    if len(labels) != len(sample_ids):
        raise ValueError(f"MM/ST observation count mismatch: {len(labels)} != {len(sample_ids)}")
    if not np.array_equal(groups, manifest_groups):
        raise ValueError("MM row participant order does not match the shared ST manifest")
    if not np.array_equal(labels, expected_labels):
        raise ValueError("MM labels do not match manifest session labels")
    reconstructed_ids = np.asarray(
        [f"sub-{group}_{session}_task-rest" for group, session in zip(groups, sessions, strict=True)],
        dtype=str,
    )
    if not np.array_equal(reconstructed_ids, sample_ids):
        raise ValueError("MM participant/session rows do not reconstruct the manifest sample IDs")
    if set(np.char.lower(np.asarray(tasks).astype(str)).tolist()) != {"rest"}:
        raise ValueError("Static participant-grouped benchmark cache must contain REST observations only")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("Sample IDs are not unique")
    for array in (corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, pseudo_fmri, pseudo_eeg):
        if len(array) != len(labels) or not np.isfinite(array).all():
            raise ValueError("MM connectivity arrays are not finite and sample-aligned")
    payload = {
        "corr_fmri": corr_fmri,
        "pcorr_fmri": pcorr_fmri,
        "corr_eeg": corr_eeg,
        "pcorr_eeg": pcorr_eeg,
        "labels": labels,
        "pseudo_fmri": pseudo_fmri,
        "pseudo_eeg": pseudo_eeg,
        "groups": groups,
        "tasks": np.asarray(tasks).astype(str),
        "sample_ids": sample_ids,
        "sessions": sessions,
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_write_pickle(cache_path, payload)
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source_text = manifest_path.read_text(encoding="utf-8")
    validated_manifest = manifest_dir / "participant_grouped_folds.json"
    if validated_manifest.exists() and validated_manifest.read_text(encoding="utf-8") != source_text:
        raise RuntimeError(f"Refusing to replace a different validated manifest: {validated_manifest}")
    atomic_write_text(validated_manifest, source_text)
    audit.update(
        {
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": file_sha256(manifest_path),
            "cache": str(cache_path.resolve()),
            "cache_sha256": file_sha256(cache_path),
            "fmri_path": str(fmri_path.resolve()),
            "eeg_path": str(eeg_path.resolve()),
        }
    )
    atomic_write_json(manifest_dir / "dataset_audit.json", audit)
    atomic_write_text(manifest_dir / "preprocessing_audit.md", preprocessing_audit_markdown(audit))
    return audit


def preprocessing_audit_markdown(audit: dict[str, Any]) -> str:
    cohort = audit["cohort"]
    return f"""# Participant-grouped static MM preprocessing audit

This is the **participant-grouped thesis re-evaluation**, not a replacement for
the historical ISBI results.

## Cohort

- Observations: {cohort['n_observations']}
- Participants: {cohort['n_participants']}
- Psilocybin/control: {cohort['n_psilocybin']}/{cohort['n_control']}
- Complete participant pairs: {cohort['n_paired']}

## Leakage classification

- Per-observation: Pearson correlation; ridge-regularized partial correlation;
  identity pseudo-features. These are computed independently for each recording.
- Training-fitted: `StandardScaler` for vector models, model weights, class
  weighting, validation-balanced-accuracy decision threshold.
- Fixed constants: five saved participant folds, `k=40` fMRI/EEG graph sparsity,
  ridge `1e-3` for partial correlation, and model architecture widths.
- Not used: cohort-fitted PCA, cohort-fitted imputation, test-fitted scaling,
  population-derived graph thresholds, or test-label threshold selection.

## Graph construction

The static MM representation is preserved. Pearson matrices form node features
(plus a learned transform of the identity pseudo-feature for MM encoders). Partial
correlation uses a per-recording covariance, diagonal ridge `1e-3`, and pseudoinverse.
`knn_thresholding` retains the top 40 raw signed values per node, preserves their
weights/signs, retains diagonal entries when selected, and does not re-symmetrize
the directed row-wise top-k result. No population statistics are used.

All manifest leakage and alignment checks passed: `{audit['all_checks_passed']}`.
"""


def load_benchmark_cache(cache_path: Path, manifest_path: Path) -> BenchmarkData:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Prepared MM cache not found: {cache_path}. Run --prepare-only first.")
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    manifest = read_manifest(manifest_path)
    validate_manifest(manifest)
    sample_ids, groups, sessions = manifest_master_order(manifest)
    if payload.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Prepared cache was created from a different fold manifest")
    if not np.array_equal(np.asarray(payload["sample_ids"]).astype(str), sample_ids):
        raise ValueError("Cache sample order differs from manifest")
    if not np.array_equal(np.asarray(payload["groups"]).astype(str), groups):
        raise ValueError("Cache participant order differs from manifest")
    data = BenchmarkData(
        corr_fmri=np.asarray(payload["corr_fmri"], dtype=np.float32),
        pcorr_fmri=np.asarray(payload["pcorr_fmri"], dtype=np.float32),
        corr_eeg=np.asarray(payload["corr_eeg"], dtype=np.float32),
        pcorr_eeg=np.asarray(payload["pcorr_eeg"], dtype=np.float32),
        pseudo_fmri=np.asarray(payload["pseudo_fmri"], dtype=np.float32),
        pseudo_eeg=np.asarray(payload["pseudo_eeg"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=int),
        groups=groups,
        tasks=np.asarray(payload["tasks"]).astype(str),
        sample_ids=sample_ids,
        sessions=sessions,
    )
    if set(np.unique(data.labels)) != {0, 1}:
        raise ValueError("Benchmark requires binary labels 0 and 1")
    validate_benchmark_data(data, manifest)
    return data


def validate_benchmark_data(data: BenchmarkData, manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if cached connectivity or cohort metadata drift from the manifest."""
    audit = validate_manifest(manifest)
    sample_ids, groups, sessions = manifest_master_order(manifest)
    expected_labels = np.asarray([_label_from_session(value) for value in sessions], dtype=int)
    if not np.array_equal(data.sample_ids, sample_ids):
        raise ValueError("Benchmark sample/session ordering differs from manifest")
    if not np.array_equal(data.groups, groups):
        raise ValueError("Benchmark participant ordering differs from manifest")
    if not np.array_equal(data.sessions, sessions):
        raise ValueError("Benchmark session ordering differs from manifest")
    if not np.array_equal(data.labels, expected_labels):
        raise ValueError("Benchmark labels differ from manifest sessions")
    if len(np.unique(data.sample_ids)) != len(data):
        raise ValueError("Benchmark observation IDs are not unique")

    dimensions: dict[str, list[int]] = {}
    for name in ("corr_fmri", "pcorr_fmri", "corr_eeg", "pcorr_eeg", "pseudo_fmri", "pseudo_eeg"):
        array = np.asarray(getattr(data, name))
        dimensions[name] = list(array.shape)
        if array.ndim != 3 or array.shape[0] != len(data) or array.shape[1] != array.shape[2]:
            raise ValueError(f"{name} must be a sample-aligned stack of square matrices; got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    for modality in ("fmri", "eeg"):
        corr = getattr(data, f"corr_{modality}")
        pcorr = getattr(data, f"pcorr_{modality}")
        pseudo = getattr(data, f"pseudo_{modality}")
        if not (corr.shape == pcorr.shape == pseudo.shape):
            raise ValueError(f"{modality} Pearson, partial-correlation, and pseudo tensors differ in shape")
        for name, array in (("Pearson", corr), ("partial correlation", pcorr), ("pseudo", pseudo)):
            if not np.allclose(array, np.swapaxes(array, 1, 2), rtol=1e-5, atol=1e-6):
                raise ValueError(f"{modality} {name} matrices are not symmetric")
    for fold in range(1, 6):
        split = fold_indices(manifest, fold)
        for split_name, indices in split.items():
            section = manifest["folds"][f"outer{fold}_inner1"][split_name]
            if not np.array_equal(data.sample_ids[indices], np.asarray(section["sample_ids"], dtype=str)):
                raise ValueError(f"Fold {fold} {split_name} observation membership differs from manifest")
            if not np.array_equal(data.groups[indices], np.asarray(section["participant_groups"], dtype=str)):
                raise ValueError(f"Fold {fold} {split_name} participant membership differs from manifest")
    return {"cohort": audit["cohort"], "tensor_shapes": dimensions, "all_checks_passed": True}


def benchmark_shape_summary(data: BenchmarkData) -> dict[str, list[int]]:
    return {
        name: list(getattr(data, name).shape)
        for name in ("corr_fmri", "pcorr_fmri", "corr_eeg", "pcorr_eeg", "pseudo_fmri", "pseudo_eeg")
    }


def vectorize_connectivity(data: BenchmarkData) -> np.ndarray:
    fmri_tri = np.triu_indices(data.corr_fmri.shape[1], k=1)
    eeg_tri = np.triu_indices(data.corr_eeg.shape[1], k=1)
    fmri = data.corr_fmri[:, fmri_tri[0], fmri_tri[1]]
    eeg = data.corr_eeg[:, eeg_tri[0], eeg_tri[1]]
    return np.concatenate([fmri, eeg], axis=1).astype(np.float32)


def metric_dict(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    scores = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    predictions = (scores >= threshold).astype(int)
    base = compute_classification_metrics(np.asarray(y_true, dtype=int), predictions, scores)
    return {
        "accuracy": float(base["acc"]),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(base["roc"]),
        "f1": float(base["f1"]),
        "precision": float(base["pre"]),
        "sensitivity": float(base["sen"]),
        "specificity": float(base["spe"]),
        "log_loss": float(base["log_loss"]),
    }


def confusion_counts(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    """Return binary confusion counts with an explicit, stable label order."""
    labels = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


class VectorMLP(nn.Module):
    """Compact multimodal FCNN for the small participant cohort."""

    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class PyGGATClassifier(nn.Module):
    """Two-layer PyG GAT using the MM row-wise weighted top-k graph."""

    def __init__(self, num_nodes: int, hidden_dim: int = 32, dropout: float = 0.3, k: int = 40) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv

        self.num_nodes = num_nodes
        self.k = k
        self.dropout = dropout
        self.conv1 = GATConv(num_nodes, hidden_dim, heads=4, concat=True, edge_dim=1, add_self_loops=False)
        self.conv2 = GATConv(hidden_dim * 4, 64, heads=1, concat=False, edge_dim=1, add_self_loops=False)
        self.norm = nn.LayerNorm(64)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, pearson: torch.Tensor, pcorr: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool

        adjacency = knn_thresholding(pcorr, k=self.k)
        features = (pearson + pseudo).reshape(-1, self.num_nodes)
        edges: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for batch_index in range(adjacency.shape[0]):
            edge = torch.nonzero(adjacency[batch_index] != 0, as_tuple=False).T
            edges.append(edge + batch_index * self.num_nodes)
            weights.append(adjacency[batch_index][edge[0], edge[1]].reshape(-1, 1))
        edge_index = torch.cat(edges, dim=1)
        edge_attr = torch.cat(weights, dim=0)
        batch = torch.arange(adjacency.shape[0], device=adjacency.device).repeat_interleave(self.num_nodes)
        x = self.conv1(features, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm(global_mean_pool(x, batch))
        return self.classifier(x)


def _grouped_calibration_splits(labels: np.ndarray, groups: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(3, len(np.unique(groups)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


def build_classical_model(name: str, labels: np.ndarray, groups: np.ndarray, seed: int):
    if name == "logistic":
        estimator = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    kernel = "linear" if name == "svm_linear" else "rbf"
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVC(C=1.0, kernel=kernel, gamma="scale", class_weight="balanced", probability=False, random_state=seed)),
        ]
    )
    return CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=_grouped_calibration_splits(labels, groups, seed),
    )


def _mm_model(data: BenchmarkData, *, use_fmri: bool, use_eeg: bool, fusion_mode: str) -> MMBrainXDL:
    return MMBrainXDL(
        in_channels_fmri=data.corr_fmri.shape[1],
        in_channels_eeg=data.corr_eeg.shape[1],
        embed_dim=128,
        fc_out=2,
        nrois_fmri=data.corr_fmri.shape[1],
        nrois_eeg=data.corr_eeg.shape[1],
        k_fmri=40,
        k_eeg=40,
        dropout=0.3,
        use_fmri=use_fmri,
        use_eeg=use_eeg,
        fusion_mode=fusion_mode,
    )


def build_neural_model(name: str, data: BenchmarkData, input_dim: int) -> nn.Module:
    if name == "fcnn":
        return VectorMLP(input_dim)
    if name == "gcn":
        return FmriGCN(data.corr_fmri.shape[1], 128, 128, 2, k=40, dropout=0.3)
    if name == "gcn_eeg":
        return FmriGCN(data.corr_eeg.shape[1], 128, 128, 2, k=40, dropout=0.3)
    if name == "gat":
        return PyGGATClassifier(data.corr_fmri.shape[1])
    if name == "eeg_only":
        return _mm_model(data, use_fmri=False, use_eeg=True, fusion_mode="gated")
    if name == "fmri_only":
        return _mm_model(data, use_fmri=True, use_eeg=False, fusion_mode="gated")
    if name == "concat_fusion":
        return LateFusionMLP(
            data.corr_fmri.shape[1], data.corr_eeg.shape[1], 128, 2,
            data.corr_fmri.shape[1], data.corr_eeg.shape[1], k_fmri=40, k_eeg=40, dropout=0.3,
        )
    if name == "equal_fusion":
        return _mm_model(data, use_fmri=True, use_eeg=True, fusion_mode="average")
    if name == "adaptive_fusion":
        return _mm_model(data, use_fmri=True, use_eeg=True, fusion_mode="gated")
    raise ValueError(f"Unsupported neural model: {name}")


def _tensor_arrays(data: BenchmarkData) -> dict[str, torch.Tensor]:
    return {
        "corr_fmri": torch.from_numpy(data.corr_fmri),
        "pcorr_fmri": torch.from_numpy(data.pcorr_fmri),
        "corr_eeg": torch.from_numpy(data.corr_eeg),
        "pcorr_eeg": torch.from_numpy(data.pcorr_eeg),
        "pseudo_fmri": torch.from_numpy(data.pseudo_fmri),
        "pseudo_eeg": torch.from_numpy(data.pseudo_eeg),
        "labels": torch.from_numpy(data.labels.astype(np.int64)),
    }


def _forward_model(name: str, model: nn.Module, arrays: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device, scaled_features: torch.Tensor | None) -> torch.Tensor:
    if name == "fcnn":
        if scaled_features is None:
            raise RuntimeError("FCNN requires scaled vector features")
        return model(scaled_features[indices].to(device))
    if name == "gcn_eeg":
        eeg = arrays["corr_eeg"][indices].to(device)
        pcorr_eeg = arrays["pcorr_eeg"][indices].to(device)
        pseudo_eeg = arrays["pseudo_eeg"][indices].to(device)
        return model(eeg, pcorr_eeg, pseudo_eeg)
    fmri = arrays["corr_fmri"][indices].to(device)
    pcorr_fmri = arrays["pcorr_fmri"][indices].to(device)
    pseudo_fmri = arrays["pseudo_fmri"][indices].to(device)
    if name in {"gcn", "gat"}:
        return model(fmri, pcorr_fmri, pseudo_fmri)
    eeg = arrays["corr_eeg"][indices].to(device)
    pcorr_eeg = arrays["pcorr_eeg"][indices].to(device)
    pseudo_eeg = arrays["pseudo_eeg"][indices].to(device)
    return model(fmri, eeg, pcorr_fmri, pcorr_eeg, pseudo_fmri, pseudo_eeg)


def _predict_neural(
    name: str,
    model: nn.Module,
    arrays: dict[str, torch.Tensor],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    scaled_features: torch.Tensor | None,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.as_tensor(indices, dtype=torch.long)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_indices,) in loader:
            logits = _forward_model(name, model, arrays, batch_indices, device, scaled_features)
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def train_neural_model(
    name: str,
    model: nn.Module,
    data: BenchmarkData,
    split: dict[str, np.ndarray],
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[nn.Module, list[dict[str, Any]], np.ndarray, np.ndarray]:
    arrays = _tensor_arrays(data)
    scaled_features: torch.Tensor | None = None
    if name == "fcnn":
        features = vectorize_connectivity(data)
        scaler = StandardScaler().fit(features[split["train"]])
        scaled_features = torch.as_tensor(scaler.transform(features).astype(np.float32))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(split["train"], dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for (batch_indices,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_model(name, model, arrays, batch_indices, device, scaled_features)
            labels = arrays["labels"][batch_indices].to(device)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(batch_indices)
            n_train += len(batch_indices)
        val_scores = _predict_neural(name, model, arrays, split["val"], device, batch_size, scaled_features)
        val_labels = data.labels[split["val"]]
        val_loss = float(compute_classification_metrics(val_labels, (val_scores >= 0.5).astype(int), val_scores)["log_loss"])
        history.append({"epoch": epoch, "train_loss": train_loss / max(n_train, 1), "val_log_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.to(device)
    val_scores = _predict_neural(name, model, arrays, split["val"], device, batch_size, scaled_features)
    test_scores = _predict_neural(name, model, arrays, split["test"], device, batch_size, scaled_features)
    return model, history, val_scores, test_scores


def _late_fusion_fold(
    data: BenchmarkData,
    split: dict[str, np.ndarray],
    device: torch.device,
    **training: Any,
) -> tuple[dict[str, nn.Module], list[dict[str, Any]], np.ndarray, np.ndarray]:
    models: dict[str, nn.Module] = {}
    histories: list[dict[str, Any]] = []
    validation: list[np.ndarray] = []
    testing: list[np.ndarray] = []
    for branch_name in ("fmri_only", "eeg_only"):
        branch = build_neural_model(branch_name, data, 0)
        trained, history, val_scores, test_scores = train_neural_model(
            branch_name, branch, data, split, device, **training
        )
        models[branch_name] = trained
        histories.extend({"branch": branch_name, **row} for row in history)
        validation.append(val_scores)
        testing.append(test_scores)
    return models, histories, np.mean(validation, axis=0), np.mean(testing, axis=0)


def _prediction_rows(
    data: BenchmarkData,
    indices: np.ndarray,
    scores: np.ndarray,
    *,
    model: str,
    seed: int,
    fold: int,
    threshold: float,
) -> list[dict[str, Any]]:
    predictions = (scores >= threshold).astype(int)
    return [
        {
            "observation_id": str(data.sample_ids[index]),
            "true_label": int(data.labels[index]),
            "predicted_class": int(predictions[position]),
            "predicted_probability": float(scores[position]),
            "participant_id": str(data.groups[index]),
            "session_id": str(data.sessions[index]),
            "sample_id": str(data.sample_ids[index]),
            "fold": fold,
            "seed": seed,
            "model": model,
            "validation_selected_threshold": float(threshold),
        }
        for position, index in enumerate(indices)
    ]


def _ensure_config(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=_json_value, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Configuration drift detected at {path}")
    else:
        atomic_write_text(path, encoded)


def _fold_directory(base: Path, fold: int, *, resume: bool, overwrite: bool) -> tuple[Path, bool]:
    destination = base / f"fold{fold:02d}"
    complete = destination / "complete.json"
    if complete.exists():
        if overwrite:
            shutil.rmtree(destination)
        elif resume:
            return destination, True
        else:
            raise FileExistsError(f"Fold is already complete: {destination}. Use --resume or --overwrite.")
    elif destination.exists() and any(destination.iterdir()):
        if overwrite:
            shutil.rmtree(destination)
        elif not resume:
            raise FileExistsError(f"Incomplete fold output exists: {destination}. Use --resume or --overwrite.")
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    return destination, False


def run_fold(
    *,
    model_name: str,
    seed: int,
    fold: int,
    data: BenchmarkData,
    manifest: dict[str, Any],
    output_root: Path,
    device_name: str,
    resume: bool,
    overwrite: bool,
    smoke_test: bool,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 8,
) -> Path:
    if model_name in UNAVAILABLE_MODELS:
        raise NotImplementedError(UNAVAILABLE_MODELS[model_name])
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model {model_name!r}")
    if not 1 <= fold <= 5:
        raise ValueError("fold must be 1..5")
    validate_manifest(manifest)
    split = fold_indices(manifest, fold)
    split_groups = {name: set(data.groups[indices]) for name, indices in split.items()}
    overlaps = {
        "train_val": split_groups["train"] & split_groups["val"],
        "train_test": split_groups["train"] & split_groups["test"],
        "val_test": split_groups["val"] & split_groups["test"],
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Participant leakage reached fold execution: {overlaps}")
    effective_seed = seed + fold
    seed_everything(effective_seed)
    model_output_name = model_name.removesuffix("_eeg")
    output_parent = output_root / ("smoke" if smoke_test else "")
    if model_name.endswith("_eeg"):
        output_parent = output_parent / "eeg_baselines"
    base = output_parent / f"seed{seed}" / model_output_name
    fold_dir, already_complete = _fold_directory(base, fold, resume=resume, overwrite=overwrite)
    if already_complete:
        print(f"[resume] {model_name} seed={seed} fold={fold} already complete")
        aggregate_experiment(base, model_name=model_name, seed=seed)
        return fold_dir
    config = {
        "benchmark": "Participant-grouped thesis re-evaluation",
        "model": model_name,
        "modality": MODEL_INFO[model_name][0],
        "model_class": MODEL_INFO[model_name][1],
        "seed": seed,
        "fold_manifest_seed": int(manifest.get("seed", 42)),
        "threshold": "validation balanced accuracy",
        "epochs": 2 if smoke_test else epochs,
        "patience": 2 if smoke_test else patience,
        "batch_size": batch_size,
        "optimizer": "Adam(lr=1e-3, weight_decay=5e-4)" if model_name in NEURAL_MODELS else None,
        "classical_features": "concatenated fMRI and EEG Pearson upper triangles" if model_name in CLASSICAL_MODELS else None,
        "pca": False,
        "k_fmri": 40,
        "k_eeg": 40,
        "smoke_test": smoke_test,
    }
    if model_name == "gcn_eeg":
        config.update({
            "connectivity": "Pearson node features; ridge partial correlation adjacency",
            "input_shape": list(data.corr_eeg.shape),
        })
    _ensure_config(base / "config.json", config)
    start = time.time()
    features = vectorize_connectivity(data)
    if model_name in CLASSICAL_MODELS:
        model = build_classical_model(
            model_name, data.labels[split["train"]], data.groups[split["train"]], effective_seed
        )
        model.fit(features[split["train"]], data.labels[split["train"]])
        val_scores = model.predict_proba(features[split["val"]])[:, 1]
        test_scores = model.predict_proba(features[split["test"]])[:, 1]
        atomic_write_pickle(fold_dir / "model.pkl", model)
        history: list[dict[str, Any]] = []
    else:
        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        training = {
            "seed": effective_seed,
            "epochs": 2 if smoke_test else epochs,
            "patience": 2 if smoke_test else patience,
            "batch_size": batch_size,
            "learning_rate": 1e-3,
            "weight_decay": 5e-4,
        }
        if model_name == "late_fusion":
            trained_models, history, val_scores, test_scores = _late_fusion_fold(data, split, device, **training)
            checkpoint = {name: model.state_dict() for name, model in trained_models.items()}
        else:
            neural_model = build_neural_model(model_name, data, features.shape[1])
            trained_model, history, val_scores, test_scores = train_neural_model(
                model_name, neural_model, data, split, device, **training
            )
            checkpoint = trained_model.state_dict()
        temporary_checkpoint = fold_dir / ".checkpoint.pt.tmp"
        torch.save(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, fold_dir / "checkpoint.pt")
    threshold, threshold_objective = choose_binary_threshold(
        data.labels[split["val"]], val_scores, "balanced_accuracy"
    )
    metrics = metric_dict(data.labels[split["test"]], test_scores, threshold)
    counts = confusion_counts(data.labels[split["test"]], test_scores >= threshold)
    fold_metrics = {
        "model": model_name,
        "seed": seed,
        "effective_seed": effective_seed,
        "fold": fold,
        "n_train": int(len(split["train"])),
        "n_val": int(len(split["val"])),
        "n_test": int(len(split["test"])),
        "train_participants": int(len(split_groups["train"])),
        "val_participants": int(len(split_groups["val"])),
        "test_participants": int(len(split_groups["test"])),
        "threshold": float(threshold),
        "threshold_objective": "validation_balanced_accuracy",
        "validation_balanced_accuracy_at_threshold": float(threshold_objective),
        **counts,
        **metrics,
    }
    atomic_write_json(fold_dir / "fold_metrics.json", fold_metrics)
    atomic_write_csv(fold_dir / "fold_metrics.csv", [fold_metrics])
    predictions = _prediction_rows(
        data, split["test"], test_scores, model=model_name, seed=seed, fold=fold, threshold=threshold
    )
    atomic_write_csv(fold_dir / "oof_predictions.csv", predictions)
    if history:
        atomic_write_csv(fold_dir / "training_history.csv", history)
    provenance = {
        "completed_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "manifest_seed": manifest.get("seed"),
        "manifest_n_samples": manifest.get("n_samples"),
        "participant_intersections": {"train_val": [], "train_test": [], "val_test": []},
        "smoke_test": smoke_test,
    }
    atomic_write_json(fold_dir / "run_metadata.json", provenance)
    atomic_write_json(fold_dir / "complete.json", {"complete": True, "model": model_name, "seed": seed, "fold": fold})
    aggregate_experiment(base, model_name=model_name, seed=seed)
    return fold_dir


def aggregate_experiment(base: Path, *, model_name: str, seed: int) -> None:
    base.mkdir(parents=True, exist_ok=True)
    lock_path = base / ".aggregate.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _aggregate_experiment_locked(base, model_name=model_name, seed=seed)


def _aggregate_experiment_locked(base: Path, *, model_name: str, seed: int) -> None:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = base / f"fold{fold:02d}"
        if not (fold_dir / "complete.json").exists():
            continue
        rows.append(json.loads((fold_dir / "fold_metrics.json").read_text(encoding="utf-8")))
        with (fold_dir / "oof_predictions.csv").open(newline="", encoding="utf-8") as handle:
            predictions.extend(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["fold"]))
    predictions.sort(key=lambda row: (int(row["fold"]), row["sample_id"]))
    atomic_write_csv(base / "fold_metrics.csv", rows)
    atomic_write_csv(base / "oof_predictions.csv", predictions)
    pooled: dict[str, Any] | None = None
    if len(rows) == 5:
        sample_ids = [row["sample_id"] for row in predictions]
        if len(predictions) != EXPECTED_COHORT["n_observations"] or len(set(sample_ids)) != EXPECTED_COHORT["n_observations"]:
            raise ValueError(f"{model_name}: expected exactly 120 unique held-out OOF observations")
        labels = np.asarray([int(row["true_label"]) for row in predictions], dtype=int)
        scores = np.asarray([float(row["predicted_probability"]) for row in predictions], dtype=float)
        hard_predictions = np.asarray([int(row["predicted_class"]) for row in predictions], dtype=int)
        hard_metrics = metric_dict(labels, hard_predictions.astype(float), 0.5)
        probability_metrics = metric_dict(labels, scores, 0.5)
        pooled = {"model": model_name, "seed": seed, "n_oof": len(predictions)}
        pooled.update(confusion_counts(labels, hard_predictions))
        for metric in METRICS:
            pooled[metric] = probability_metrics[metric] if metric in {"roc_auc", "log_loss"} else hard_metrics[metric]
        atomic_write_json(base / "pooled_oof_metrics.json", pooled)
        atomic_write_csv(base / "pooled_oof_metrics.csv", [pooled])
        confusion_rows = [
            {
                "fold": int(row["fold"]),
                "n_test": int(row["n_test"]),
                "threshold": float(row["threshold"]),
                **confusion_counts(
                    np.asarray([int(item["true_label"]) for item in predictions if int(item["fold"]) == int(row["fold"])]),
                    np.asarray([int(item["predicted_class"]) for item in predictions if int(item["fold"]) == int(row["fold"])]),
                ),
            }
            for row in rows
        ]
        confusion_rows.append({
            "fold": "pooled",
            "n_test": len(predictions),
            "threshold": "fold-specific validation thresholds",
            **confusion_counts(labels, hard_predictions),
        })
        atomic_write_csv(base / "confusion_counts.csv", confusion_rows)
    metadata = {
        "model": model_name,
        "seed": seed,
        "completed_folds": [int(row["fold"]) for row in rows],
        "complete": len(rows) == 5,
        "n_oof_predictions": len(predictions),
        "pooled_oof_metrics": pooled,
        "updated_at_unix": time.time(),
    }
    atomic_write_json(base / "run_metadata.json", metadata)
