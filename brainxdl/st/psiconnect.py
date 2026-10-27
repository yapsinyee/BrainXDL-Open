"""Prepare PsiConnect fMRI time series for the ST-BrainXDL pipeline.

The adapter deliberately reuses the established ST Ledoit-Wolf connectivity
and PyG graph conversion functions.  Its dataset-specific responsibilities are
limited to loading Schaefer time-series MAT files, assigning the same session
labels as the MM pipeline, and keeping all sessions from one participant in the
same cross-validation split.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold

from brainxdl.st.data_prep import convert2graphs, pad_graph_seq
from brainxdl.st.preprocessing import extract_ldw_corr, extract_ldw_corr_torch


DEFAULT_PATTERN = (
    "sub-*/ses-*/task-rest/"
    "*_task-rest_*_timeseries-tedanaGLM.mat"
)
_BIDS_PART_RE = re.compile(r"^(sub|ses)-(.+)$")


def _bids_entities(path: Path) -> tuple[str, str]:
    """Extract subject and session values from a PsiConnect derivative path."""
    entities: dict[str, str] = {}
    for part in path.parts:
        if "_" in part or "." in part:
            continue
        match = _BIDS_PART_RE.match(part)
        if match:
            entities[match.group(1)] = match.group(2)
    if "sub" not in entities or "ses" not in entities:
        filename_match = re.search(r"sub-([^_]+)_ses-([^_]+)_", path.name)
        if filename_match:
            entities["sub"] = filename_match.group(1)
            entities["ses"] = filename_match.group(2)
    if "sub" not in entities or "ses" not in entities:
        raise ValueError(f"Could not identify subject/session from path: {path}")
    return entities["sub"], entities["ses"]


def psiconnect_session_label(session: str) -> int:
    """Return the MM-compatible PsiConnect label (01=psilocybin, 02=control)."""
    value = str(session).removeprefix("ses-")
    if value.startswith("01") or value == "1":
        return 1
    if value.startswith("02") or value == "2":
        return 0
    raise ValueError(
        f"Unsupported PsiConnect session {session!r}; expected ses-01 or ses-02."
    )


def load_psiconnect_timeseries(
    timeseries_root: str | Path,
    pattern: str = DEFAULT_PATTERN,
    expected_rois: int = 200,
    matched_eeg_root: str | Path | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Load one rest-task TEDANA-GLM Schaefer time series per session.

    Returns time series, binary session labels, participant group identifiers,
    and stable sample identifiers in subject/session order.
    """
    root = Path(timeseries_root)
    if not root.is_dir():
        raise FileNotFoundError(f"PsiConnect time-series directory not found: {root}")

    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No PsiConnect MAT files matching {pattern!r} under {root}"
        )

    allowed_keys: set[tuple[str, str]] | None = None
    if matched_eeg_root is not None:
        eeg_root = Path(matched_eeg_root)
        if not eeg_root.is_dir():
            raise FileNotFoundError(f"Matched EEG directory not found: {eeg_root}")
        allowed_keys = set()
        for eeg_path in eeg_root.glob("*_task-rest_*.mat"):
            match = re.search(r"sub-([^_]+)_ses-([^_]+)_task-rest", eeg_path.name)
            if match:
                allowed_keys.add((match.group(1), match.group(2)))
        if not allowed_keys:
            raise FileNotFoundError(f"No rest-task EEG MAT files found in {eeg_root}")

    records = []
    seen_keys: set[tuple[str, str]] = set()
    for path in paths:
        subject, session = _bids_entities(path)
        key = (subject, session)
        if allowed_keys is not None and key not in allowed_keys:
            continue
        if key in seen_keys:
            raise ValueError(
                "Multiple time-series files matched participant/session "
                f"{subject}/{session}; narrow --pattern to one derivative."
            )
        seen_keys.add(key)

        content = loadmat(path)
        if "time_series" not in content:
            raise KeyError(f"time_series not found in {path}")
        timeseries = np.asarray(content["time_series"], dtype=np.float64)
        if timeseries.ndim != 2:
            raise ValueError(f"Expected a 2-D time_series in {path}, got {timeseries.shape}")
        if timeseries.shape[1] == expected_rois:
            pass
        elif timeseries.shape[0] == expected_rois:
            timeseries = timeseries.T
        else:
            raise ValueError(
                f"Expected {expected_rois} Schaefer ROIs in {path}, got {timeseries.shape}"
            )
        if not np.isfinite(timeseries).all():
            raise ValueError(f"Non-finite values found in {path}")

        records.append(
            (subject, session, timeseries, psiconnect_session_label(session), str(path))
        )

    if not records:
        raise ValueError("No PsiConnect fMRI samples remain after EEG cohort matching")

    records.sort(key=lambda item: (item[0], item[1]))
    data = [item[2] for item in records]
    labels = np.asarray([item[3] for item in records], dtype=np.int64)
    groups = np.asarray([item[0] for item in records], dtype=object)
    sample_ids = [f"sub-{item[0]}_ses-{item[1]}_task-rest" for item in records]
    return data, labels, groups, sample_ids


def get_grouped_outer_fold_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Build outer CV plus one inner validation split without group leakage."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups):
        raise ValueError("labels and groups must be equal-length one-dimensional arrays")
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"At least {n_splits} participant groups are required")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: dict[str, dict[str, np.ndarray]] = {}
    dummy = np.zeros(len(labels), dtype=np.uint8)
    for outer_index, (trainval_idx, test_idx) in enumerate(
        splitter.split(dummy, labels, groups), start=1
    ):
        inner_groups = groups[trainval_idx]
        inner_labels = labels[trainval_idx]
        inner_splits = min(n_splits, len(np.unique(inner_groups)))
        if inner_splits < 2:
            raise ValueError("At least two training participant groups are required")
        inner_splitter = StratifiedGroupKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=seed,
        )
        train_rel, val_rel = next(
            inner_splitter.split(
                np.zeros(len(trainval_idx), dtype=np.uint8),
                inner_labels,
                inner_groups,
            )
        )
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]

        split_groups = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
        if any(split_groups[i] & split_groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise RuntimeError("Participant leakage detected while constructing grouped folds")

        folds[f"outer{outer_index}_inner1"] = {
            "train": np.asarray(train_idx),
            "val": np.asarray(val_idx),
            "test": np.asarray(test_idx),
        }
    return folds


def get_grouped_nested_fold_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    outer_splits: int = 5,
    inner_splits: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Build strict grouped outer folds with grouped inner validation folds.

    Each key is ``outer{outer}_inner{inner}``. Test indices are identical across
    inner folds for one outer fold and are never included in inner splitting.
    """
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups):
        raise ValueError("labels and groups must be equal-length one-dimensional arrays")
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must both be at least 2")
    if len(np.unique(groups)) < outer_splits:
        raise ValueError(f"At least {outer_splits} participant groups are required")

    splitter = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=seed,
    )
    folds: dict[str, dict[str, np.ndarray]] = {}
    dummy = np.zeros(len(labels), dtype=np.uint8)
    for outer_index, (trainval_idx, test_idx) in enumerate(
        splitter.split(dummy, labels, groups), start=1
    ):
        inner_groups = groups[trainval_idx]
        inner_labels = labels[trainval_idx]
        effective_inner = min(inner_splits, len(np.unique(inner_groups)))
        if effective_inner < 2:
            raise ValueError("At least two training participant groups are required")
        inner_splitter = StratifiedGroupKFold(
            n_splits=effective_inner,
            shuffle=True,
            random_state=seed,
        )
        for inner_index, (train_rel, val_rel) in enumerate(
            inner_splitter.split(
                np.zeros(len(trainval_idx), dtype=np.uint8),
                inner_labels,
                inner_groups,
            ),
            start=1,
        ):
            train_idx = trainval_idx[train_rel]
            val_idx = trainval_idx[val_rel]
            split_groups = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
            if any(
                split_groups[i] & split_groups[j]
                for i, j in ((0, 1), (0, 2), (1, 2))
            ):
                raise RuntimeError(
                    "Participant leakage detected while constructing grouped folds"
                )
            folds[f"outer{outer_index}_inner{inner_index}"] = {
                "train": np.asarray(train_idx),
                "val": np.asarray(val_idx),
                "test": np.asarray(test_idx),
            }
    return folds


def _slice_and_convert(
    node_features: Sequence,
    adjacency: Sequence,
    sequence_lengths: torch.Tensor,
    labels: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return convert2graphs(
        [node_features[index] for index in indices],
        [adjacency[index] for index in indices],
        [sequence_lengths[index] for index in indices],
        [labels[index] for index in indices],
    )


def prepare_psiconnect_st(
    timeseries_root: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = DEFAULT_PATTERN,
    expected_rois: int = 200,
    matched_eeg_root: str | Path | None = None,
    window_size: int = 20,
    shift: int = 10,
    threshold_p: float = 0.40,
    n_splits: int = 5,
    inner_splits: int = 1,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Create grouped ST fold files and preprocessing provenance metadata."""
    if device != "cpu" and not str(device).startswith("cuda"):
        raise ValueError("device must be 'cpu' or a CUDA device such as 'cuda:0'")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA preprocessing was requested, but no CUDA device is available. "
            "Run inside a GPU allocation or pass --device cpu explicitly."
        )
    data, labels, groups, sample_ids = load_psiconnect_timeseries(
        timeseries_root,
        pattern=pattern,
        expected_rois=expected_rois,
        matched_eeg_root=matched_eeg_root,
    )
    if window_size <= 1 or shift <= 0 or shift > window_size:
        raise ValueError("Require window_size > 1 and 0 < shift <= window_size")
    too_short = [sample_ids[i] for i, item in enumerate(data) if item.shape[0] < window_size]
    if too_short:
        raise ValueError(
            f"Time series shorter than window_size={window_size}: {', '.join(too_short[:10])}"
        )

    if str(device).startswith("cuda"):
        node_features, adjacency, n_windows = extract_ldw_corr_torch(
            data,
            w_size=window_size,
            shift=shift,
            threshold_p=threshold_p,
            device=device,
        )
    else:
        node_features, adjacency, n_windows = extract_ldw_corr(
            data, w_size=window_size, shift=shift, threshold_p=threshold_p
        )
    padded_features, sequence_lengths = pad_graph_seq(node_features)
    padded_adjacency, _ = pad_graph_seq(adjacency)
    if inner_splits == 1:
        folds = get_grouped_outer_fold_indices(labels, groups, n_splits=n_splits, seed=seed)
    else:
        folds = get_grouped_nested_fold_indices(
            labels,
            groups,
            outer_splits=n_splits,
            inner_splits=inner_splits,
            seed=seed,
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for fold_key, split_indices in folds.items():
        payload = {
            f"{split}_graphs": _slice_and_convert(
                padded_features,
                padded_adjacency,
                sequence_lengths,
                labels,
                indices,
            )
            for split, indices in split_indices.items()
        }
        payload["metadata"] = {
            "dataset": "PsiConnect",
            "task": "rest",
            "label_mapping": {"0": "control", "1": "psilocybin"},
            "split_indices": {key: value.tolist() for key, value in split_indices.items()},
            "sample_ids": sample_ids,
            "participant_groups": groups.tolist(),
        }
        outer = fold_key.split("_")[0].removeprefix("outer")
        inner = fold_key.split("_")[1].removeprefix("inner")
        torch.save(payload, destination / f"graphs_outer{outer}_inner{inner}.pkl")

    ldw_payload = {
        "adj_mat": adjacency,
        "node_feat": node_features,
        "labels": labels,
        "groups": groups,
        "sample_ids": sample_ids,
    }
    with (destination / "LDW_psiconnect_data.pkl").open("wb") as handle:
        pickle.dump(ldw_payload, handle, protocol=4)

    manifest = {
        "dataset": "PsiConnect",
        "source_root": str(Path(timeseries_root).resolve()),
        "pattern": pattern,
        "task": "rest",
        "atlas": f"Schaefer-{expected_rois}",
        "timeseries_derivative": "tedanaGLM",
        "matched_eeg_root": (
            str(Path(matched_eeg_root).resolve()) if matched_eeg_root is not None else None
        ),
        "label_mapping": {"0": "control (ses-02)", "1": "psilocybin (ses-01)"},
        "n_samples": len(data),
        "n_participants": int(len(np.unique(groups))),
        "class_counts": {
            str(label): int(np.sum(labels == label)) for label in np.unique(labels)
        },
        "window_size": window_size,
        "shift": shift,
        "threshold_p": threshold_p,
        "windows_per_sample": n_windows,
        "n_splits": n_splits,
        "inner_splits": inner_splits,
        "seed": seed,
        "preprocessing_device": device,
        "fold_files": [f"graphs_{key}.pkl" for key in folds],
        "folds": {
            key: {
                split: {
                    "indices": indices.tolist(),
                    "sample_ids": [sample_ids[index] for index in indices],
                    "participant_groups": [str(groups[index]) for index in indices],
                }
                for split, indices in split_indices.items()
            }
            for key, split_indices in folds.items()
        },
    }
    with (destination / "psiconnect_st_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest
