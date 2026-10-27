"""COBRE static/dynamic fMRI multi-view data loading."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from brainxdl.common.connectivity import generate_connectivity_matrix


@dataclass(frozen=True)
class CobreStaticDynamicData:
    """Aligned static and dynamic fMRI views for COBRE."""

    static_corr: np.ndarray
    static_pcorr: np.ndarray
    dynamic_corr: np.ndarray
    dynamic_adj: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    seq_lengths: np.ndarray


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_manifest(path: str | Path, require_usable: bool = True) -> list[dict[str, str]]:
    """Read the COBRE static/dynamic manifest.

    Args:
        path: Manifest CSV path.
        require_usable: Keep only rows marked usable for static/dynamic fMRI.

    Returns:
        List of manifest row dictionaries.
    """
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if require_usable:
        rows = [row for row in rows if _truthy(row.get("usable_for_static_dynamic_fmri", ""))]
    return rows


def _load_ldw(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _pad_square_sequences(sequences: Sequence[Sequence[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("No dynamic sequences provided.")
    seq_lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    if np.any(seq_lengths < 1):
        raise ValueError("Dynamic sequences must contain at least one window per subject.")

    first = np.asarray(sequences[0][0], dtype=np.float32)
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError(f"Dynamic windows must be square matrices, got shape {first.shape}.")
    n_nodes = first.shape[0]
    max_windows = int(seq_lengths.max())
    output = np.zeros((len(sequences), max_windows, n_nodes, n_nodes), dtype=np.float32)

    for subject_index, sequence in enumerate(sequences):
        for window_index, matrix in enumerate(sequence):
            matrix_arr = np.asarray(matrix, dtype=np.float32)
            if matrix_arr.shape != (n_nodes, n_nodes):
                raise ValueError(
                    "Dynamic window shape mismatch for subject "
                    f"{subject_index}, window {window_index}: {matrix_arr.shape} != {(n_nodes, n_nodes)}"
                )
            output[subject_index, window_index] = matrix_arr
    return output, seq_lengths


def load_cobre_static_dynamic_fmri(
    manifest_path: str | Path,
    *,
    require_usable: bool = True,
    expected_rois: int | None = 116,
    sort_by_dynamic_index: bool = True,
) -> CobreStaticDynamicData:
    """Load aligned COBRE static and dynamic fMRI views.

    The manifest must include ``dynamic_subject_index`` values assigned from
    sorted AAL ROI file order, matching the order used to build
    ``LDW_abide_data.pkl``.
    """
    rows = read_manifest(manifest_path, require_usable=require_usable)
    if not rows:
        raise ValueError(f"No manifest rows available from {manifest_path}.")
    if sort_by_dynamic_index:
        rows = sorted(rows, key=lambda row: int(row["dynamic_subject_index"]))

    ldw_paths = {row.get("dynamic_ldw_data_path", "") for row in rows}
    ldw_paths.discard("")
    if len(ldw_paths) != 1:
        raise ValueError(f"Expected exactly one dynamic LDW path, found {sorted(ldw_paths)}")

    ldw_path = Path(next(iter(ldw_paths)))
    if not ldw_path.is_file():
        raise FileNotFoundError(f"Dynamic LDW data not found: {ldw_path}")
    ldw_payload = _load_ldw(ldw_path)

    dynamic_corr_all = ldw_payload.get("node_feat")
    dynamic_adj_all = ldw_payload.get("adj_mat")
    dynamic_labels_all = np.asarray(ldw_payload.get("labels"), dtype=np.int64)
    if dynamic_corr_all is None or dynamic_adj_all is None:
        raise KeyError(f"LDW payload must contain 'node_feat' and 'adj_mat': {ldw_path}")

    static_corr = []
    static_pcorr = []
    dynamic_corr = []
    dynamic_adj = []
    labels = []
    subject_ids = []

    for row in rows:
        subject_id = row["subject_id"]
        roi_path = Path(row["static_roi_path"])
        if not roi_path.is_file():
            raise FileNotFoundError(f"Static ROI time-series not found for {subject_id}: {roi_path}")
        timeseries = np.loadtxt(roi_path)
        if timeseries.ndim != 2:
            raise ValueError(f"Static ROI time-series for {subject_id} is not 2D: {timeseries.shape}")
        if expected_rois is not None and timeseries.shape[1] != expected_rois:
            raise ValueError(
                f"Static ROI count mismatch for {subject_id}: "
                f"{timeseries.shape[1]} != {expected_rois}"
            )

        dynamic_subject_index = int(row["dynamic_subject_index"])
        if dynamic_subject_index < 0 or dynamic_subject_index >= len(dynamic_labels_all):
            raise IndexError(
                f"dynamic_subject_index out of bounds for {subject_id}: {dynamic_subject_index}"
            )

        label = int(row["label"])
        dynamic_label = int(dynamic_labels_all[dynamic_subject_index])
        if label != dynamic_label:
            raise ValueError(
                f"Label mismatch for {subject_id}: manifest label={label}, "
                f"dynamic label={dynamic_label}, dynamic_subject_index={dynamic_subject_index}"
            )

        static_corr.append(generate_connectivity_matrix(timeseries, "correlation"))
        static_pcorr.append(generate_connectivity_matrix(timeseries, "partial correlation"))
        dynamic_corr.append(dynamic_corr_all[dynamic_subject_index])
        dynamic_adj.append(dynamic_adj_all[dynamic_subject_index])
        labels.append(label)
        subject_ids.append(subject_id)

    dynamic_corr_arr, seq_lengths = _pad_square_sequences(dynamic_corr)
    dynamic_adj_arr, adj_seq_lengths = _pad_square_sequences(dynamic_adj)
    if not np.array_equal(seq_lengths, adj_seq_lengths):
        raise ValueError("Dynamic correlation and adjacency sequence lengths do not match.")

    return CobreStaticDynamicData(
        static_corr=np.stack(static_corr).astype(np.float32),
        static_pcorr=np.stack(static_pcorr).astype(np.float32),
        dynamic_corr=dynamic_corr_arr,
        dynamic_adj=dynamic_adj_arr,
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids, dtype=object),
        seq_lengths=seq_lengths,
    )


def _format_shape(array: np.ndarray) -> str:
    return " x ".join(str(dim) for dim in array.shape)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-load the COBRE static/dynamic fMRI multi-view manifest."
    )
    parser.add_argument(
        "--manifest_path",
        default="./data/cobre/manifests/cobre_static_dynamic_fmri_manifest.csv",
    )
    parser.add_argument("--expected_rois", type=int, default=116)
    parser.add_argument(
        "--include_unusable",
        action="store_true",
        help="Load all manifest rows instead of filtering to usable rows.",
    )
    args = parser.parse_args()

    try:
        data = load_cobre_static_dynamic_fmri(
            args.manifest_path,
            require_usable=not args.include_unusable,
            expected_rois=args.expected_rois,
        )
    except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
        print(f"Failed to load COBRE static/dynamic fMRI data: {exc}", file=sys.stderr)
        return 1
    counts = np.bincount(data.labels.astype(int), minlength=int(data.labels.max()) + 1)
    count_text = ", ".join(f"class {index}: {count}" for index, count in enumerate(counts))

    print(f"Loaded subjects: {len(data.labels)}")
    print(f"Static correlation shape: {_format_shape(data.static_corr)}")
    print(f"Static partial-correlation shape: {_format_shape(data.static_pcorr)}")
    print(f"Dynamic correlation shape: {_format_shape(data.dynamic_corr)}")
    print(f"Dynamic adjacency shape: {_format_shape(data.dynamic_adj)}")
    print(f"Sequence length range: {int(data.seq_lengths.min())}-{int(data.seq_lengths.max())}")
    print(f"Labels: {count_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
