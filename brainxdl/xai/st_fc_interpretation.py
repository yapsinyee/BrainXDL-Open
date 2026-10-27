"""Group-level FC interpretation utilities for ST-BrainXDL attention."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from brainxdl.xai.st_fidelity import undirected_edge_pairs


CANONICAL_NETWORKS = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)

NETWORK_DISPLAY = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience/Ventral Attention",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default Mode",
}


def edge_vector_to_matrix(
    values: np.ndarray,
    num_nodes: int,
    pairs: np.ndarray | None = None,
) -> np.ndarray:
    """Expand one undirected upper-triangle vector to a symmetric matrix."""
    values = np.asarray(values, dtype=float)
    pairs = undirected_edge_pairs(num_nodes) if pairs is None else np.asarray(pairs)
    if values.shape != (len(pairs),) or pairs.shape != (len(pairs), 2):
        raise ValueError("Edge values/pairs have incompatible shapes.")
    matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    matrix[pairs[:, 0], pairs[:, 1]] = values
    matrix[pairs[:, 1], pairs[:, 0]] = values
    if not np.allclose(matrix, matrix.T, atol=0.0, rtol=0.0):
        raise AssertionError("Expanded importance matrix is not symmetric.")
    if not np.array_equal(np.diag(matrix), np.zeros(num_nodes)):
        raise AssertionError("Expanded importance matrix diagonal is not zero.")
    return matrix


def group_importance(
    subject_importance: np.ndarray,
    num_nodes: int,
    pairs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return group mean/SD edge vectors and the symmetric group matrix."""
    values = np.asarray(subject_importance, dtype=float)
    pairs = undirected_edge_pairs(num_nodes) if pairs is None else np.asarray(pairs)
    if values.ndim != 2 or values.shape[1] != len(pairs):
        raise ValueError("subject_importance must have shape (subjects, edges).")
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Subject importance must contain at least two finite rows.")
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0, ddof=1)
    return mean, standard_deviation, edge_vector_to_matrix(mean, num_nodes, pairs)


def load_schaefer_labels(path: Path, expected_rois: int = 200) -> list[dict[str, object]]:
    """Load canonical Schaefer order labels and derive 7-network membership."""
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        index = int(fields[0])
        label = fields[1]
        parts = label.split("_")
        if len(parts) < 4 or parts[0] != "7Networks":
            raise ValueError(f"Unexpected Schaefer label: {label!r}.")
        network = parts[2]
        if network not in CANONICAL_NETWORKS:
            raise ValueError(f"Unknown Schaefer-7 network in {label!r}.")
        rows.append(
            {
                "node": index,
                "label": label,
                "hemisphere": parts[1],
                "network": network,
                "network_display": NETWORK_DISPLAY[network],
            }
        )
    expected_indices = list(range(1, expected_rois + 1))
    if [int(row["node"]) for row in rows] != expected_indices:
        raise ValueError(
            f"Atlas label order must be exactly 1..{expected_rois}; found {len(rows)} rows."
        )
    return rows


def rank_edges(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    pairs: np.ndarray,
    atlas_rows: Sequence[dict[str, object]],
    n_subjects: int,
) -> list[dict[str, object]]:
    """Create a deterministic descending ranking of all undirected FCs."""
    mean = np.asarray(mean, dtype=float)
    standard_deviation = np.asarray(standard_deviation, dtype=float)
    pairs = np.asarray(pairs, dtype=np.int64)
    if mean.shape != standard_deviation.shape or mean.shape != (len(pairs),):
        raise ValueError("Mean, SD, and edge-pair arrays differ in length.")
    tie_breaker = np.arange(len(mean), dtype=np.int64)
    order = np.lexsort((tie_breaker, -mean))
    output = []
    for rank, edge_index in enumerate(order, start=1):
        node_i, node_j = map(int, pairs[edge_index])
        roi_i, roi_j = atlas_rows[node_i], atlas_rows[node_j]
        output.append(
            {
                "rank": rank,
                "node_i": node_i + 1,
                "node_j": node_j + 1,
                "region_i": roi_i["label"],
                "region_j": roi_j["label"],
                "network_i": roi_i["network_display"],
                "network_j": roi_j["network_display"],
                "mean_attention_importance": float(mean[edge_index]),
                "std_attention_importance": float(standard_deviation[edge_index]),
                "n_subjects": int(n_subjects),
            }
        )
    return output


def aggregate_network_pairs(
    group_matrix: np.ndarray,
    atlas_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate undirected FC importance by Schaefer-7 network pair."""
    matrix = np.asarray(group_matrix, dtype=float)
    if matrix.shape != (len(atlas_rows), len(atlas_rows)):
        raise ValueError("Group matrix and atlas rows differ in size.")
    networks = np.asarray([str(row["network"]) for row in atlas_rows], dtype=str)
    output = []
    for first_index, network_1 in enumerate(CANONICAL_NETWORKS):
        first_nodes = np.flatnonzero(networks == network_1)
        if not len(first_nodes):
            raise ValueError(f"Atlas contains no nodes for {network_1}.")
        for network_2 in CANONICAL_NETWORKS[first_index:]:
            second_nodes = np.flatnonzero(networks == network_2)
            if network_1 == network_2:
                block = matrix[np.ix_(first_nodes, first_nodes)]
                values = block[np.triu_indices(len(first_nodes), k=1)]
                connection_type = "within-network"
            else:
                values = matrix[np.ix_(first_nodes, second_nodes)].reshape(-1)
                connection_type = "between-network"
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"Invalid network-pair values for {network_1}/{network_2}.")
            output.append(
                {
                    "network_1": NETWORK_DISPLAY[network_1],
                    "network_2": NETWORK_DISPLAY[network_2],
                    "connection_type": connection_type,
                    "n_edges": int(values.size),
                    "mean_attention_importance": float(values.mean()),
                    "sum_attention_importance": float(values.sum()),
                    "network_1_key": network_1,
                    "network_2_key": network_2,
                }
            )
    if sum(int(row["n_edges"]) for row in output) != len(undirected_edge_pairs(len(atlas_rows))):
        raise AssertionError("Network-pair aggregation does not cover every candidate edge once.")
    return output


def node_involvement(group_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return summed and max-normalised attention involvement per ROI."""
    matrix = np.asarray(group_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("group_matrix must be square.")
    score = matrix.sum(axis=1)
    maximum = float(score.max())
    normalised = score / maximum if maximum > 0 else np.zeros_like(score)
    return score, normalised
