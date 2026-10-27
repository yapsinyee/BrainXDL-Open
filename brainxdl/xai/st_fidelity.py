"""Utilities for frozen-model ST-BrainXDL edge-perturbation fidelity tests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from brainxdl.xai.st_attention import (
    STAttentionRecord,
    records_to_component_series,
    symmetrize_attention,
)


EDGE_IMPORTANCE_DEFINITION = (
    "Per held-out observation, mean TransformerConv attention across heads, "
    "the two edge directions, all valid dynamic-FC windows, and all captured "
    "components; absent component/window edges contribute zero. Labels are not used."
)


def undirected_edge_pairs(num_nodes: int) -> np.ndarray:
    """Return every off-diagonal undirected FC candidate in canonical order."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2.")
    source, target = np.triu_indices(num_nodes, k=1)
    return np.column_stack((source, target)).astype(np.int64, copy=False)


def aggregate_attention_importance(
    records: Iterable[STAttentionRecord],
    num_nodes: int,
    valid_window_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Aggregate label-free attention into one score per undirected FC edge.

    Attention is averaged over heads by ``records_to_component_series``. Each
    directed matrix is then symmetrised by averaging the two directions. The
    final matrix is the equal-weight mean over valid temporal windows and all
    captured TransformerConv components. An edge absent from a component/window
    contributes zero, matching the descriptive attention extractor.
    """
    valid_window_mask = np.asarray(valid_window_mask, dtype=bool)
    if valid_window_mask.ndim != 1 or not valid_window_mask.any():
        raise ValueError("valid_window_mask must be one-dimensional and contain a valid window.")
    series_by_component = records_to_component_series(records, num_nodes)
    if not series_by_component:
        raise ValueError("No TransformerConv attention records were captured.")

    component_matrices = []
    component_names = list(series_by_component)
    for name in component_names:
        series = np.asarray(series_by_component[name], dtype=np.float32)
        if series.shape != (len(valid_window_mask), num_nodes, num_nodes):
            raise ValueError(
                f"{name}: attention shape {series.shape} does not match "
                f"({len(valid_window_mask)}, {num_nodes}, {num_nodes})."
            )
        symmetric = symmetrize_attention(series[valid_window_mask])
        component_matrices.append(symmetric.mean(axis=0))

    importance_matrix = np.mean(component_matrices, axis=0, dtype=np.float64)
    if not np.allclose(importance_matrix, importance_matrix.T, atol=1e-7):
        raise AssertionError("Aggregated edge importance is not symmetric.")
    np.fill_diagonal(importance_matrix, 0.0)
    pairs = undirected_edge_pairs(num_nodes)
    scores = importance_matrix[pairs[:, 0], pairs[:, 1]].astype(np.float32)
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Edge importance contains invalid values.")
    return scores, component_names


def removal_count(num_edges: int, fraction: float) -> int:
    """Convert a requested fraction to an exact, non-zero candidate count."""
    if num_edges < 1:
        raise ValueError("num_edges must be positive.")
    if not 0.0 < fraction < 1.0:
        raise ValueError("Removal fractions must be strictly between zero and one.")
    count = int(round(num_edges * float(fraction)))
    return min(max(count, 1), num_edges)


def select_edge_indices(
    importance: np.ndarray,
    fraction: float,
    condition: str,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select matched top, low, or deterministic-random FC candidate indices."""
    importance = np.asarray(importance, dtype=float)
    if importance.ndim != 1 or not np.isfinite(importance).all():
        raise ValueError("importance must be a finite one-dimensional array.")
    count = removal_count(len(importance), fraction)
    indices = np.arange(len(importance), dtype=np.int64)
    if condition == "top":
        order = np.lexsort((indices, -importance))
        selected = order[:count]
    elif condition == "low":
        order = np.lexsort((indices, importance))
        selected = order[:count]
    elif condition == "random":
        if rng is None:
            raise ValueError("Random edge selection requires an explicit generator.")
        selected = rng.choice(indices, size=count, replace=False)
    else:
        raise ValueError(f"Unknown perturbation condition: {condition!r}.")
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    if selected.shape != (count,) or len(np.unique(selected)) != count:
        raise AssertionError("Edge selection did not produce the requested unique count.")
    return selected


def deterministic_random_seed(
    seed: int,
    outer_fold: int,
    sample_index: int,
    fraction: float,
    repetition: int,
) -> int:
    """Derive a stable NumPy seed without relying on process-level hash state."""
    payload = f"{seed}|{outer_fold}|{sample_index}|{fraction:.8f}|{repetition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _assert_square_symmetric(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {tuple(tensor.shape)}.")
    if not torch.allclose(tensor, tensor.T, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"{name} is not symmetric before masking.")


def mask_graph_sequence(
    graphs: Sequence,
    edge_pairs: np.ndarray,
) -> np.ndarray:
    """Clone a graph sequence and remove selected FCs from every window.

    Dense correlation features, dense thresholded adjacency, sparse topology,
    and aligned edge attributes are updated together. Original graph objects are
    never modified and diagonal entries are preserved exactly.
    """
    edge_pairs = np.asarray(edge_pairs, dtype=np.int64)
    if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
        raise ValueError("edge_pairs must have shape (n_edges, 2).")
    if len(edge_pairs) != len(np.unique(edge_pairs, axis=0)):
        raise ValueError("edge_pairs contains duplicates.")

    output = np.empty(len(graphs), dtype=object)
    for window_index, original in enumerate(graphs):
        graph = original.clone()
        num_nodes = int(graph.x.shape[0])
        if np.any(edge_pairs < 0) or np.any(edge_pairs >= num_nodes):
            raise IndexError("Selected edge index lies outside the FC matrix.")
        if np.any(edge_pairs[:, 0] >= edge_pairs[:, 1]):
            raise ValueError("Each selected edge must use canonical source < target ordering.")

        _assert_square_symmetric(original.x, "graph.x")
        _assert_square_symmetric(original.adj, "graph.adj")
        x_diagonal = torch.diagonal(original.x).clone()
        adj_diagonal = torch.diagonal(original.adj).clone()
        source = torch.as_tensor(edge_pairs[:, 0], device=graph.x.device)
        target = torch.as_tensor(edge_pairs[:, 1], device=graph.x.device)
        graph.x[source, target] = 0
        graph.x[target, source] = 0

        adj_source = source.to(graph.adj.device)
        adj_target = target.to(graph.adj.device)
        graph.adj[adj_source, adj_target] = 0
        graph.adj[adj_target, adj_source] = 0

        sparse_source = graph.edge_index[0]
        sparse_target = graph.edge_index[1]
        sparse_low = torch.minimum(sparse_source, sparse_target)
        sparse_high = torch.maximum(sparse_source, sparse_target)
        selected_keys = torch.as_tensor(
            edge_pairs[:, 0] * num_nodes + edge_pairs[:, 1],
            dtype=torch.long,
            device=graph.edge_index.device,
        )
        sparse_keys = sparse_low * num_nodes + sparse_high
        remove = torch.isin(sparse_keys, selected_keys)
        keep = ~remove
        graph.edge_index = graph.edge_index[:, keep]
        if getattr(graph, "edge_attr", None) is not None:
            graph.edge_attr = graph.edge_attr[keep]

        if not torch.allclose(graph.x, graph.x.T, atol=1e-6, rtol=1e-6):
            raise AssertionError("Masked graph.x lost symmetry.")
        if not torch.allclose(graph.adj, graph.adj.T, atol=1e-6, rtol=1e-6):
            raise AssertionError("Masked graph.adj lost symmetry.")
        if not torch.equal(torch.diagonal(graph.x), x_diagonal):
            raise AssertionError("Masking changed the graph.x diagonal.")
        if not torch.equal(torch.diagonal(graph.adj), adj_diagonal):
            raise AssertionError("Masking changed the graph.adj diagonal.")
        # Assign one element at a time. ``torch_geometric.data.Data`` is
        # iterable, so ``np.asarray(list_of_data, dtype=object)`` incorrectly
        # expands each graph into its key/value pairs.
        output[window_index] = graph
    return output


def model_state_digest(model: nn.Module) -> str:
    """Hash model parameters and buffers to prove frozen evaluation is non-mutating."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def graph_sequence_digest(graphs: Sequence) -> str:
    """Hash every model-relevant graph tensor to detect accidental input mutation."""
    digest = hashlib.sha256()
    for window_index, graph in enumerate(graphs):
        digest.update(str(window_index).encode())
        for name in ("x", "adj", "edge_index", "edge_attr", "pad", "last", "y"):
            value = getattr(graph, name, None)
            if value is None:
                continue
            if torch.is_tensor(value):
                array = value.detach().cpu().contiguous().numpy()
            else:
                array = np.asarray(value)
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()
