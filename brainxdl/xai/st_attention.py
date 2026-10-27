"""Attention capture and aggregation utilities for ST-BrainXDL.

The recorder instruments existing ``TransformerConv`` modules only while its
context manager is active. It returns the original layer output to the model, so
the ST forward pass and checkpoint state are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


@dataclass(frozen=True)
class STAttentionRecord:
    """One TransformerConv invocation captured during an ST forward pass."""

    module_name: str
    call_index: int
    edge_index: torch.Tensor
    alpha: torch.Tensor


class TransformerAttentionCapture:
    """Temporarily capture attention coefficients from all TransformerConv layers."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.records: list[STAttentionRecord] = []
        self._original_forwards: list[tuple[TransformerConv, Callable]] = []
        self._call_counts: dict[str, int] = {}

    def __enter__(self) -> "TransformerAttentionCapture":
        if self._original_forwards:
            raise RuntimeError("TransformerAttentionCapture cannot be entered twice.")
        for module_name, module in self.model.named_modules():
            if not isinstance(module, TransformerConv):
                continue
            original_forward = module.forward
            self._original_forwards.append((module, original_forward))

            def wrapped_forward(
                layer_self,
                *args,
                _name=module_name,
                _original=original_forward,
                **kwargs,
            ):
                if "return_attention_weights" in kwargs:
                    raise ValueError(
                        "ST attention capture owns return_attention_weights while active."
                    )
                output, attention = _original(
                    *args,
                    **kwargs,
                    return_attention_weights=True,
                )
                if not isinstance(attention, tuple) or len(attention) != 2:
                    raise TypeError(f"Unsupported attention return from {_name}.")
                edge_index, alpha = attention
                if not torch.is_tensor(edge_index) or not torch.is_tensor(alpha):
                    raise TypeError(
                        "ST attention capture currently requires tensor edge_index inputs."
                    )
                call_index = self._call_counts.get(_name, 0)
                self._call_counts[_name] = call_index + 1
                self.records.append(
                    STAttentionRecord(
                        module_name=_name,
                        call_index=call_index,
                        edge_index=edge_index.detach().cpu().clone(),
                        alpha=alpha.detach().cpu().clone(),
                    )
                )
                return output

            module.forward = MethodType(wrapped_forward, module)
        if not self._original_forwards:
            raise ValueError("Model contains no TransformerConv modules.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for module, original_forward in self._original_forwards:
            module.forward = original_forward
        self._original_forwards.clear()


_GRU_COMPONENTS = {
    "rnn_cell.weight_xz": "gru.update.input",
    "rnn_cell.weight_hz": "gru.update.hidden",
    "rnn_cell.weight_xr": "gru.reset.input",
    "rnn_cell.weight_hr": "gru.reset.hidden",
    "rnn_cell.weight_xh": "gru.candidate.input",
    "rnn_cell.weight_hh": "gru.candidate.hidden",
}


def component_name(module_name: str) -> str:
    """Map model module names to stable, human-readable attention components."""
    if module_name.startswith("enc_z_hidden."):
        return "posterior_encoder." + module_name.removeprefix("enc_z_hidden.")
    if module_name.startswith("prior_z_hidden."):
        return "prior_encoder." + module_name.removeprefix("prior_z_hidden.")
    for suffix, name in _GRU_COMPONENTS.items():
        if module_name == suffix or module_name.endswith("." + suffix):
            return name
    return module_name


def _record_time_index(record: STAttentionRecord) -> tuple[int, bool]:
    """Return the time index and whether the record feeds the current ST output."""
    return record.call_index, True


def dense_attention_matrix(
    record: STAttentionRecord,
    num_nodes: int,
    *,
    remove_self_loops: bool = True,
) -> np.ndarray:
    """Convert a captured directed edge list into a dense mean-head matrix."""
    edge_index = record.edge_index.numpy()
    alpha = record.alpha.numpy()
    if alpha.ndim == 2:
        alpha = alpha.mean(axis=1)
    elif alpha.ndim != 1:
        raise ValueError(f"Unexpected attention shape: {alpha.shape}")
    if edge_index.shape != (2, len(alpha)):
        raise ValueError("edge_index and alpha lengths differ.")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError(
            "Attention edge indices exceed num_nodes; extract one observation per forward pass."
        )

    matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    counts = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    for source, target, value in zip(edge_index[0], edge_index[1], alpha):
        source = int(source)
        target = int(target)
        if remove_self_loops and source == target:
            continue
        matrix[source, target] += float(value)
        counts[source, target] += 1
    present = counts > 0
    matrix[present] /= counts[present]
    return matrix


def records_to_component_series(
    records: Iterable[STAttentionRecord],
    num_nodes: int,
) -> dict[str, np.ndarray]:
    """Return directed attention matrices as ``component -> (time, ROI, ROI)``."""
    grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
    for record in records:
        time_index, keep = _record_time_index(record)
        if not keep:
            continue
        name = component_name(record.module_name)
        grouped.setdefault(name, []).append(
            (time_index, dense_attention_matrix(record, num_nodes))
        )

    output: dict[str, np.ndarray] = {}
    for name, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        time_indices = [item[0] for item in rows]
        if time_indices != list(range(len(rows))):
            raise ValueError(
                f"Attention calls for {name} do not form one record per time step: {time_indices[:20]}"
            )
        output[name] = np.stack([item[1] for item in rows]).astype(np.float32)
    return output


def symmetrize_attention(attention: np.ndarray) -> np.ndarray:
    """Create an undirected descriptive view by averaging both directions."""
    return 0.5 * (attention + np.swapaxes(attention, -1, -2))


def summarize_attention_series(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize temporal attention while keeping topology separate.

    Returns:
        mean_with_absent_zero: Mean attention over all windows; absent edges are zero.
        conditional_mean: Mean attention only over windows where the edge exists.
        edge_presence: Fraction of windows in which the directed edge exists.
    """
    if series.ndim != 3 or series.shape[-1] != series.shape[-2]:
        raise ValueError("Attention series must have shape (time, nodes, nodes).")
    presence = series > 0
    counts = presence.sum(axis=0)
    summed = series.sum(axis=0)
    conditional = np.divide(
        summed,
        counts,
        out=np.zeros_like(summed, dtype=np.float32),
        where=counts > 0,
    )
    return (
        series.mean(axis=0).astype(np.float32),
        conditional.astype(np.float32),
        presence.mean(axis=0).astype(np.float32),
    )


def top_undirected_edges(matrix: np.ndarray, top_k: int) -> list[tuple[int, int, float]]:
    """Rank ROI pairs from a directed attention matrix using a symmetric mean."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    symmetric = symmetrize_attention(matrix)
    upper = np.triu_indices(matrix.shape[0], k=1)
    values = symmetric[upper]
    order = np.argsort(values)[::-1]
    rows = []
    for position in order[: max(0, int(top_k))]:
        rows.append(
            (int(upper[0][position]), int(upper[1][position]), float(values[position]))
        )
    return rows
