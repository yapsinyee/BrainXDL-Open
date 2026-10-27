"""Temporal-block perturbation utilities for frozen ST-BrainXDL models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


CONTROL_FIELDS = ("y", "pad", "last")


def temporal_block_starts(
    num_windows: int,
    block_size: int,
    step: int = 1,
) -> np.ndarray:
    """Return every valid start for a fixed-width sliding temporal block."""
    if num_windows < 1:
        raise ValueError("num_windows must be positive.")
    if not 1 <= block_size <= num_windows:
        raise ValueError("block_size must lie between 1 and num_windows.")
    if step < 1:
        raise ValueError("step must be positive.")
    starts = np.arange(0, num_windows - block_size + 1, step, dtype=np.int64)
    if starts[-1] != num_windows - block_size:
        starts = np.append(starts, num_windows - block_size)
    return np.unique(starts)


def _clone_value(value):
    return value.clone() if torch.is_tensor(value) else value


def replace_temporal_blocks(
    graphs: Sequence,
    blocks: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Replace selected windows with their nearest unselected observed graph.

    Replacement preserves the target position's label, padding flag, and final-step
    flag while copying the complete, internally consistent graph representation
    from the nearest unselected window. The input sequence is never mutated.
    """
    num_windows = len(graphs)
    if num_windows < 2:
        raise ValueError("Temporal replacement requires at least two windows.")
    selected = np.zeros(num_windows, dtype=bool)
    for start, end in blocks:
        if not 0 <= int(start) < int(end) <= num_windows:
            raise ValueError(f"Invalid temporal block [{start}, {end}).")
        selected[int(start) : int(end)] = True
    if not selected.any():
        raise ValueError("At least one temporal window must be selected.")
    available = np.flatnonzero(~selected)
    if not len(available):
        raise ValueError("At least one temporal window must remain unselected.")

    output = np.empty(num_windows, dtype=object)
    for index, original in enumerate(graphs):
        if not selected[index]:
            replacement = original.clone()
        else:
            distances = np.abs(available - index)
            # np.argmin breaks equal-distance ties toward the earlier window.
            reference_index = int(available[int(np.argmin(distances))])
            replacement = graphs[reference_index].clone()
            for field in CONTROL_FIELDS:
                if hasattr(original, field):
                    setattr(replacement, field, _clone_value(getattr(original, field)))
        output[index] = replacement
    return output


def block_scores_to_windows(
    block_scores: Sequence[float],
    block_starts: Sequence[int],
    block_size: int,
    num_windows: int,
) -> np.ndarray:
    """Average overlapping block scores onto their constituent windows."""
    scores = np.asarray(block_scores, dtype=float)
    starts = np.asarray(block_starts, dtype=np.int64)
    if scores.ndim != 1 or starts.ndim != 1 or len(scores) != len(starts):
        raise ValueError("block_scores and block_starts must be aligned vectors.")
    if not np.isfinite(scores).all():
        raise ValueError("block_scores must be finite.")
    if block_size < 1 or num_windows < block_size:
        raise ValueError("Invalid block_size or num_windows.")
    totals = np.zeros(num_windows, dtype=float)
    counts = np.zeros(num_windows, dtype=np.int64)
    for score, start in zip(scores, starts):
        end = int(start) + block_size
        if start < 0 or end > num_windows:
            raise ValueError(f"Block [{start}, {end}) falls outside the sequence.")
        totals[int(start) : end] += float(score)
        counts[int(start) : end] += 1
    if np.any(counts == 0):
        raise ValueError("At least one window is not covered by any block.")
    return totals / counts


def select_nonoverlapping_blocks(
    scores: Sequence[float],
    starts: Sequence[int],
    block_size: int,
    count: int,
    condition: str,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Greedily select matched non-overlapping top, low, or random blocks."""
    values = np.asarray(scores, dtype=float)
    positions = np.asarray(starts, dtype=np.int64)
    if values.ndim != 1 or positions.ndim != 1 or len(values) != len(positions):
        raise ValueError("scores and starts must be aligned vectors.")
    if not np.isfinite(values).all() or block_size < 1 or count < 1:
        raise ValueError("Scores must be finite and block_size/count positive.")
    indices = np.arange(len(values), dtype=np.int64)
    if condition == "top":
        order = np.lexsort((indices, positions, -values))
    elif condition == "low":
        order = np.lexsort((indices, positions, values))
    elif condition == "random":
        if rng is None:
            raise ValueError("Random block selection requires an explicit generator.")
        order = rng.permutation(indices)
    else:
        raise ValueError(f"Unknown temporal selection condition: {condition!r}.")

    chosen: list[int] = []
    for index in order:
        start = int(positions[index])
        end = start + block_size
        if all(end <= int(positions[other]) or int(positions[other]) + block_size <= start for other in chosen):
            chosen.append(int(index))
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise ValueError(
            f"Could select only {len(chosen)} non-overlapping blocks; requested {count}."
        )
    return np.asarray(sorted(chosen, key=lambda index: int(positions[index])), dtype=np.int64)


def threshold_centered_log_odds(
    logits: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return binary log odds centred on a validation-selected threshold."""
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be strictly between zero and one.")
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("logits must have shape (batch, classes).")
    if logits.shape[1] == 1:
        log_odds = logits[:, 0]
    elif logits.shape[1] == 2:
        log_odds = logits[:, 1] - logits[:, 0]
    else:
        raise ValueError("Temporal importance currently supports binary classifiers only.")
    threshold_tensor = torch.as_tensor(
        float(threshold), dtype=log_odds.dtype, device=log_odds.device
    )
    threshold_log_odds = torch.logit(threshold_tensor)
    return log_odds - threshold_log_odds


def oriented_prediction_margin(
    logits: torch.Tensor,
    threshold: float,
    predicted_class: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distance from the decision threshold oriented to the chosen class."""
    centered = threshold_centered_log_odds(logits, threshold)
    if predicted_class is None:
        predicted_class = (centered >= 0).to(torch.long)
    predicted_class = predicted_class.to(centered.device, dtype=torch.long).reshape(-1)
    if predicted_class.shape != centered.shape or not torch.all(
        (predicted_class == 0) | (predicted_class == 1)
    ):
        raise ValueError("predicted_class must be a binary vector aligned with logits.")
    direction = predicted_class.to(centered.dtype).mul(2).sub(1)
    return direction * centered, predicted_class

