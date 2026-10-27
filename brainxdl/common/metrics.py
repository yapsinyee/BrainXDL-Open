"""Classification metrics shared across ST and MM modules.

Provides:
- compute_probs_and_preds: logits → probabilities + hard predictions.
- compute_classification_metrics: full sklearn metric suite from arrays.
- METRIC_DIRECTIONS and metric_is_better are in common.logging; import
  from there if you need them alongside metrics.

Sources: consolidated from legacy_code/2-MM/main.py
  (compute_probs_and_preds, evaluate_metrics).
"""

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_probs_and_preds(
    logits: torch.Tensor,
    binary_mode: bool,
) -> tuple:
    """Convert raw logits to class probabilities and hard predictions.

    Args:
        logits:      Raw model outputs, shape (N, C) for multi-class or
                     (N, 1) / (N,) for binary.
        binary_mode: When True, apply sigmoid and threshold at 0.5.
                     When False, apply softmax and argmax.

    Returns:
        probs: Positive-class probability, shape (N,).
        preds: Hard integer predictions, shape (N,).

    Source: extracted from legacy_code/2-MM/main.py.
    """
    if binary_mode:
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs >= 0.5).long()
    else:
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
    return probs, preds


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    loss: float = float("nan"),
) -> Dict[str, float]:
    """Return a standard classification metric suite.

    Args:
        y_true:  Ground-truth integer labels, shape (N,).
        y_pred:  Hard predictions, shape (N,).
        y_score: Positive-class probability scores, shape (N,).
        loss:    Pre-computed scalar loss to include in the dict.

    Returns:
        dict with keys: loss, log_loss, acc, pre, sen, spe, f1, roc.

        - sen  = sensitivity = recall for the positive class (label 1).
        - spe  = specificity = recall for the negative class (label 0).

    Source: extracted from legacy_code/2-MM/main.py (evaluate_metrics).
    """
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="binary", zero_division=0)
    sen = recall_score(y_true, y_pred, average="binary", zero_division=0)
    spe = recall_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc = float("nan")
    try:
        nll = log_loss(y_true, y_score, labels=[0, 1])
    except ValueError:
        nll = float("nan")

    return {
        "loss": loss,
        "log_loss": nll,
        "acc": acc,
        "pre": pre,
        "sen": sen,
        "spe": spe,
        "f1": f1,
        "roc": roc,
    }


def binary_threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """Return deterministic candidate thresholds spanning finite scores."""
    scores = np.asarray(scores, dtype=float)
    unique_scores = np.unique(scores[np.isfinite(scores)])
    if unique_scores.size == 0:
        return np.asarray([0.5], dtype=float)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.concatenate(([0.0, 0.5, 1.0], unique_scores, midpoints))
    return np.unique(np.clip(candidates, 0.0, 1.0))


def choose_binary_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    objective: str,
) -> Tuple[float, float]:
    """Choose a binary threshold using labels from one tuning split only.

    Supported objectives are ``balanced_accuracy``, ``f1``, and ``youden``. Ties are
    resolved in favour of the threshold closest to 0.5, preserving the default
    decision boundary when validation evidence is equivalent. If equally good
    thresholds are also equally distant from 0.5, the smaller threshold wins.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores):
        raise ValueError("y_true and scores must be equal-length one-dimensional arrays")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Binary threshold selection requires labels 0 and 1")
    if objective not in {"balanced_accuracy", "f1", "youden"}:
        raise ValueError(f"Unknown threshold objective: {objective}")

    best_threshold = 0.5
    best_value = -np.inf
    for threshold in binary_threshold_candidates(scores):
        predictions = (scores >= threshold).astype(int)
        metrics = compute_classification_metrics(y_true, predictions, scores)
        if objective == "f1":
            value = metrics["f1"]
        elif objective == "balanced_accuracy":
            value = (metrics["sen"] + metrics["spe"]) / 2.0
        else:
            value = metrics["sen"] + metrics["spe"] - 1.0
        objective_better = value > best_value
        objective_tied = not objective_better and np.isclose(value, best_value)
        distance = abs(threshold - 0.5)
        best_distance = abs(best_threshold - 0.5)
        tie_break_better = objective_tied and (
            distance < best_distance and not np.isclose(distance, best_distance)
            or np.isclose(distance, best_distance) and threshold < best_threshold
        )
        if objective_better or tie_break_better:
            best_value = float(value)
            best_threshold = float(threshold)
    return best_threshold, best_value
