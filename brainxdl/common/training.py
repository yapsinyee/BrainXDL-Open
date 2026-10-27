"""Shared training-loop utilities for ST and MM modules.

Provides:
- EarlyStopping: patience-based early stopping with best-param tracking.
- Generic train_one_epoch / evaluate / evaluate_full functions that wrap
  a caller-supplied forward_fn, suitable for the MM classification setup.
  The ST module has a more specialised loop (DAT, ST-BrainXDL losses) and uses
  these utilities only partially via EarlyStopping.

Sources: consolidated from
  legacy_code/1-ST/train.py  (early stopping logic inside train())
  legacy_code/2-MM/main.py   (train_one_epoch, evaluate, evaluate_metrics)
"""

import copy
import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brainxdl.common.metrics import compute_classification_metrics, compute_probs_and_preds


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Track a validation metric and decide when to stop training.

    Stores a deep copy of the best model state_dict so the caller can
    restore it after training completes.

    Usage::

        stopper = EarlyStopping(patience=30, metric="loss", mode="min")
        for epoch in range(max_epochs):
            val_loss = ...
            stopper.step(val_loss, model)
            if stopper.should_stop:
                break
        model.load_state_dict(stopper.best_state)

    Args:
        patience:    Number of epochs with no improvement before stopping.
        metric:      Name used only for display / logging purposes.
        mode:        'min' or 'max' — which direction is considered better.
                     Defaults to 'min' (lower is better, e.g. loss).
    """

    def __init__(
        self,
        patience: int = 30,
        metric: str = "loss",
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")
        self.patience = patience
        self.metric = metric
        self.mode = mode

        self.best_value: float = math.inf if mode == "min" else -math.inf
        self.best_state: Optional[Dict[str, Any]] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

    def _is_better(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def step(self, value: float, model: nn.Module, epoch: int = 0) -> bool:
        """Update the stopper with a new metric value.

        Args:
            value:  Current epoch's validation metric.
            model:  Model whose state_dict to snapshot when improved.
            epoch:  Current epoch number (for bookkeeping only).

        Returns:
            True if training should stop, False otherwise.
        """
        if self._is_better(value):
            self.best_value = value
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    def restore_best(self, model: nn.Module) -> None:
        """Load the best recorded state_dict back into *model*."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Generic training loop functions (MM-style classification)
# ---------------------------------------------------------------------------

def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    forward_fn: Callable,
    grad_clip: float = 0.0,
    binary_mode: bool = False,
) -> Tuple[float, float]:
    """Run one full pass over *loader* in training mode.

    Args:
        loader:      DataLoader yielding batches.
        model:       Model to train.
        criterion:   Loss function (nn.CrossEntropyLoss or nn.BCEWithLogitsLoss).
        optimizer:   Optimiser to step.
        device:      Device for tensors.
        forward_fn:  Callable(model, batch, device) → (logits, labels_int).
                     Encapsulates model-specific data unpacking.
        grad_clip:   Max L2 norm for gradient clipping.  0 disables clipping.
        binary_mode: When True use sigmoid + BCE; otherwise softmax + CE.

    Returns:
        (mean_loss, accuracy)

    Source: extracted from legacy_code/2-MM/main.py (train_one_epoch).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        logits, labels_int = forward_fn(model, batch, device)
        optimizer.zero_grad()

        if binary_mode:
            loss = criterion(logits, labels_int.float().unsqueeze(1))
        else:
            loss = criterion(logits, labels_int)

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = labels_int.size(0)
        total_loss += loss.item() * batch_size
        _, preds = compute_probs_and_preds(logits.detach(), binary_mode)
        correct += (preds == labels_int).sum().item()
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    forward_fn: Callable,
    binary_mode: bool = False,
) -> Tuple[float, float]:
    """Evaluate *model* over *loader* without gradient computation.

    Args:
        loader:      DataLoader for the eval split.
        model:       Model to evaluate.
        criterion:   Same loss function used during training.
        device:      Device for tensors.
        forward_fn:  Callable(model, batch, device) → (logits, labels_int).
        binary_mode: When True use sigmoid; otherwise softmax.

    Returns:
        (mean_loss, accuracy)

    Source: extracted from legacy_code/2-MM/main.py (evaluate).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            logits, labels_int = forward_fn(model, batch, device)

            if binary_mode:
                loss = criterion(logits, labels_int.float().unsqueeze(1))
            else:
                loss = criterion(logits, labels_int)

            batch_size = labels_int.size(0)
            total_loss += loss.item() * batch_size
            _, preds = compute_probs_and_preds(logits, binary_mode)
            correct += (preds == labels_int).sum().item()
            total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_full(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    forward_fn: Callable,
    binary_mode: bool = False,
) -> Dict[str, float]:
    """Evaluate *model* and return the full classification metric suite.

    Returns a dict with keys: loss, acc, pre, sen, spe, f1, roc.
    See brainxdl.common.metrics.compute_classification_metrics for details.

    Source: extracted from legacy_code/2-MM/main.py (evaluate_metrics).
    """
    model.eval()
    total_loss, total = 0.0, 0
    all_scores, all_preds, all_true = [], [], []

    with torch.no_grad():
        for batch in loader:
            logits, labels_int = forward_fn(model, batch, device)

            if binary_mode:
                loss = criterion(logits, labels_int.float().unsqueeze(1))
            else:
                loss = criterion(logits, labels_int)

            batch_size = labels_int.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            probs, preds = compute_probs_and_preds(logits, binary_mode)
            all_scores.extend(probs.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_true.extend(labels_int.detach().cpu().tolist())

    mean_loss = total_loss / max(total, 1)
    return compute_classification_metrics(
        y_true=np.array(all_true),
        y_pred=np.array(all_preds),
        y_score=np.array(all_scores),
        loss=mean_loss,
    )
