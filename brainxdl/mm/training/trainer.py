"""Training loop, evaluation, and k-fold cross-validation for the MM module.

Source: extracted from legacy_code/2-MM/main.py (lines 927–2849).
"""

import argparse
import csv
import copy
import json
import os
import shutil
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from scipy.stats import wilcoxon
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from brainxdl.common.checkpointing import to_cpu_state_dict
from brainxdl.common.logging import METRIC_DIRECTIONS, format_metric_value, metric_is_better
from brainxdl.mm.utils.io import compute_run_slug, ensure_unique_path, get_task_list, resolve_checkpoint_dir, slugify


def compute_probs_and_preds(logits: torch.Tensor, binary_mode: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return class probabilities for the positive class and hard predictions."""
    if binary_mode:
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs >= 0.5).long()
    else:
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
    return probs, preds


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    grad_clip: float = 0.0,
    binary_mode: bool = False,
) -> Tuple[float, float]:
    """Run one training epoch and return (mean_loss, accuracy).

    Source: extracted from legacy_code/2-MM/main.py.
    """
    from brainxdl.mm.models.brainmgf import forward_model

    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        logits, labels_int, _, _ = forward_model(model, batch, device, args)
        optimizer.zero_grad()
        if binary_mode:
            loss = criterion(logits, labels_int.float().unsqueeze(1))
        else:
            loss = criterion(logits, labels_int)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    args: argparse.Namespace,
    binary_mode: bool = False,
) -> Tuple[float, float]:
    """Evaluate loss and accuracy on a loader without gradient updates.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    from brainxdl.mm.models.brainmgf import forward_model

    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            logits, labels_int, _, _ = forward_model(model, batch, device, args)
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


def evaluate_metrics(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    binary_mode: bool = False,
) -> Dict[str, float]:
    """Evaluate full metrics dict (loss, acc, pre, sen, spe, f1, roc).

    Source: extracted from legacy_code/2-MM/main.py.
    """
    from brainxdl.mm.models.brainmgf import forward_model

    model.eval()
    total_loss, total = 0.0, 0
    all_scores, all_preds, all_true = [], [], []

    with torch.no_grad():
        for batch in loader:
            logits, labels_int, _, _ = forward_model(model, batch, device, args)
            if binary_mode:
                loss = criterion(logits, labels_int.float().unsqueeze(1))
            else:
                loss = criterion(logits, labels_int)

            batch_size = labels_int.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            probs, preds = compute_probs_and_preds(logits, binary_mode)
            probs = probs.detach().cpu()
            preds = preds.detach().cpu()

            all_scores.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(labels_int.detach().cpu().tolist())

    y = np.array(all_true)
    p = np.array(all_preds)
    s = np.array(all_scores)

    acc = accuracy_score(y, p)
    pre = precision_score(y, p, average="binary", zero_division=0)
    sen = recall_score(y, p, average="binary", zero_division=0)
    spe = recall_score(y, p, average="binary", pos_label=0)
    f1 = f1_score(y, p, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y, s)
    except ValueError:
        roc = float("nan")
    try:
        nll = log_loss(y, s, labels=[0, 1])
    except ValueError:
        nll = float("nan")

    return {
        "loss": total_loss / max(total, 1),
        "log_loss": nll,
        "acc": acc,
        "pre": pre,
        "sen": sen,
        "spe": spe,
        "f1": f1,
        "roc": roc,
    }


def collect_prediction_records(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    binary_mode: bool = False,
) -> Tuple[float, List[Dict[str, float]]]:
    """Collect positive-class scores and labels for threshold diagnostics."""
    from brainxdl.mm.models.brainmgf import forward_model

    model.eval()
    total_loss, total = 0.0, 0
    records: List[Dict[str, float]] = []
    cursor = 0

    if isinstance(loader.dataset, torch.utils.data.Subset):
        source_indices = list(loader.dataset.indices)
    else:
        source_indices = list(range(len(loader.dataset)))

    with torch.no_grad():
        for batch in loader:
            logits, labels_int, _, _ = forward_model(model, batch, device, args)
            if binary_mode:
                loss = criterion(logits, labels_int.float().unsqueeze(1))
            else:
                loss = criterion(logits, labels_int)

            probs, preds = compute_probs_and_preds(logits, binary_mode)
            probs = probs.detach().cpu().tolist()
            preds = preds.detach().cpu().tolist()
            labels = labels_int.detach().cpu().tolist()
            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            total += batch_size

            for offset, (label, score, pred) in enumerate(zip(labels, probs, preds)):
                records.append(
                    {
                        "sample_index": int(source_indices[cursor + offset]),
                        "label": int(label),
                        "score": float(score),
                        "prediction": int(pred),
                    }
                )
            cursor += batch_size

    return total_loss / max(total, 1), records


def write_prediction_records(
    path: str,
    *,
    fold: int,
    split: str,
    records: List[Dict[str, float]],
    groups: Optional[np.ndarray] = None,
    tasks: Optional[np.ndarray] = None,
) -> None:
    """Write prediction records with optional sample metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["fold", "split", "sample_index", "group", "task", "label", "score", "prediction"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            sample_index = int(record["sample_index"])
            writer.writerow(
                {
                    "fold": fold,
                    "split": split,
                    "sample_index": sample_index,
                    "group": "" if groups is None else str(groups[sample_index]),
                    "task": "" if tasks is None else str(tasks[sample_index]),
                    "label": int(record["label"]),
                    "score": f"{float(record['score']):.10f}",
                    "prediction": int(record["prediction"]),
                }
            )


def analyze_fusion_gate_bias(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
    device: torch.device,
    all_tasks: np.ndarray,
    split_name: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Aggregate fusion gate weights per task and run Wilcoxon tests vs 0.5.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    from brainxdl.mm.models.brainmgf import forward_model

    if not hasattr(model, "fusion_mode") or getattr(model, "fusion_mode", None) != "gated":
        return None
    if not getattr(model, "use_fmri", False) or not getattr(model, "use_eeg", False):
        return None
    if all_tasks is None or len(all_tasks) == 0:
        warnings.warn("Task labels unavailable; skipping fusion gate analysis.")
        return None

    if isinstance(dataset, torch.utils.data.Subset):
        subset_indices = np.asarray(dataset.indices)
    else:
        subset_indices = np.arange(len(dataset))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    gathered_indices: List[np.ndarray] = []
    gathered_weights: List[np.ndarray] = []

    model.eval()
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            _, _, _, gate_weights = forward_model(model, batch, device, args, return_gates=True)
            if gate_weights is None:
                continue
            weights_np = np.asarray(gate_weights.detach().cpu().tolist(), dtype=np.float32)
            batch_size = weights_np.shape[0]
            batch_indices = subset_indices[cursor : cursor + batch_size]
            cursor += batch_size
            gathered_indices.append(np.asarray(batch_indices))
            gathered_weights.append(weights_np)

    if not gathered_weights:
        warnings.warn("No gate weights collected; skipping fusion gate analysis.")
        return None

    all_indices = np.concatenate(gathered_indices)
    all_weights = np.concatenate(gathered_weights, axis=0)
    if all_indices.size != all_weights.shape[0]:
        warnings.warn("Mismatch between collected gate weights and indices; skipping analysis.")
        return None

    try:
        task_subset = np.asarray(all_tasks)[all_indices]
    except IndexError:
        warnings.warn("Task indices out of range; skipping fusion gate analysis.")
        return None

    unique_tasks = sorted({str(t) for t in task_subset})
    if not unique_tasks:
        warnings.warn("No tasks found for gate analysis.")
        return None

    metrics: Dict[str, Dict[str, float]] = {}
    lines: List[str] = []
    header = f"Fusion gate analysis on {split_name} split (n={all_weights.shape[0]}, tasks={len(unique_tasks)})"
    print(header)
    lines.append(header)
    lines.append("Task | n | α_fMRI_mean | α_fMRI_var | α_EEG_mean | α_EEG_var | Δα (fMRI-0.5) | Wilcoxon p | Lean")

    for task in unique_tasks:
        mask = task_subset == task
        if not np.any(mask):
            continue
        task_weights = all_weights[mask]
        fmri_alpha = task_weights[:, 0]
        eeg_alpha = task_weights[:, 1]
        n_samples = fmri_alpha.size
        fmri_mean = float(fmri_alpha.mean())
        eeg_mean = float(eeg_alpha.mean())
        fmri_var = float(np.var(fmri_alpha, ddof=1)) if n_samples > 1 else 0.0
        eeg_var = float(np.var(eeg_alpha, ddof=1)) if n_samples > 1 else 0.0
        delta = fmri_mean - 0.5
        try:
            stat, p_value = wilcoxon(fmri_alpha - 0.5, zero_method="wilcox")
            stat = float(stat)
            p_value = float(p_value)
        except ValueError:
            stat = float("nan")
            p_value = float("nan")
        lean_dir = "fMRI" if fmri_mean > 0.5 else ("EEG" if fmri_mean < 0.5 else "Balanced")
        title_task = task.replace("_", " ").title()
        lines.append(
            f"{title_task} | {n_samples} | {fmri_mean:.3f} | {fmri_var:.3f} | "
            f"{eeg_mean:.3f} | {eeg_var:.3f} | {delta:.3f} | {p_value:.4g} | {lean_dir}"
        )
        metrics[task] = {
            "n": n_samples,
            "fmri_mean": fmri_mean,
            "fmri_var": fmri_var,
            "eeg_mean": eeg_mean,
            "eeg_var": eeg_var,
            "delta": delta,
            "wilcoxon_stat": stat,
            "wilcoxon_p": p_value,
            "lean": lean_dir,
        }

    for line in lines[1:]:
        print("  " + line)

    metrics_dir = getattr(args, "metrics_dir", None) or "result"
    if not os.path.isabs(metrics_dir):
        metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)

    gate_dir = os.path.join(metrics_dir, "fusion_gates")
    os.makedirs(gate_dir, exist_ok=True)
    slug = compute_run_slug(args)
    report_path = os.path.join(gate_dir, f"gate_analysis_{slug}_{split_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved fusion gate report to {report_path}")

    return metrics


def plot_training_history(history: Dict[str, List[float]], fold_idx: int, args: argparse.Namespace) -> None:
    """Plot training/validation loss and accuracy for a fold.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is not installed; skipping training curve plotting.")
        return

    report_stem = getattr(args, "_kfold_report_stem", None)
    run_dir = getattr(args, "_kfold_run_dir", None)

    if run_dir:
        curve_root = os.path.join(run_dir, "train-val curves")
    else:
        plot_dir = getattr(args, "plot_dir", "plots")
        if not os.path.isabs(plot_dir):
            plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_dir)
        curve_root = os.path.join(plot_dir, "train-val curves")
    if report_stem:
        curve_dir = os.path.join(curve_root, report_stem)
    else:
        curve_dir = curve_root
    os.makedirs(curve_dir, exist_ok=True)

    epochs = history["epoch"]
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"Fold {fold_idx} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title(f"Fold {fold_idx} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    model_name = getattr(args, "model", "model")
    modality = getattr(args, "modality", "modality")
    loss_type = getattr(args, "loss_type", "loss")
    tasks_list = get_task_list(args)
    if tasks_list:
        task_descriptor = "-".join(sorted(set(tasks_list)))
    else:
        task_descriptor = str(getattr(args, "task", "all"))

    if report_stem:
        filename = f"{report_stem}_fold{fold_idx}.png"
    else:
        filename = "training_curve_{model}_{task}_{modality}_loss-{loss}_fold{fold}.png".format(
            model=slugify(model_name),
            task=slugify(task_descriptor),
            modality=slugify(modality),
            loss=slugify(loss_type),
            fold=fold_idx,
        )
    out_path = os.path.join(curve_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved training curve to {out_path}")
    should_show = getattr(args, "plot_show", False) and not getattr(args, "no_plot_show", False)
    if should_show:
        try:
            try:
                manager = plt.get_current_fig_manager()
                if manager is not None:
                    manager.set_window_title(f"Training Curve · Fold {fold_idx}")
            except Exception:
                pass
            plt.show(block=True)
        except Exception as exc:
            warnings.warn(f"Unable to display training curve interactively: {exc}")
    plt.close(fig)


def run_kfold_cv(
    dataset: TensorDataset,
    labels_np: np.ndarray,
    args: argparse.Namespace,
    tasks: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """Run stratified k-fold cross-validation and return per-fold metric dicts.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    from brainxdl.mm.data.dataset import build_weighted_sampler, compute_min_max_ratio, infer_pos_weight, summarize_split_counts
    from brainxdl.xai.explainer import maybe_run_gnn_explanations
    from brainxdl.mm.models.brainmgf import build_model
    from brainxdl.xai.latent import generate_latent_manifolds

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: List[Dict[str, float]] = []
    num_classes = int(labels_np.max()) + 1
    binary_mode = args.loss_type == "bce"
    model_out_dim = 1 if binary_mode else args.out_channels
    sample_indices = np.arange(len(labels_np))
    split_strategy = "stratified"
    outer_splits = None
    split_seed = getattr(args, "split_seed", None)
    if split_seed is None:
        split_seed = args.seed

    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(labels_np):
            if getattr(args, "require_grouped_cv", False):
                raise ValueError(
                    "Participant groups must be a one-dimensional label-aligned array; "
                    "refusing observation-level cross-validation."
                )
            warnings.warn("Ignoring participant groups because they are not a one-dimensional label-aligned array.")
            groups = None
    elif getattr(args, "require_grouped_cv", False):
        raise ValueError(
            "Participant groups are required; refusing observation-level cross-validation."
        )

    fold_splits = getattr(args, "_fold_splits", None)
    if fold_splits is not None:
        outer_splits = [
            (np.concatenate([train_idx, val_idx]), test_idx)
            for train_idx, val_idx, test_idx in fold_splits
        ]
        split_strategy = "manifest_grouped"
        print("Using fold splits from external manifest.")
    elif groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < args.n_folds:
            raise ValueError(
                f"At least {args.n_folds} participant groups are required for grouped k-fold CV; "
                f"found {len(unique_groups)}."
            )
        splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=split_seed)
        outer_splits = splitter.split(sample_indices, labels_np, groups)
        split_strategy = "stratified_group"
        print(f"Using StratifiedGroupKFold with {len(unique_groups)} participant groups.")
    else:
        splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=split_seed)
        outer_splits = splitter.split(sample_indices, labels_np)
        print("Using StratifiedKFold without participant grouping.")

    metrics_dir = getattr(args, "metrics_dir", None) or getattr(args, "plot_dir", "plots")
    if not os.path.isabs(metrics_dir):
        metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)

    save_best = getattr(args, "save_best_model", False)
    monitored_metric = getattr(args, "save_metric", "acc")
    if save_best and monitored_metric not in METRIC_DIRECTIONS:
        warnings.warn(
            f"Metric '{monitored_metric}' is not supported for checkpointing; disabling --save_best_model."
        )
        save_best = False
    best_checkpoint_metric: Optional[float] = None
    best_checkpoint_fold: Optional[int] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None
    metric_warning_emitted = False

    tasks_list = get_task_list(args)
    if tasks_list:
        task_descriptor = "-".join(sorted(set(tasks_list)))
    else:
        task_descriptor = str(getattr(args, "task", "all"))

    report_stem = "kfold_results_{model}_{task}_{modality}_loss-{loss}".format(
        model=slugify(getattr(args, "model", "model")),
        task=slugify(task_descriptor),
        modality=slugify(getattr(args, "modality", "modality")),
        loss=slugify(getattr(args, "loss_type", "loss")),
    )
    report_filename = f"{report_stem}.txt"
    result_root = os.path.join(metrics_dir, "result")
    base_run_dir = os.path.join(result_root, report_stem)
    run_dir = ensure_unique_path(base_run_dir)
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, report_filename)
    base_report_path = os.path.join(base_run_dir, report_filename)

    args._kfold_run_dir = run_dir
    args._kfold_report_path = report_path
    args._kfold_report_stem = os.path.basename(run_dir)

    checkpoint_identifier = args._kfold_report_stem
    if save_best:
        if not checkpoint_identifier:
            checkpoint_identifier = f"kfold_{compute_run_slug(args)}"
        checkpoint_root = resolve_checkpoint_dir(args, metrics_dir)
        checkpoint_dir = os.path.join(checkpoint_root, checkpoint_identifier)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_identifier}_best.pt")

    for fold, (trainval_idx, test_idx) in enumerate(outer_splits, start=1):
        fold_dataset = dataset
        st_embeddings_by_fold = getattr(args, "_st_embeddings_by_fold", None)
        if st_embeddings_by_fold is not None:
            if fold > len(st_embeddings_by_fold):
                raise ValueError(f"Missing ST embeddings for fold {fold}.")
            st_embedding_t = torch.tensor(st_embeddings_by_fold[fold - 1], dtype=torch.float32)
            fold_dataset = TensorDataset(*dataset.tensors, st_embedding_t)
            fold_embedding_dims = getattr(args, "_st_embedding_dims_by_fold", None)
            if fold_embedding_dims is not None:
                args.st_embedding_dim = int(fold_embedding_dims[fold - 1])

        t_labels = labels_np[trainval_idx]
        if fold_splits is not None:
            train_idx, val_idx, manifest_test_idx = fold_splits[fold - 1]
            if not np.array_equal(np.asarray(test_idx), np.asarray(manifest_test_idx)):
                raise RuntimeError(f"Manifest test indices differ for fold {fold}.")
            print(
                f"Fold {fold} manifest split -> "
                f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
            )
        elif groups is not None:
            trainval_groups = groups[trainval_idx]
            unique_trainval_groups = np.unique(trainval_groups)
            requested_val_ratio = float(getattr(args, "val_ratio", 0.2))
            requested_val_ratio = min(max(requested_val_ratio, 1.0 / max(len(unique_trainval_groups), 1)), 0.5)
            val_splits = int(round(1.0 / requested_val_ratio))
            val_splits = min(max(val_splits, 2), len(unique_trainval_groups))
            val_splitter = StratifiedGroupKFold(n_splits=val_splits, shuffle=True, random_state=split_seed + fold)
            train_rel, val_rel = next(val_splitter.split(trainval_idx, t_labels, trainval_groups))
            train_idx = trainval_idx[train_rel]
            val_idx = trainval_idx[val_rel]
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            test_groups = set(groups[test_idx])
            if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
                raise RuntimeError(f"Participant leakage detected in fold {fold}.")
            print(
                f"Fold {fold} participant groups -> "
                f"train={len(train_groups)}, val={len(val_groups)}, test={len(test_groups)}"
            )
        else:
            train_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=getattr(args, "val_ratio", 0.2),
                stratify=t_labels,
                random_state=split_seed,
            )

        if groups is not None:
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            test_groups = set(groups[test_idx])
            if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
                raise RuntimeError(f"Participant leakage detected in fold {fold}.")

        train_dataset = Subset(fold_dataset, train_idx)
        val_dataset = Subset(fold_dataset, val_idx)
        test_dataset = Subset(fold_dataset, test_idx)

        train_hist = summarize_split_counts(f"Fold {fold} train", train_idx, labels_np, num_classes=num_classes)
        val_hist = summarize_split_counts(f"Fold {fold} val", val_idx, labels_np, num_classes=num_classes)
        test_hist = summarize_split_counts(f"Fold {fold} test", test_idx, labels_np, num_classes=num_classes)

        imbalance_ratio = compute_min_max_ratio(train_hist)
        auto_balance = getattr(args, "balance_threshold", 0.0) > 0 and imbalance_ratio < args.balance_threshold
        use_sampler = getattr(args, "balance_batches", False) or auto_balance
        if use_sampler:
            sampler = build_weighted_sampler(labels_np[train_idx], train_hist)
            if auto_balance and not getattr(args, "balance_batches", False):
                print(
                    f"  Fold {fold}: activating WeightedRandomSampler (ratio={imbalance_ratio:.3f} < "
                    f"threshold={args.balance_threshold:.3f})"
                )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        n_fmri = train_dataset[0][0].shape[0]
        n_eeg = train_dataset[0][1].shape[0]

        model = build_model(args, n_fmri, n_eeg, model_out_dim, device)

        if binary_mode:
            if args.label_smoothing != 0.0:
                warnings.warn("Label smoothing is ignored when using BCE loss.")
            if args.pos_weight is not None:
                pos_weight_value = float(args.pos_weight)
            else:
                pos_weight_value = infer_pos_weight(train_hist)
            pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            print(f"  Fold {fold}: Using BCEWithLogitsLoss (pos_weight={pos_weight_value:.3f})")
        else:
            if getattr(args, "class_weighted", False):
                class_counts = np.bincount(labels_np[train_idx], minlength=num_classes)
                class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
                class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
                criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
                print(f"  Fold {fold}: Using class-weighted CrossEntropyLoss with weights={class_weights.detach().cpu().tolist()}")
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        best_metric = float("inf") if args.es_metric == "loss" else -float("inf")
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0

        history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        print(f"Fold {fold}/{args.n_folds}")
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                device,
                args,
                grad_clip=getattr(args, "grad_clip", 0.0),
                binary_mode=binary_mode,
            )
            val_loss, val_acc = evaluate(val_loader, model, criterion, device, args, binary_mode=binary_mode)
            scheduler.step(val_loss)
            print(
                f"  Epoch {epoch:03d} | Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
                f"Val loss={val_loss:.4f}, acc={val_acc:.4f}"
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            curr = val_loss if args.es_metric == "loss" else val_acc
            improved = (
                (best_metric - curr) > args.es_min_delta
                if args.es_metric == "loss"
                else (curr - best_metric) > args.es_min_delta
            )
            if improved:
                best_metric = curr
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if (not getattr(args, "no_early_stop", False)) and epoch >= args.min_epochs and no_improve >= args.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if getattr(args, "plot_curves", False):
            plot_training_history(history, fold, args)

        model.load_state_dict(best_state)
        if getattr(args, "save_predictions", False):
            prediction_dir = os.path.join(run_dir, "predictions")
            _, val_records = collect_prediction_records(
                val_loader,
                model,
                criterion,
                device,
                args,
                binary_mode=binary_mode,
            )
            _, test_records = collect_prediction_records(
                test_loader,
                model,
                criterion,
                device,
                args,
                binary_mode=binary_mode,
            )
            write_prediction_records(
                os.path.join(prediction_dir, f"fold{fold:02d}_val_predictions.csv"),
                fold=fold,
                split="val",
                records=val_records,
                groups=groups,
                tasks=tasks,
            )
            write_prediction_records(
                os.path.join(prediction_dir, f"fold{fold:02d}_test_predictions.csv"),
                fold=fold,
                split="test",
                records=test_records,
                groups=groups,
                tasks=tasks,
            )
        fold_metrics = evaluate_metrics(
            test_loader,
            model,
            criterion,
            device,
            args,
            binary_mode=binary_mode,
        )
        print(
            f"  Test: acc={fold_metrics['acc']:.4f}, loss={fold_metrics['loss']:.4f}, "
            f"f1={fold_metrics['f1']:.4f}, roc={fold_metrics['roc']:.4f}"
        )

        if tasks is not None:
            analyze_fusion_gate_bias(
                model,
                test_dataset,
                args,
                device,
                tasks,
                split_name=f"fold{fold}_test",
            )

        split_map = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        maybe_run_gnn_explanations(
            model,
            split_map,
            args,
            device,
            binary_mode,
            labels_np,
            fold_tag=f"fold{fold}",
        )

        if save_best:
            metric_value = fold_metrics.get(monitored_metric)
            if metric_value is None:
                if not metric_warning_emitted:
                    warnings.warn(
                        f"Metric '{monitored_metric}' not found in evaluation outputs; "
                        "disabling checkpoint saving for this run."
                    )
                    metric_warning_emitted = True
                    save_best = False
            elif metric_is_better(monitored_metric, metric_value, best_checkpoint_metric):
                best_checkpoint_metric = float(metric_value)
                best_checkpoint_fold = fold
                metrics_snapshot = {key: float(value) for key, value in fold_metrics.items()}
                payload = {
                    "state_dict": to_cpu_state_dict(best_state),
                    "fold": fold,
                    "metric_name": monitored_metric,
                    "metric_value": best_checkpoint_metric,
                    "metrics": metrics_snapshot,
                    "command": getattr(args, "command", None),
                    "report_path": getattr(args, "_kfold_report_path", None),
                }
                torch.save(payload, checkpoint_path)
                print(
                    f"  Saved new best checkpoint to {checkpoint_path} "
                    f"(fold {fold}, {monitored_metric}={format_metric_value(monitored_metric, metric_value)})"
                )

        if getattr(args, "plot_latents", False):
            if args.latent_split == "train":
                target_dataset = train_dataset
                split_tag = f"fold{fold}_train"
            elif args.latent_split == "val":
                target_dataset = val_dataset
                split_tag = f"fold{fold}_val"
            elif args.latent_split == "train_test":
                target_dataset = ConcatDataset([train_dataset, test_dataset])
                split_tag = f"fold{fold}_train-test"
            elif args.latent_split == "all":
                target_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
                split_tag = f"fold{fold}_all"
            else:
                target_dataset = test_dataset
                split_tag = f"fold{fold}_test"
            generate_latent_manifolds(model, target_dataset, args, device, split_tag)

        results.append(fold_metrics)

    report_lines: List[str] = []
    command = getattr(args, "command", None)
    if command:
        report_lines.append("Command:")
        report_lines.append(f"  {command}")
        report_lines.append("")
    report_lines.append(f"Split strategy: {split_strategy}")
    if groups is not None:
        report_lines.append(f"Participant groups: {len(np.unique(groups))}")
    report_lines.append("")
    report_lines.append("Arguments:")
    for key, value in sorted(vars(args).items()):
        if key == "command" or key.startswith("_"):
            continue
        report_lines.append(f"  {key}: {value}")
    report_lines.append("")
    report_lines.append("KFold Test Results (%):")
    print(report_lines[-1])
    keys = ["acc", "pre", "sen", "spe", "f1", "roc", "log_loss", "loss"]
    for idx, m in enumerate(results, start=1):
        parts = []
        for key in keys:
            value = m[key]
            if key in {"loss", "log_loss"}:
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value * 100:.2f}")
        line = f"  Fold {idx}: " + ", ".join(parts)
        print(line)
        report_lines.append(line)

    summary_parts = []
    for key in keys:
        values = np.array([m[key] for m in results], dtype=float)
        sample_sd = values.std(ddof=1) if values.size > 1 else 0.0
        if key in {"loss", "log_loss"}:
            summary_parts.append(f"{key}={values.mean():.4f} ± {sample_sd:.4f}")
        else:
            summary_parts.append(f"{key}={values.mean() * 100:.2f} ± {sample_sd * 100:.2f}")
    summary_line = "  Average ± Std: " + ", ".join(summary_parts)
    print(summary_line)
    report_lines.append(summary_line)

    if report_path != base_report_path:
        print(f"  Existing report found, writing to {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"  Saved KFold summary to {report_path}")

    avg_acc = float(np.mean([m["acc"] for m in results])) if results else float("nan")
    acc_std = float(np.std([m["acc"] for m in results], ddof=1)) if len(results) > 1 else 0.0
    best_root = os.path.join(metrics_dir, "best")
    os.makedirs(best_root, exist_ok=True)
    model_slug = slugify(getattr(args, "model", "model"))
    best_model_root = os.path.join(best_root, model_slug)
    os.makedirs(best_model_root, exist_ok=True)
    best_meta_path = os.path.join(best_model_root, f"{report_stem}.json")
    best_copy_path = os.path.join(best_model_root, f"{report_stem}.txt")

    previous_best_acc: Optional[float] = None
    previous_report_path: Optional[str] = None
    if os.path.exists(best_meta_path):
        try:
            with open(best_meta_path, "r", encoding="utf-8") as meta_file:
                payload = json.load(meta_file)
            previous_best_acc = payload.get("avg_acc")
            previous_report_path = (
                payload.get("report_path")
                or payload.get("copied_report")
                or payload.get("shortcut_report")
                or payload.get("run_report")
            )
        except (OSError, ValueError, TypeError):
            previous_best_acc = None
            previous_report_path = None

    if metric_is_better("acc", avg_acc, previous_best_acc):
        shutil.copy2(report_path, best_copy_path)
        best_payload = {
            "avg_acc": avg_acc,
            "acc_std": acc_std,
            "report_path": report_path,
            "run_dir": run_dir,
            "copied_report": best_copy_path,
            "model_dir": best_model_root,
        }
        with open(best_meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(best_payload, meta_file, indent=2, sort_keys=True)
        previous_text = format_metric_value("acc", previous_best_acc) if previous_best_acc is not None else "n/a"
        print(
            f"  New best average accuracy {format_metric_value('acc', avg_acc)} "
            f"(previous best {previous_text}); saved copies to {best_copy_path}"
        )
    else:
        best_text = format_metric_value("acc", previous_best_acc) if previous_best_acc is not None else "n/a"
        if previous_report_path:
            print(
                f"  Average accuracy {format_metric_value('acc', avg_acc)} "
                f"did not beat best {best_text} stored in {previous_report_path}"
            )
        else:
            print(
                f"  Average accuracy {format_metric_value('acc', avg_acc)} "
                f"did not beat best {best_text}"
            )

    if save_best and best_checkpoint_fold is not None and checkpoint_path:
        print(
            f"  Best checkpoint saved from fold {best_checkpoint_fold} "
            f"({monitored_metric}={format_metric_value(monitored_metric, best_checkpoint_metric)}) "
            f"at {checkpoint_path}"
        )
    return results
