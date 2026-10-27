#!/usr/bin/env python
"""Collect ST-BrainXDL outer-fold test metrics into summary tables.

Example:
    python scripts/st/collect_st_fold_results.py \
      --checkpoint_dir ./runs/st_aal_balanced \
      --data_dir ./data/folds_aal_balanced \
      --version aal_balanced_200ep \
      --folds 1 2 3 4 5 \
      --output_dir ./reports/st_aal_balanced
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from brainxdl.common.checkpointing import load_checkpoint
from brainxdl.common.device import pick_device
from brainxdl.common.metrics import choose_binary_threshold, compute_classification_metrics
from brainxdl.st.dataset import STGraphDataset, load_fold_data, padseq
from brainxdl.st.model import STBrainXDL


METRIC_COLUMNS = [
    ("acc", "Accuracy"),
    ("roc", "ROC-AUC"),
    ("log_loss", "Log Loss"),
    ("f1", "F1-score"),
    ("sen", "Sensitivity"),
    ("spe", "Specificity"),
    ("pre", "Precision"),
]
LABEL_MAP = {0: "control", 1: "autism"}
POSITIVE_LABEL = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise five-fold ST-BrainXDL dynamic test metrics."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./runs/st_aal_balanced",
        help="Directory containing ST-BrainXDL fold checkpoints.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data/folds_aal_balanced",
        help="Directory containing graphs_outerX_innerY.pkl fold files.",
    )
    parser.add_argument(
        "--version",
        default="aal_balanced_200ep",
        help="Checkpoint version suffix used in ST-BrainXDL filenames.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Outer folds to collect.",
    )
    parser.add_argument(
        "--inner",
        type=int,
        default=1,
        help="Inner fold index used by the outer-five run.",
    )
    parser.add_argument(
        "--output_dir",
        default="./reports/st_aal_balanced",
        help="Base directory for CSV, JSON, and Markdown summaries.",
    )
    parser.add_argument(
        "--timestamped_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs into a generation-time subdirectory under --output_dir.",
    )
    parser.add_argument(
        "--prefer_mps",
        action="store_true",
        default=False,
        help="Prefer MPS over CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--sparse_on_cpu",
        action="store_true",
        help=(
            "Force graph/PyG modules to CPU during evaluation. By default the "
            "checkpoint training setting is preserved."
        ),
    )
    parser.add_argument(
        "--control_label_name",
        default="control",
        help="Display name for numeric label 0 in the generated report.",
    )
    parser.add_argument(
        "--positive_label_name",
        default="autism/ASD positive class",
        help="Display name for numeric label 1 in the generated report.",
    )
    parser.add_argument(
        "--threshold_objective",
        choices=["balanced_accuracy", "f1", "fixed"],
        default="balanced_accuracy",
        help=(
            "Choose each fold's classification threshold on validation data. "
            "Use fixed to reproduce the legacy 0.5 decision boundary."
        ),
    )
    return parser.parse_args()


def checkpoint_path(checkpoint_dir: str, outer: int, inner: int, version: str) -> Path:
    filename = f"ST-BrainXDL_softmax_adv_fold{outer}{inner}_{version}.pth"
    return Path(checkpoint_dir) / filename


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def predict_graphs(
    model: STBrainXDL,
    setting: dict,
    graphs: np.ndarray,
    kl_weight: float,
    bce_weight: float,
    mce_weight: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return labels, positive-class scores, and mean total loss for one split."""
    dataset = STGraphDataset(graphs)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0,
        collate_fn=padseq,
        pin_memory=False,
    )
    all_targets: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            x_nll, z_kld, a_nll, readout, target = model(batch, setting, sample=False)
            y_bce, _, y_prob, _ = model.classifier(readout, target, sample=True)
            total = x_nll + kl_weight * z_kld + mce_weight * a_nll + bce_weight * y_bce
            probabilities = tensor_to_numpy(y_prob)
            if probabilities.ndim == 2:
                if probabilities.shape[1] < 2:
                    raise ValueError("Expected at least two probability columns.")
                scores = probabilities[:, 1]
            else:
                scores = probabilities
            all_targets.append(tensor_to_numpy(target).astype(int))
            all_scores.append(scores.astype(float))
            losses.append(float(total.item()))
    return (
        np.concatenate(all_targets),
        np.concatenate(all_scores),
        float(np.mean(losses)) if losses else float("nan"),
    )


def evaluate_fold(
    checkpoint: Path,
    data_dir: str,
    outer: int,
    inner: int,
    device: torch.device,
    force_sparse_on_cpu: bool,
    threshold_objective: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    _, test_graphs, val_graphs = load_fold_data(outer, inner, data_dir=data_dir)

    ckpt = load_checkpoint(str(checkpoint), device=device)
    setting = ckpt["training_setting"]
    setting["device"] = str(device)
    if force_sparse_on_cpu:
        setting["sparse_on_cpu"] = True
    sparse_on_cpu = bool(setting.get("sparse_on_cpu", True))
    kl_weight = float(setting.get("kl_weight", 1.0))
    bce_weight = float(setting.get("bce_weight", 1.0))
    mce_weight = float(setting.get("mce_weight", 1.0))

    model = STBrainXDL(setting)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.apply_device_split(device, sparse_on_cpu=sparse_on_cpu)
    model.eval()

    val_true, val_score, _ = predict_graphs(
        model, setting, val_graphs, kl_weight, bce_weight, mce_weight
    )
    if threshold_objective == "fixed":
        threshold = 0.5
        validation_objective_value = float("nan")
    else:
        threshold, validation_objective_value = choose_binary_threshold(
            val_true, val_score, threshold_objective
        )

    y_true, y_score, test_loss = predict_graphs(
        model, setting, test_graphs, kl_weight, bce_weight, mce_weight
    )
    y_pred = (y_score >= threshold).astype(int)

    labels = set(y_true.tolist()) | set(y_pred.tolist())
    if not labels.issubset(set(LABEL_MAP)):
        raise ValueError(
            "This summary script expects binary labels 0/1 for sensitivity, "
            f"specificity, ROC-AUC, and F1. Saw labels: {sorted(labels)}"
        )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_score,
        loss=test_loss,
    )
    metrics["fold"] = outer
    metrics["inner"] = inner
    metrics["n_test"] = int(len(y_true))
    metrics["kl_weight"] = kl_weight
    metrics["bce_weight"] = bce_weight
    metrics["mce_weight"] = mce_weight
    metrics["threshold"] = threshold
    metrics["threshold_objective"] = threshold_objective
    metrics["validation_objective_value"] = validation_objective_value
    metrics["n_validation"] = int(len(val_true))
    metrics.update(compute_fold_diagnostics(y_true, y_pred, y_score))
    return metrics, y_true, y_pred, y_score


def mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def compute_fold_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_counts = {
        str(label): int((y_true == label).sum())
        for label in sorted(LABEL_MAP)
    }
    pred_counts = {
        str(label): int((y_pred == label).sum())
        for label in sorted(LABEL_MAP)
    }
    return {
        "true_counts": true_counts,
        "pred_counts": pred_counts,
        "confusion_matrix": cm.tolist(),
        "mean_asd_prob_true_control": mean_or_nan(y_score[y_true == 0]),
        "mean_asd_prob_true_asd": mean_or_nan(y_score[y_true == 1]),
    }


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.4f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    fieldnames = [
        "fold",
        "inner",
        "n_test",
        "roc",
        "log_loss",
        "f1",
        "sen",
        "spe",
        "pre",
        "acc",
        "loss",
        "true_control",
        "true_asd",
        "pred_control",
        "pred_asd",
        "cm_tn",
        "cm_fp",
        "cm_fn",
        "cm_tp",
        "mean_asd_prob_true_control",
        "mean_asd_prob_true_asd",
        "kl_weight",
        "bce_weight",
        "mce_weight",
        "threshold",
        "threshold_objective",
        "validation_objective_value",
        "n_validation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cm = row["confusion_matrix"]
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fieldnames},
                    "true_control": row["true_counts"]["0"],
                    "true_asd": row["true_counts"]["1"],
                    "pred_control": row["pred_counts"]["0"],
                    "pred_asd": row["pred_counts"]["1"],
                    "cm_tn": cm[0][0],
                    "cm_fp": cm[0][1],
                    "cm_fn": cm[1][0],
                    "cm_tp": cm[1][1],
                }
            )


def main() -> None:
    args = parse_args()
    generated_at = datetime.now().astimezone()
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    generated_at_text = generated_at.isoformat(timespec="seconds")

    output_dir = Path(args.output_dir)
    if args.timestamped_dir:
        output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(prefer_mps=args.prefer_mps)
    print(f"[Device] {device}")

    fold_rows: List[Dict[str, float]] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_score: List[np.ndarray] = []

    for outer in args.folds:
        ckpt_path = checkpoint_path(args.checkpoint_dir, outer, args.inner, args.version)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"[Fold {outer}] Evaluating {ckpt_path}")
        metrics, y_true, y_pred, y_score = evaluate_fold(
            ckpt_path,
            args.data_dir,
            outer,
            args.inner,
            device,
            args.sparse_on_cpu,
            args.threshold_objective,
        )
        fold_rows.append(metrics)
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_score.append(y_score)

    metric_values = {
        key: np.array([float(row[key]) for row in fold_rows], dtype=float)
        for key, _ in METRIC_COLUMNS
    }
    summary = {
        key: {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for key, values in metric_values.items()
    }

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    y_score_all = np.concatenate(all_score)
    aggregate_metrics = compute_classification_metrics(y_true_all, y_pred_all, y_score_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

    fold_headers = ["Fold", "N"] + [title for _, title in METRIC_COLUMNS]
    fold_table_rows = []
    for row in fold_rows:
        fold_table_rows.append(
            [str(int(row["fold"])), str(int(row["n_test"]))]
            + [format_float(float(row[key])) for key, _ in METRIC_COLUMNS]
        )

    summary_headers = ["Metric", "Mean", "Std", "Mean +/- Std"]
    summary_table_rows = []
    for key, title in METRIC_COLUMNS:
        mean = summary[key]["mean"]
        std = summary[key]["std"]
        summary_table_rows.append(
            [title, format_float(mean), format_float(std), f"{format_float(mean)} +/- {format_float(std)}"]
        )

    cm_headers = ["", "Pred 0", "Pred 1"]
    cm_rows = [
        ["True 0", str(int(cm[0, 0])), str(int(cm[0, 1]))],
        ["True 1", str(int(cm[1, 0])), str(int(cm[1, 1]))],
    ]
    count_headers = [
        "Fold",
        f"True {args.control_label_name}",
        f"True {args.positive_label_name}",
        f"Pred {args.control_label_name}",
        f"Pred {args.positive_label_name}",
    ]
    count_rows = [
        [
            str(int(row["fold"])),
            str(row["true_counts"]["0"]),
            str(row["true_counts"]["1"]),
            str(row["pred_counts"]["0"]),
            str(row["pred_counts"]["1"]),
        ]
        for row in fold_rows
    ]
    prob_headers = [
        "Fold",
        f"Mean Positive-Class Prob True {args.control_label_name}",
        f"Mean Positive-Class Prob True {args.positive_label_name}",
    ]
    prob_rows = [
        [
            str(int(row["fold"])),
            format_float(float(row["mean_asd_prob_true_control"])),
            format_float(float(row["mean_asd_prob_true_asd"])),
        ]
        for row in fold_rows
    ]
    per_fold_cm_headers = ["Fold", "TN", "FP", "FN", "TP"]
    per_fold_cm_rows = []
    for row in fold_rows:
        fold_cm = row["confusion_matrix"]
        per_fold_cm_rows.append(
            [
                str(int(row["fold"])),
                str(int(fold_cm[0][0])),
                str(int(fold_cm[0][1])),
                str(int(fold_cm[1][0])),
                str(int(fold_cm[1][1])),
            ]
        )
    weight_headers = ["Fold", "KL Weight", "BCE Weight", "MCE Weight"]
    weight_rows = [
        [
            str(int(row["fold"])),
            format_float(float(row["kl_weight"])),
            format_float(float(row["bce_weight"])),
            format_float(float(row["mce_weight"])),
        ]
        for row in fold_rows
    ]
    threshold_headers = [
        "Fold",
        "Validation N",
        "Objective",
        "Selected Threshold",
        "Validation Objective Value",
    ]
    threshold_rows = [
        [
            str(int(row["fold"])),
            str(int(row["n_validation"])),
            str(row["threshold_objective"]),
            format_float(float(row["threshold"])),
            format_float(float(row["validation_objective_value"])),
        ]
        for row in fold_rows
    ]

    report_parts = [
        "# ST-BrainXDL Five-Fold Dynamic Results",
        "",
        f"Generated: {generated_at_text}",
        "",
        (
            f"Label convention: 0 = {args.control_label_name}, "
            f"{POSITIVE_LABEL} = {args.positive_label_name}."
        ),
        "",
        (
            "Classification thresholds were selected independently for each fold "
            f"using validation {args.threshold_objective.replace('_', ' ')} only; "
            "test labels were not used for threshold selection."
            if args.threshold_objective != "fixed"
            else "Classification used the fixed legacy threshold of 0.5."
        ),
        "",
        "## Loss Weights",
        markdown_table(weight_headers, weight_rows),
        "",
        "## Validation-Selected Classification Thresholds",
        markdown_table(threshold_headers, threshold_rows),
        "",
        "## Per-Fold Metrics",
        markdown_table(fold_headers, fold_table_rows),
        "",
        "## Mean +/- Std Across Folds",
        markdown_table(summary_headers, summary_table_rows),
        "",
        "## Aggregate Metrics",
        markdown_table(
            ["Metric", "Value"],
            [[title, format_float(float(aggregate_metrics[key]))] for key, title in METRIC_COLUMNS],
        ),
        "",
        "## Per-Fold Label Counts",
        markdown_table(count_headers, count_rows),
        "",
        "## Per-Fold Confusion Matrices",
        markdown_table(per_fold_cm_headers, per_fold_cm_rows),
        "",
        "## Per-Fold Positive-Class Probability Diagnostics",
        markdown_table(prob_headers, prob_rows),
        "",
        "## Aggregate Confusion Matrix",
        markdown_table(cm_headers, cm_rows),
        "",
    ]
    report = "\n".join(report_parts)

    print()
    print(report)

    write_csv(output_dir / "st_fold_metrics.csv", fold_rows)
    (output_dir / "st_fold_results.md").write_text(report, encoding="utf-8")
    (output_dir / "st_fold_results.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at_text,
                "label_map": {
                    0: args.control_label_name,
                    POSITIVE_LABEL: args.positive_label_name,
                },
                "positive_label": POSITIVE_LABEL,
                "threshold_selection": {
                    "split": "validation",
                    "objective": args.threshold_objective,
                    "test_labels_used": False,
                },
                "folds": fold_rows,
                "mean_std": summary,
                "aggregate_metrics": aggregate_metrics,
                "confusion_matrix_labels": [0, 1],
                "confusion_matrix": cm.tolist(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Saved CSV:  {output_dir / 'st_fold_metrics.csv'}")
    print(f"Saved MD:   {output_dir / 'st_fold_results.md'}")
    print(f"Saved JSON: {output_dir / 'st_fold_results.json'}")


if __name__ == "__main__":
    main()
