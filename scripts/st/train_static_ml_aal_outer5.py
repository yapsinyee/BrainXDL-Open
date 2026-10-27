#!/usr/bin/env python3
"""Train/evaluate classical static AAL connectivity baselines on outer-five folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from brainxdl.common.metrics import compute_classification_metrics
from brainxdl.common.seed import seed_everything
from brainxdl.st.data_prep import get_outer_fold_indices
from brainxdl.st.preprocessing import (
    build_static_connectivity,
    load_abide_aal_timeseries,
    load_cobre_aal_timeseries,
)


METRIC_COLUMNS = [
    ("acc", "Accuracy"),
    ("roc", "ROC-AUC"),
    ("log_loss", "Log Loss"),
    ("f1", "F1-score"),
    ("sen", "Sensitivity"),
    ("spe", "Specificity"),
    ("pre", "Precision"),
]
POSITIVE_LABEL = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classical static AAL connectivity baselines on five outer folds."
    )
    parser.add_argument("--source", choices=["abide_aal", "cobre_aal"], default="abide_aal")
    parser.add_argument(
        "--rois_dir",
        default="./data/nilearn/ABIDE_pcp/cpac/nofilt_noglobal_balanced",
    )
    parser.add_argument(
        "--phenotypic_csv",
        default="./data/manifests/abide_rois_aal_balanced_phenotypic.csv",
    )
    parser.add_argument(
        "--cobre_subject_column",
        default=None,
        help="COBRE CSV subject/file column. Defaults to the first column.",
    )
    parser.add_argument(
        "--cobre_label_column",
        default="sz",
        help="COBRE CSV diagnosis column, default sz (1=schizophrenia, 0=control).",
    )
    parser.add_argument("--model", choices=["svm"], default="svm")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--threshold_p", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--svm_kernel", choices=["linear", "rbf"], default="rbf")
    parser.add_argument("--svm_gamma", default="scale")
    parser.add_argument("--output_dir", default="./reports/static_aal_svm")
    parser.add_argument(
        "--timestamped_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs into a generation-time subdirectory under --output_dir.",
    )
    return parser.parse_args()


def load_timeseries(args: argparse.Namespace) -> tuple[list[np.ndarray], np.ndarray, Dict[int, str]]:
    if args.source == "cobre_aal":
        timeseries, labels = load_cobre_aal_timeseries(
            rois_dir=args.rois_dir,
            phenotypic_csv=args.phenotypic_csv,
            subject_column=args.cobre_subject_column,
            label_column=args.cobre_label_column,
        )
        return timeseries, labels, {0: "control", 1: "schizophrenia"}

    timeseries, labels = load_abide_aal_timeseries(args.rois_dir, args.phenotypic_csv)
    return timeseries, labels, {0: "control", 1: "autism/ASD positive class"}


def vectorize_connectivity(correlations: np.ndarray) -> np.ndarray:
    """Return upper-triangle connectivity features, excluding the diagonal."""
    if correlations.ndim != 3 or correlations.shape[1] != correlations.shape[2]:
        raise ValueError(f"Expected correlations with shape (n, p, p), got {correlations.shape}")
    tri = np.triu_indices(correlations.shape[1], k=1)
    return correlations[:, tri[0], tri[1]].astype(np.float32)


def build_model(args: argparse.Namespace):
    if args.model == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=args.svm_c,
                kernel=args.svm_kernel,
                gamma=args.svm_gamma,
                class_weight="balanced",
                probability=True,
                random_state=args.seed,
            ),
        )
    raise ValueError(f"Unsupported model: {args.model}")


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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    generated_at = datetime.now().astimezone()
    generated_at_text = generated_at.isoformat(timespec="seconds")

    output_dir = Path(args.output_dir)
    if args.timestamped_dir:
        output_dir = output_dir / generated_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    timeseries, labels_np, label_map = load_timeseries(args)
    correlations, _ = build_static_connectivity(timeseries, args.threshold_p)
    features = vectorize_connectivity(correlations)
    fold_indices = get_outer_fold_indices(labels_np)

    print(f"Subjects: {len(labels_np)}  Features: {features.shape[1]}")
    print(
        f"Label counts: {label_map[0]}={(labels_np == 0).sum()} "
        f"{label_map[1]}={(labels_np == 1).sum()}"
    )
    print(f"Model: {args.model}  threshold_p={args.threshold_p}")

    fold_rows: List[Dict[str, float]] = []
    all_true = []
    all_pred = []
    all_score = []

    for fold in args.folds:
        split = fold_indices[f"outer{fold}_inner1"]
        train_idx = np.asarray(split["train"])
        test_idx = np.asarray(split["test"])
        model = build_model(args)
        model.fit(features[train_idx], labels_np[train_idx])
        y_true = labels_np[test_idx].astype(int)
        y_pred = model.predict(features[test_idx]).astype(int)
        y_score = model.predict_proba(features[test_idx])[:, POSITIVE_LABEL].astype(float)
        metrics = compute_classification_metrics(y_true, y_pred, y_score)
        metrics.update({"fold": fold, "inner": 1, "n_test": int(len(y_true))})
        fold_rows.append(metrics)
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_score.append(y_score)
        print(
            f"Fold {fold}: acc={metrics['acc']:.4f} roc={metrics['roc']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    y_score_all = np.concatenate(all_score)
    aggregate_metrics = compute_classification_metrics(y_true_all, y_pred_all, y_score_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

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

    fold_rows_md = [
        [str(int(row["fold"])), str(int(row["n_test"]))]
        + [format_float(float(row[key])) for key, _ in METRIC_COLUMNS]
        for row in fold_rows
    ]
    summary_rows = []
    for key, title in METRIC_COLUMNS:
        mean = summary[key]["mean"]
        std = summary[key]["std"]
        summary_rows.append(
            [title, format_float(mean), format_float(std), f"{format_float(mean)} +/- {format_float(std)}"]
        )

    report = "\n".join(
        [
            f"# Static AAL {args.model} Five-Fold Results",
            "",
            f"Generated: {generated_at_text}",
            "",
            f"Label convention: 0 = {label_map[0]}, 1 = {label_map[1]}.",
            "",
            "## Per-Fold Metrics",
            markdown_table(["Fold", "N"] + [title for _, title in METRIC_COLUMNS], fold_rows_md),
            "",
            "## Mean +/- Std Across Folds",
            markdown_table(["Metric", "Mean", "Std", "Mean +/- Std"], summary_rows),
            "",
            "## Aggregate Metrics",
            markdown_table(
                ["Metric", "Value"],
                [[title, format_float(float(aggregate_metrics[key]))] for key, title in METRIC_COLUMNS],
            ),
            "",
            "## Aggregate Confusion Matrix",
            markdown_table(
                ["", "Pred 0", "Pred 1"],
                [
                    ["True 0", str(int(cm[0, 0])), str(int(cm[0, 1]))],
                    ["True 1", str(int(cm[1, 0])), str(int(cm[1, 1]))],
                ],
            ),
            "",
        ]
    )

    print()
    print(report)
    write_csv(output_dir / "static_ml_fold_metrics.csv", fold_rows)
    (output_dir / "static_ml_fold_results.md").write_text(report, encoding="utf-8")
    (output_dir / "static_ml_fold_results.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at_text,
                "model": args.model,
                "label_map": label_map,
                "positive_label": POSITIVE_LABEL,
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
    print(f"Saved CSV:  {output_dir / 'static_ml_fold_metrics.csv'}")
    print(f"Saved MD:   {output_dir / 'static_ml_fold_results.md'}")
    print(f"Saved JSON: {output_dir / 'static_ml_fold_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
