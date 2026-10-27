#!/usr/bin/env python3
"""Train/evaluate ML baselines on averaged dynamic LDW connectivity features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import dill
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from brainxdl.common.metrics import compute_classification_metrics
from brainxdl.common.seed import seed_everything
from brainxdl.st.data_prep import get_outer_fold_indices


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
        description=(
            "Train classical ML baselines on subject-level averaged dynamic LDW "
            "correlation features using five outer folds."
        )
    )
    parser.add_argument(
        "--ldw_dir",
        default="./data/ldw_aal_balanced_w40_s20_p0p50",
        help="Directory containing LDW_abide_data.pkl.",
    )
    parser.add_argument(
        "--label_positive_name",
        default="autism/ASD positive class",
        help="Name to display for label 1.",
    )
    parser.add_argument(
        "--model",
        choices=["svm", "logreg"],
        default="svm",
    )
    parser.add_argument(
        "--feature_mode",
        choices=["mean_corr", "length_only"],
        default="mean_corr",
        help=(
            "Feature construction mode. mean_corr preserves the existing dynamic "
            "baseline; length_only uses only each subject's number of LDW windows."
        ),
    )
    parser.add_argument(
        "--min_windows",
        type=int,
        default=1,
        help="Drop subjects with fewer than this many LDW windows before building folds.",
    )
    parser.add_argument(
        "--fixed_windows",
        type=int,
        default=0,
        help=(
            "When >0, average exactly this many windows per subject after applying "
            "--min_windows. Requires every retained subject to have at least this many windows."
        ),
    )
    parser.add_argument(
        "--fixed_window_strategy",
        choices=["first", "last", "center", "even"],
        default="first",
        help="Which windows to use when --fixed_windows is set.",
    )
    parser.add_argument(
        "--residualize_n_windows",
        action="store_true",
        help=(
            "For mean_corr features, regress every feature on n_windows within each "
            "training fold and apply the train-fold residualization to test rows."
        ),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--svm_kernel", choices=["linear", "rbf"], default="rbf")
    parser.add_argument("--svm_gamma", default="scale")
    parser.add_argument("--logreg_c", type=float, default=1.0)
    parser.add_argument("--logreg_max_iter", type=int, default=5000)
    parser.add_argument(
        "--output_dir",
        default="./reports/dynamic_ldw_aal_ml",
    )
    parser.add_argument(
        "--timestamped_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs into a generation-time subdirectory under --output_dir.",
    )
    return parser.parse_args()


def select_fixed_windows(n_windows: int, fixed_windows: int, strategy: str) -> np.ndarray:
    if fixed_windows <= 0:
        return np.arange(n_windows)
    if n_windows < fixed_windows:
        raise ValueError(
            f"Cannot select {fixed_windows} windows from a subject with {n_windows} windows."
        )
    if strategy == "first":
        return np.arange(fixed_windows)
    if strategy == "last":
        return np.arange(n_windows - fixed_windows, n_windows)
    if strategy == "center":
        start = (n_windows - fixed_windows) // 2
        return np.arange(start, start + fixed_windows)
    if strategy == "even":
        return np.unique(np.linspace(0, n_windows - 1, fixed_windows).round().astype(int))
    raise ValueError(f"Unsupported fixed-window strategy: {strategy}")


def load_ldw_features(
    ldw_dir: str,
    *,
    feature_mode: str,
    min_windows: int,
    fixed_windows: int,
    fixed_window_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    ldw_path = Path(ldw_dir) / "LDW_abide_data.pkl"
    if not ldw_path.exists():
        raise FileNotFoundError(f"Missing LDW data: {ldw_path}")

    with ldw_path.open("rb") as handle:
        payload = dill.load(handle)

    node_feats = payload["node_feat"]
    labels = np.asarray(payload["labels"], dtype=int)
    if len(node_feats) != len(labels):
        raise ValueError(f"node_feat/labels length mismatch: {len(node_feats)} vs {len(labels)}")

    if min_windows < 1:
        raise ValueError("--min_windows must be >= 1")
    if fixed_windows < 0:
        raise ValueError("--fixed_windows must be >= 0")
    if fixed_windows and min_windows < fixed_windows:
        raise ValueError("--min_windows must be >= --fixed_windows when fixed windows are used")

    subject_corrs = []
    retained_labels = []
    retained_subject_indices = []
    n_windows = []
    for subject_idx, corr_seq in enumerate(node_feats):
        if len(corr_seq) == 0:
            raise ValueError(f"Subject {subject_idx} has no windows.")
        corr_arr = np.asarray(corr_seq, dtype=np.float32)
        if corr_arr.ndim != 3 or corr_arr.shape[1] != corr_arr.shape[2]:
            raise ValueError(
                f"Subject {subject_idx} expected windows x nodes x nodes, got {corr_arr.shape}"
            )
        subject_n_windows = int(corr_arr.shape[0])
        if subject_n_windows < min_windows:
            continue
        selected = select_fixed_windows(
            subject_n_windows,
            fixed_windows=fixed_windows,
            strategy=fixed_window_strategy,
        )
        if fixed_windows and len(selected) != fixed_windows:
            raise ValueError(
                f"Fixed-window selection returned {len(selected)} windows, expected {fixed_windows}."
            )
        subject_corrs.append(np.mean(corr_arr[selected], axis=0))
        retained_labels.append(int(labels[subject_idx]))
        retained_subject_indices.append(int(subject_idx))
        n_windows.append(subject_n_windows)

    if not retained_labels:
        raise ValueError("No subjects retained after applying --min_windows.")

    labels = np.asarray(retained_labels, dtype=int)
    n_windows_arr = np.asarray(n_windows, dtype=np.float32)
    if feature_mode == "length_only":
        features = n_windows_arr.reshape(-1, 1)
        n_nodes = int(np.asarray(node_feats[retained_subject_indices[0]][0]).shape[0])
    elif feature_mode == "mean_corr":
        correlations = np.stack(subject_corrs, axis=0)
        tri = np.triu_indices(correlations.shape[1], k=1)
        features = correlations[:, tri[0], tri[1]].astype(np.float32)
        n_nodes = int(correlations.shape[1])
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    metadata = {
        "n_subjects": int(len(labels)),
        "n_subjects_original": int(len(node_feats)),
        "n_subjects_dropped_min_windows": int(len(node_feats) - len(labels)),
        "retained_subject_indices": retained_subject_indices,
        "feature_mode": feature_mode,
        "min_windows": int(min_windows),
        "fixed_windows": int(fixed_windows),
        "fixed_window_strategy": fixed_window_strategy if fixed_windows else "",
        "n_nodes": n_nodes,
        "n_features": int(features.shape[1]),
        "n_windows_min": int(np.min(n_windows_arr)),
        "n_windows_max": int(np.max(n_windows_arr)),
        "n_windows_mean": float(np.mean(n_windows_arr)),
    }
    return features, labels, n_windows_arr, metadata


def residualize_against_n_windows(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_windows_train: np.ndarray,
    n_windows_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design_train = np.column_stack(
        [np.ones(len(n_windows_train), dtype=np.float32), n_windows_train.astype(np.float32)]
    )
    design_test = np.column_stack(
        [np.ones(len(n_windows_test), dtype=np.float32), n_windows_test.astype(np.float32)]
    )
    beta, *_ = np.linalg.lstsq(design_train, x_train, rcond=None)
    train_residual = x_train - design_train @ beta
    test_residual = x_test - design_test @ beta
    return train_residual.astype(np.float32), test_residual.astype(np.float32)


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
    if args.model == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=args.logreg_c,
                class_weight="balanced",
                max_iter=args.logreg_max_iter,
                random_state=args.seed,
                solver="lbfgs",
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
    if args.feature_mode == "length_only" and args.residualize_n_windows:
        raise ValueError("--residualize_n_windows is not meaningful with --feature_mode length_only")
    generated_at = datetime.now().astimezone()
    generated_at_text = generated_at.isoformat(timespec="seconds")

    output_dir = Path(args.output_dir)
    if args.timestamped_dir:
        output_dir = output_dir / generated_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels_np, n_windows_np, feature_metadata = load_ldw_features(
        args.ldw_dir,
        feature_mode=args.feature_mode,
        min_windows=args.min_windows,
        fixed_windows=args.fixed_windows,
        fixed_window_strategy=args.fixed_window_strategy,
    )
    feature_metadata["residualize_n_windows"] = bool(args.residualize_n_windows)
    fold_indices = get_outer_fold_indices(labels_np)
    label_map = {0: "control", 1: args.label_positive_name}

    print(f"LDW dir: {args.ldw_dir}")
    print(f"Subjects: {len(labels_np)}  Features: {features.shape[1]}")
    print(
        f"Label counts: {label_map[0]}={(labels_np == 0).sum()} "
        f"{label_map[1]}={(labels_np == 1).sum()}"
    )
    print(f"Model: {args.model}")
    print(f"Feature mode: {args.feature_mode}")

    fold_rows: List[Dict[str, float]] = []
    all_true = []
    all_pred = []
    all_score = []

    for fold in args.folds:
        split = fold_indices[f"outer{fold}_inner1"]
        train_idx = np.asarray(split["train"])
        test_idx = np.asarray(split["test"])
        x_train = features[train_idx]
        x_test = features[test_idx]
        if args.residualize_n_windows:
            x_train, x_test = residualize_against_n_windows(
                x_train,
                x_test,
                n_windows_np[train_idx],
                n_windows_np[test_idx],
            )
        model = build_model(args)
        model.fit(x_train, labels_np[train_idx])
        y_true = labels_np[test_idx].astype(int)
        y_pred = model.predict(x_test).astype(int)
        y_score = model.predict_proba(x_test)[:, POSITIVE_LABEL].astype(float)
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

    feature_rows = [
        ["LDW directory", f"`{args.ldw_dir}`"],
        ["Feature mode", args.feature_mode],
        ["Residualize n_windows", str(bool(args.residualize_n_windows))],
        ["Minimum windows", str(feature_metadata["min_windows"])],
        ["Fixed windows", str(feature_metadata["fixed_windows"])],
        ["Fixed-window strategy", str(feature_metadata["fixed_window_strategy"])],
        ["Subjects", str(feature_metadata["n_subjects"])],
        ["Original subjects", str(feature_metadata["n_subjects_original"])],
        ["Dropped by min windows", str(feature_metadata["n_subjects_dropped_min_windows"])],
        ["Nodes", str(feature_metadata["n_nodes"])],
        ["Features", str(feature_metadata["n_features"])],
        ["Min windows", str(feature_metadata["n_windows_min"])],
        ["Max windows", str(feature_metadata["n_windows_max"])],
        ["Mean windows", format_float(float(feature_metadata["n_windows_mean"]))],
    ]

    report = "\n".join(
        [
            f"# Dynamic LDW {args.model} Five-Fold Results",
            "",
            f"Generated: {generated_at_text}",
            "",
            f"Label convention: 0 = {label_map[0]}, 1 = {label_map[1]}.",
            "",
            (
                "Feature construction: length-only n_windows baseline."
                if args.feature_mode == "length_only"
                else "Feature construction: average selected dynamic LDW correlation windows, then flatten the upper triangle."
            ),
            (
                "Fold-wise control: each feature was residualized against n_windows using the training fold only."
                if args.residualize_n_windows
                else "Fold-wise control: none."
            ),
            "",
            "## Feature Metadata",
            markdown_table(["Field", "Value"], feature_rows),
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
    write_csv(output_dir / "dynamic_ldw_ml_fold_metrics.csv", fold_rows)
    (output_dir / "dynamic_ldw_ml_fold_results.md").write_text(report, encoding="utf-8")
    (output_dir / "dynamic_ldw_ml_fold_results.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at_text,
                "model": args.model,
                "ldw_dir": args.ldw_dir,
                "feature_construction": args.feature_mode,
                "feature_metadata": feature_metadata,
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
    print(f"Saved CSV:  {output_dir / 'dynamic_ldw_ml_fold_metrics.csv'}")
    print(f"Saved MD:   {output_dir / 'dynamic_ldw_ml_fold_results.md'}")
    print(f"Saved JSON: {output_dir / 'dynamic_ldw_ml_fold_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
