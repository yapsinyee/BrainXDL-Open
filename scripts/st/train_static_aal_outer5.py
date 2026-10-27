#!/usr/bin/env python3
"""Train/evaluate a static fMRI baseline on matched outer-five splits.

It uses full-series Ledoit-Wolf connectivity per observation. ABIDE and COBRE
retain the established outer-fold convention; PsiConnect uses participant-
grouped outer folds so both sessions from one participant stay together.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from brainxdl.common.device import pick_device
from brainxdl.common.metrics import compute_classification_metrics
from brainxdl.common.seed import seed_everything
from brainxdl.mm.models.baselines import FmriGCN
from brainxdl.st.data_prep import get_outer_fold_indices
from brainxdl.st.psiconnect import (
    DEFAULT_PATTERN as PSICONNECT_DEFAULT_PATTERN,
    get_grouped_outer_fold_indices,
    load_psiconnect_timeseries,
)
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
        description="Train a static fMRI neural baseline on five outer folds."
    )
    parser.add_argument(
        "--rois_dir",
        default="./data/nilearn/ABIDE_pcp/cpac/nofilt_noglobal_balanced",
    )
    parser.add_argument(
        "--phenotypic_csv",
        default="./data/manifests/abide_rois_aal_balanced_phenotypic.csv",
    )
    parser.add_argument(
        "--source",
        choices=["abide_aal", "cobre_aal", "psiconnect"],
        default="abide_aal",
        help="Dataset source and label convention.",
    )
    parser.add_argument(
        "--psiconnect_timeseries_root",
        default=None,
        help="PsiConnect Schaefer time-series root.",
    )
    parser.add_argument(
        "--psiconnect_matched_eeg_root",
        default=None,
        help="Restrict PsiConnect fMRI to the matched BrainXDL cohort.",
    )
    parser.add_argument(
        "--psiconnect_pattern",
        default=PSICONNECT_DEFAULT_PATTERN,
        help="Glob pattern below the PsiConnect time-series root.",
    )
    parser.add_argument("--psiconnect_expected_rois", type=int, default=200)
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
    parser.add_argument(
        "--model",
        choices=["gcn"],
        default="gcn",
        help="Static baseline architecture to train.",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--threshold_p", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default="./runs/static_aal_balanced_outer5")
    parser.add_argument("--output_dir", default="./reports/static_aal_balanced")
    parser.add_argument(
        "--timestamped_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs into a generation-time subdirectory under --output_dir.",
    )
    return parser.parse_args()


def make_loader(
    correlations: torch.Tensor,
    adjacencies: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    idx = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(correlations[idx], adjacencies[idx], labels[idx])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total = 0
    probs_all = []
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for correlations, adjacencies, labels in loader:
            correlations = correlations.to(device)
            adjacencies = adjacencies.to(device)
            labels = labels.to(device)
            logits = forward_static_model(model, correlations, adjacencies)
            loss = F.cross_entropy(logits, labels)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            probs_all.append(probs.detach().cpu())
            preds_all.append(preds.detach().cpu())
            labels_all.append(labels.detach().cpu())

    return {
        "loss": total_loss / max(total, 1),
        "probs": torch.cat(probs_all) if probs_all else torch.empty(0, 2),
        "preds": torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.long),
        "labels": torch.cat(labels_all) if labels_all else torch.empty(0, dtype=torch.long),
    }


def build_static_model(args: argparse.Namespace, num_nodes: int) -> torch.nn.Module:
    if args.model == "gcn":
        return FmriGCN(
            num_nodes=num_nodes,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            out_dim=2,
            k=args.k,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported static model: {args.model}")


def forward_static_model(
    model: torch.nn.Module,
    correlations: torch.Tensor,
    adjacencies: torch.Tensor,
) -> torch.Tensor:
    return model(correlations, adjacencies)


def train_one_fold(
    args: argparse.Namespace,
    fold: int,
    fold_indices: Dict[str, Dict[str, np.ndarray]],
    correlations: torch.Tensor,
    adjacencies: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    key = f"outer{fold}_inner1"
    split = fold_indices[key]
    train_idx = np.asarray(split["train"])
    val_idx = np.asarray(split["val"])
    test_idx = np.asarray(split["test"])

    train_loader = make_loader(correlations, adjacencies, labels, train_idx, args.batch_size, True)
    val_loader = make_loader(correlations, adjacencies, labels, val_idx, args.batch_size, False)
    test_loader = make_loader(correlations, adjacencies, labels, test_idx, args.batch_size, False)

    model = build_static_model(args, num_nodes=correlations.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    print(
        f"\nFold {fold}/5 | train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for corr_b, adj_b, labels_b in train_loader:
            corr_b = corr_b.to(device)
            adj_b = adj_b.to(device)
            labels_b = labels_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_static_model(model, corr_b, adj_b)
            loss = F.cross_entropy(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        val_eval = evaluate(model, val_loader, device)
        val_labels = val_eval["labels"].numpy()
        val_preds = val_eval["preds"].numpy()
        val_acc = float((val_labels == val_preds).mean()) if len(val_labels) else 0.0
        val_loss = float(val_eval["loss"])
        is_better = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"Fold {fold} epoch {epoch:03d} | "
                f"val loss={val_loss:.4f} acc={val_acc:.4f}"
            )

        if bad_epochs >= args.patience:
            print(f"Fold {fold} early stopped at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_eval = evaluate(model, test_loader, device)
    y_true = test_eval["labels"].numpy().astype(int)
    y_pred = test_eval["preds"].numpy().astype(int)
    y_score = test_eval["probs"].numpy()[:, POSITIVE_LABEL].astype(float)
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_score,
        loss=float(test_eval["loss"]),
    )
    metrics.update(
        {
            "fold": fold,
            "inner": 1,
            "n_test": int(len(y_true)),
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val_acc),
            "best_val_loss": float(best_val_loss),
        }
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir
        / f"{args.model}_static_{args.source}_outer{fold}_inner1_seed{args.seed}.pth"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "test_metrics": {
                "loss": test_eval["loss"],
                "labels": test_eval["labels"],
                "preds": test_eval["preds"],
                "probs": test_eval["probs"],
            },
            "splits": {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            },
        },
        checkpoint_path,
    )
    print(
        f"Fold {fold} test: roc={metrics['roc']:.4f} f1={metrics['f1']:.4f} "
        f"sen={metrics['sen']:.4f} spe={metrics['spe']:.4f} pre={metrics['pre']:.4f}"
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred, "y_score": y_score}


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
        "best_epoch",
        "best_val_acc",
        "best_val_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_report(
    output_dir: Path,
    generated_at_text: str,
    model_name: str,
    label_map: Dict[int, str],
    fold_rows: List[Dict[str, float]],
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    y_score_all: np.ndarray,
) -> None:
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
    aggregate_metrics = compute_classification_metrics(y_true_all, y_pred_all, y_score_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

    fold_headers = ["Fold", "N"] + [title for _, title in METRIC_COLUMNS]
    fold_table_rows = [
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
            f"# Static {model_name} Five-Fold Results",
            "",
            f"Generated: {generated_at_text}",
            "",
            f"Label convention: 0 = {label_map[0]}, 1 = {label_map[1]}.",
            "",
            "## Per-Fold Metrics",
            markdown_table(fold_headers, fold_table_rows),
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
    write_csv(output_dir / "static_fold_metrics.csv", fold_rows)
    (output_dir / "static_fold_results.md").write_text(report, encoding="utf-8")
    (output_dir / "static_fold_results.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at_text,
                "model": model_name,
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
    print(f"Saved CSV:  {output_dir / 'static_fold_metrics.csv'}")
    print(f"Saved MD:   {output_dir / 'static_fold_results.md'}")
    print(f"Saved JSON: {output_dir / 'static_fold_results.json'}")


def load_source_dataset(
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], np.ndarray, Dict[int, str], Dict[str, Dict[str, np.ndarray]]]:
    """Load one supported cohort and return its leakage-safe outer folds."""
    if args.source == "cobre_aal":
        timeseries, labels_np = load_cobre_aal_timeseries(
            rois_dir=args.rois_dir,
            phenotypic_csv=args.phenotypic_csv,
            subject_column=args.cobre_subject_column,
            label_column=args.cobre_label_column,
        )
        label_map = {0: "control", 1: "schizophrenia"}
        fold_indices = get_outer_fold_indices(labels_np)
    elif args.source == "psiconnect":
        timeseries, labels_np, groups_np, _ = load_psiconnect_timeseries(
            args.psiconnect_timeseries_root,
            pattern=args.psiconnect_pattern,
            expected_rois=args.psiconnect_expected_rois,
            matched_eeg_root=args.psiconnect_matched_eeg_root,
        )
        label_map = {0: "control", 1: "psilocybin"}
        fold_indices = get_grouped_outer_fold_indices(
            labels_np,
            groups_np,
            n_splits=5,
            seed=args.seed,
        )
    else:
        timeseries, labels_np = load_abide_aal_timeseries(args.rois_dir, args.phenotypic_csv)
        label_map = {0: "control", 1: "autism/ASD positive class"}
        fold_indices = get_outer_fold_indices(labels_np)
    return timeseries, labels_np, label_map, fold_indices


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device()
    print(f"[Device] {device}")

    generated_at = datetime.now().astimezone()
    generated_at_text = generated_at.isoformat(timespec="seconds")
    output_dir = Path(args.output_dir)
    if args.timestamped_dir:
        output_dir = output_dir / generated_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    timeseries, labels_np, label_map, fold_indices = load_source_dataset(args)

    corr_np, adjacency_np = build_static_connectivity(timeseries, args.threshold_p)
    labels = torch.as_tensor(labels_np, dtype=torch.long)
    correlations = torch.as_tensor(corr_np, dtype=torch.float32)
    adjacencies = torch.as_tensor(adjacency_np, dtype=torch.float32)
    print(f"Subjects: {len(labels_np)}  ROIs: {correlations.shape[-1]}")
    print(
        f"Label counts: {label_map[0]}={(labels == 0).sum().item()} "
        f"{label_map[1]}={(labels == 1).sum().item()}"
    )
    print(f"Static connectivity: full-series LDW correlation, threshold_p={args.threshold_p}")

    fold_rows = []
    all_true = []
    all_pred = []
    all_score = []
    for fold in args.folds:
        result = train_one_fold(
            args,
            fold,
            fold_indices,
            correlations,
            adjacencies,
            labels,
            device,
        )
        fold_rows.append(result["metrics"])
        all_true.append(result["y_true"])
        all_pred.append(result["y_pred"])
        all_score.append(result["y_score"])

    write_report(
        output_dir,
        generated_at_text,
        args.model,
        label_map,
        fold_rows,
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_score),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
