#!/usr/bin/env python3
"""Train/evaluate an LSTM baseline on dynamic FC window sequences."""

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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from brainxdl.common.device import pick_device
from brainxdl.common.metrics import compute_classification_metrics
from brainxdl.common.seed import seed_everything
from brainxdl.st.dataset import load_fold_data


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


class DynamicFCLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        head_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Dropout(dropout),
            nn.Linear(head_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last = h_n[-1]
        return self.classifier(last)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM baseline on dynamic FC graph-window sequences."
    )
    parser.add_argument("--data_dir", default="./data/folds_cobre_aal")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--label_positive_name", default="positive class")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default="./runs/dynamic_fc_lstm_cobre_aal")
    parser.add_argument("--output_dir", default="./reports/dynamic_fc_lstm_cobre_aal")
    parser.add_argument(
        "--timestamped_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs into a generation-time subdirectory under --output_dir.",
    )
    return parser.parse_args()


def sequence_length(graph_seq) -> int:
    length = 0
    for graph in graph_seq:
        if bool(getattr(graph, "pad", False)):
            break
        length += 1
        if bool(getattr(graph, "last", False)):
            break
    return max(length, 1)


def graph_sequence_to_features(graph_seq) -> tuple[np.ndarray, int, int]:
    num_nodes = int(graph_seq[0].num_nodes)
    tri = torch.triu_indices(num_nodes, num_nodes, offset=1)
    length = sequence_length(graph_seq)
    features = []
    for graph in graph_seq:
        corr = graph.x.detach().cpu()
        features.append(corr[tri[0], tri[1]].numpy().astype(np.float32))
    return np.stack(features), length, int(graph_seq[0].y.item())


def split_to_tensors(graphs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_rows = []
    lengths = []
    labels = []
    for graph_seq in graphs:
        seq_features, seq_len, label = graph_sequence_to_features(graph_seq)
        feature_rows.append(seq_features)
        lengths.append(seq_len)
        labels.append(label)
    return (
        torch.as_tensor(np.stack(feature_rows), dtype=torch.float32),
        torch.as_tensor(lengths, dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.long),
    )


def make_loader(
    x: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(x, lengths, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(
    model: DynamicFCLSTM,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total = 0
    probs_all = []
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for x_b, len_b, labels_b in loader:
            x_b = x_b.to(device)
            len_b = len_b.to(device)
            labels_b = labels_b.to(device)
            logits = model(x_b, len_b)
            loss = F.cross_entropy(logits, labels_b)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            batch_size = labels_b.size(0)
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            probs_all.append(probs.detach().cpu())
            preds_all.append(preds.detach().cpu())
            labels_all.append(labels_b.detach().cpu())

    return {
        "loss": total_loss / max(total, 1),
        "probs": torch.cat(probs_all) if probs_all else torch.empty(0, 2),
        "preds": torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.long),
        "labels": torch.cat(labels_all) if labels_all else torch.empty(0, dtype=torch.long),
    }


def train_one_fold(
    args: argparse.Namespace,
    fold: int,
    device: torch.device,
) -> Dict[str, object]:
    train_graphs, test_graphs, val_graphs = load_fold_data(fold, args.inner, args.data_dir)
    x_train, len_train, y_train = split_to_tensors(train_graphs)
    x_val, len_val, y_val = split_to_tensors(val_graphs)
    x_test, len_test, y_test = split_to_tensors(test_graphs)

    train_loader = make_loader(x_train, len_train, y_train, args.batch_size, True)
    val_loader = make_loader(x_val, len_val, y_val, args.batch_size, False)
    test_loader = make_loader(x_test, len_test, y_test, args.batch_size, False)

    model = DynamicFCLSTM(
        input_dim=x_train.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        out_dim=2,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = 0
    bad_epochs = 0

    print(
        f"\nFold {fold}/5 | train={len(y_train)} val={len(y_val)} test={len(y_test)} "
        f"windows={x_train.shape[1]} features={x_train.shape[2]}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x_b, len_b, labels_b in train_loader:
            x_b = x_b.to(device)
            len_b = len_b.to(device)
            labels_b = labels_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_b, len_b)
            loss = F.cross_entropy(logits, labels_b)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        val_eval = evaluate(model, val_loader, device)
        val_labels = val_eval["labels"].numpy()
        val_preds = val_eval["preds"].numpy()
        val_acc = float((val_labels == val_preds).mean()) if len(val_labels) else 0.0
        val_loss = float(val_eval["loss"])
        is_better = val_loss < best_val_loss or (
            val_loss == best_val_loss and val_acc > best_val_acc
        )
        if is_better:
            best_val_loss = val_loss
            best_val_acc = val_acc
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
            "inner": args.inner,
            "n_test": int(len(y_true)),
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val_acc),
            "best_val_loss": float(best_val_loss),
        }
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"dynamic_fc_lstm_outer{fold}_inner{args.inner}_seed{args.seed}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    print(
        f"Fold {fold} test: acc={metrics['acc']:.4f} roc={metrics['roc']:.4f} "
        f"f1={metrics['f1']:.4f}"
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
    label_positive_name: str,
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
            "# Dynamic FC LSTM Five-Fold Results",
            "",
            f"Generated: {generated_at_text}",
            "",
            f"Label convention: 0 = control, 1 = {label_positive_name}.",
            "",
            "## Per-Fold Metrics",
            markdown_table(["Fold", "N"] + [title for _, title in METRIC_COLUMNS], fold_table_rows),
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
    write_csv(output_dir / "dynamic_fc_lstm_fold_metrics.csv", fold_rows)
    (output_dir / "dynamic_fc_lstm_fold_results.md").write_text(report, encoding="utf-8")
    (output_dir / "dynamic_fc_lstm_fold_results.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at_text,
                "model": "dynamic_fc_lstm",
                "label_map": {0: "control", POSITIVE_LABEL: label_positive_name},
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
    print(f"Saved CSV:  {output_dir / 'dynamic_fc_lstm_fold_metrics.csv'}")
    print(f"Saved MD:   {output_dir / 'dynamic_fc_lstm_fold_results.md'}")
    print(f"Saved JSON: {output_dir / 'dynamic_fc_lstm_fold_results.json'}")


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

    fold_rows = []
    all_true = []
    all_pred = []
    all_score = []
    for fold in args.folds:
        result = train_one_fold(args, fold, device)
        fold_rows.append(result["metrics"])
        all_true.append(result["y_true"])
        all_pred.append(result["y_pred"])
        all_score.append(result["y_score"])

    write_report(
        output_dir,
        generated_at_text,
        args.label_positive_name,
        fold_rows,
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_score),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
