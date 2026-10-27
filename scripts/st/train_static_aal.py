#!/usr/bin/env python3
"""Train a single-split static AAL fMRI connectivity reference baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from brainxdl.common.device import pick_device
from brainxdl.common.seed import seed_everything
from brainxdl.mm.models.baselines import FmriGCN
from brainxdl.st.preprocessing import (
    build_static_connectivity,
    load_abide_aal_timeseries,
    load_cobre_aal_timeseries,
)


def make_splits(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.25,
        random_state=seed,
        stratify=labels[trainval_idx],
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


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


def evaluate(model: FmriGCN, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    probs_all = []
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for correlations, adjacencies, labels in loader:
            correlations = correlations.to(device)
            adjacencies = adjacencies.to(device)
            labels = labels.to(device)
            logits = model(correlations, adjacencies)
            loss = F.cross_entropy(logits, labels)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            correct += (preds == labels).sum().item()
            probs_all.append(probs.cpu())
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "probs": torch.cat(probs_all) if probs_all else torch.empty(0, 2),
        "preds": torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.long),
        "labels": torch.cat(labels_all) if labels_all else torch.empty(0, dtype=torch.long),
    }


def train(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    device = pick_device()
    print(f"[Device] {device}")

    if args.source == "cobre_aal":
        timeseries, labels_np = load_cobre_aal_timeseries(
            rois_dir=args.rois_dir,
            phenotypic_csv=args.phenotypic_csv,
            subject_column=args.cobre_subject_column,
            label_column=args.cobre_label_column,
        )
        label_names = {0: "control", 1: "schizophrenia"}
        positive_name = "schizophrenia"
    else:
        timeseries, labels_np = load_abide_aal_timeseries(
            rois_dir=args.rois_dir,
            phenotypic_csv=args.phenotypic_csv,
        )
        label_names = {0: "control", 1: "autism"}
        positive_name = "autism"
    corr_np, adjacency_np = build_static_connectivity(timeseries, args.threshold_p)
    labels = torch.as_tensor(labels_np, dtype=torch.long)
    correlations = torch.as_tensor(corr_np, dtype=torch.float32)
    adjacencies = torch.as_tensor(adjacency_np, dtype=torch.float32)

    train_idx, val_idx, test_idx = make_splits(labels_np, args.seed)
    print(f"Subjects: {len(labels_np)}  ROIs: {correlations.shape[-1]}")
    print(f"Static connectivity: full-series LDW correlation, threshold_p={args.threshold_p}")
    print(f"Label counts: control={(labels == 0).sum().item()}  {positive_name}={(labels == 1).sum().item()}")
    print(f"Split sizes: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    train_loader = make_loader(correlations, adjacencies, labels, train_idx, args.batch_size, True)
    val_loader = make_loader(correlations, adjacencies, labels, val_idx, args.batch_size, False)
    test_loader = make_loader(correlations, adjacencies, labels, test_idx, args.batch_size, False)

    model = FmriGCN(
        num_nodes=correlations.shape[-1],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        out_dim=2,
        k=args.k,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for corr_b, adj_b, labels_b in train_loader:
            corr_b = corr_b.to(device)
            adj_b = adj_b.to(device)
            labels_b = labels_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(corr_b, adj_b)
            loss = F.cross_entropy(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        is_better = (
            val_metrics["acc"] > best_val_acc
            or (
                val_metrics["acc"] == best_val_acc
                and val_metrics["loss"] < best_val_loss
            )
        )
        if is_better:
            best_val_acc = val_metrics["acc"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % args.print_every == 0:
            train_metrics = evaluate(model, train_loader, device)
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}"
            )

        if bad_epochs >= args.patience:
            print(f"Early stopped at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    print(
        f"\nBest epoch: {best_epoch}  "
        f"best val loss={best_val_loss:.4f}  best val acc={best_val_acc:.4f}"
    )
    print(f"Test loss={test_metrics['loss']:.4f}  test acc={test_metrics['acc']:.4f}")
    print("Targets:", [label_names[int(v)] for v in test_metrics["labels"].tolist()])
    print("Preds:  ", [label_names[int(v)] for v in test_metrics["preds"].tolist()])
    print(
        f"Prob control/{positive_name}:",
        [[round(float(x), 4) for x in row] for row in test_metrics["probs"].tolist()],
    )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "test_metrics": {
                    "loss": test_metrics["loss"],
                    "acc": test_metrics["acc"],
                    "labels": test_metrics["labels"],
                    "preds": test_metrics["preds"],
                    "probs": test_metrics["probs"],
                },
                "splits": {
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "test_idx": test_idx,
                },
            },
            save_path,
        )
        print(f"Saved checkpoint to: {save_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a static full-series AAL fMRI connectivity baseline."
    )
    parser.add_argument("--rois_dir", default="./data/nilearn/ABIDE_pcp/cpac/nofilt_noglobal")
    parser.add_argument("--phenotypic_csv", default="./data/manifests/abide_rois_aal_phenotypic.csv")
    parser.add_argument(
        "--source",
        choices=("abide_aal", "cobre_aal"),
        default="abide_aal",
        help="AAL ROI source and label convention.",
    )
    parser.add_argument(
        "--cobre_subject_column",
        default=None,
        help="Optional COBRE CSV subject label column. Defaults to the first column.",
    )
    parser.add_argument(
        "--cobre_label_column",
        default="sz",
        help="COBRE CSV diagnosis column, default sz (1=schizophrenia, 0=control).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--threshold_p", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--save_path", default="./runs/static_aal_10/fmri_gcn_static_aal10.pth")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return train(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
