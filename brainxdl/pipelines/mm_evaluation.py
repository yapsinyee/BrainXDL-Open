"""Reusable MM checkpoint evaluation workflow."""

import argparse
import os
import warnings
from typing import Optional

import dill
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from brainxdl.common.seed import seed_everything
from brainxdl.mm.data.dataset import prepare_data, resolve_cache_path
from brainxdl.mm.models.brainmgf import build_model
from brainxdl.mm.training.trainer import evaluate_metrics


def evaluate_mm_checkpoint(args: argparse.Namespace) -> dict[str, float]:
    """Load an MM checkpoint, evaluate it on the resolved dataset, and return metrics."""
    seed_everything(args.seed)

    args.model = args.model.lower()
    if args.model == "brainmgf_dynamic":
        args.model = "mm_brainxdl_dynamic"
    args.graph_mode = getattr(args, "graph_mode", "static").lower()
    args.loss_type = args.loss_type.lower()
    binary_mode = args.loss_type == "bce"

    if args.model == "gcn":
        args.modality = "fmri"
        args.use_fmri = True
        args.use_eeg = False
    else:
        args.modality = args.modality.lower()
        args.use_fmri = args.modality in ("fusion", "fmri")
        args.use_eeg = args.modality in ("fusion", "eeg")

    if getattr(args, "graph_mode", "static") == "dynamic" and args.model != "mm_brainxdl_dynamic":
        raise ValueError("Dynamic graph mode currently supported only with --model mm_brainxdl_dynamic.")

    args.cache_path = resolve_cache_path(args)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_cache and args.cache_path and os.path.isfile(args.cache_path):
        with open(args.cache_path, "rb") as f:
            cache_payload = dill.load(f)
        if isinstance(cache_payload, dict):
            corr_fmri = cache_payload["corr_fmri"]
            pcorr_fmri = cache_payload["pcorr_fmri"]
            corr_eeg = cache_payload["corr_eeg"]
            pcorr_eeg = cache_payload["pcorr_eeg"]
            labels_np = cache_payload["labels"]
            pseudo_fmri = cache_payload["pseudo_fmri"]
            pseudo_eeg = cache_payload["pseudo_eeg"]
        else:
            corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels_np, pseudo_fmri, pseudo_eeg, *_ = cache_payload
    else:
        corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels_np, pseudo_fmri, pseudo_eeg, *_ = prepare_data(args)

    dataset = TensorDataset(
        torch.tensor(corr_fmri, dtype=torch.float32),
        torch.tensor(corr_eeg, dtype=torch.float32),
        torch.tensor(pcorr_fmri, dtype=torch.float32),
        torch.tensor(pcorr_eeg, dtype=torch.float32),
        torch.tensor(labels_np, dtype=torch.long),
        torch.tensor(pseudo_fmri, dtype=torch.float32),
        torch.tensor(pseudo_eeg, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    def _infer_num_nodes(array: np.ndarray, label: str) -> int:
        if array.ndim < 3:
            raise ValueError(f"Unexpected shape for {label}: {array.shape}")
        last, second_last = int(array.shape[-1]), int(array.shape[-2])
        if last == second_last:
            return last
        if array.ndim >= 4 and int(array.shape[-3]) == second_last:
            return second_last
        return max(last, second_last)

    n_fmri = _infer_num_nodes(corr_fmri, "corr_fmri")
    n_eeg = _infer_num_nodes(corr_eeg, "corr_eeg")
    model_out_dim = 1 if binary_mode else args.out_channels
    model = build_model(args, n_fmri, n_eeg, model_out_dim, device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    if "metric_name" in ckpt and "metric_value" in ckpt:
        print(f"  Saved metric: {ckpt['metric_name']}={ckpt['metric_value']}")

    if binary_mode:
        if args.pos_weight is not None:
            pos_weight_value = float(args.pos_weight)
        else:
            pos_weight_value = 1.0
            warnings.warn("No pos_weight supplied for BCE evaluation; using pos_weight=1.0.")
        criterion: nn.Module = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    metrics = evaluate_metrics(loader, model, criterion, device, args, binary_mode=binary_mode)

    print("Evaluation metrics (% where applicable):")
    for key, value in metrics.items():
        if key == "loss":
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value * 100:.2f}")

    return metrics
