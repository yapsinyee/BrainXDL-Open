"""Reusable ST-BrainXDL training workflow orchestration.

This module intentionally starts after CLI parsing: callers provide an object
with the same attributes as ``scripts/st/st_train.py`` arguments, and the workflow
loads fold data, builds datasets/loaders, configures checkpoints, and runs
training without owning ST model logic.
"""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from brainxdl.common.device import pick_device
from brainxdl.common.seed import seed_everything
from brainxdl.st.dataset import STGraphDataset, load_fold_data, padseq
from brainxdl.st.trainer import init_checkpoint, train


@dataclass(frozen=True)
class STWorkflowResult:
    """Artifacts returned by ``run_st_training_workflow``."""

    model: Any
    setting: dict
    save_path: str
    partition: list[int]
    train_losses: dict
    val_losses: dict
    test_losses: dict


def infer_model_dimensions(*graph_arrays):
    """Infer ST model input dimensions from preprocessed graph folds."""
    first_graph = None
    labels = []
    for graph_array in graph_arrays:
        for graph in graph_array.reshape(-1):
            if graph is None:
                continue
            if first_graph is None and not bool(graph.pad):
                first_graph = graph
            if not bool(graph.pad):
                labels.append(int(graph.y))

    if first_graph is None:
        raise ValueError("Could not infer model dimensions from empty graph folds.")

    y_dim = max(labels) + 1 if labels else int(first_graph.num_classes)
    return {
        "num_nodes": int(first_graph.num_nodes),
        "num_classes": max(int(first_graph.num_classes), y_dim),
        "x_dim": int(first_graph.num_node_features),
        "y_dim": y_dim,
    }


def subsample_graph_windows(graphs: np.ndarray, factor: int) -> np.ndarray:
    """Uniformly thin graph windows while retaining every subject's last step."""
    factor = int(factor)
    if factor <= 1:
        return graphs
    selected = set(range(0, graphs.shape[1], factor))
    for row in graphs:
        selected.update(
            index for index, graph in enumerate(row)
            if graph is not None and bool(graph.last)
        )
    output = graphs[:, sorted(selected)]
    for row in output:
        if sum(bool(graph.last) for graph in row if graph is not None) != 1:
            raise ValueError("Window subsampling must retain exactly one last graph per subject")
    return output


def run_st_training_workflow(args) -> STWorkflowResult:
    """Run ST-BrainXDL training after CLI/config parsing has already happened."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    seed = getattr(args, "seed", None)
    if seed is not None:
        seed_everything(seed)
        print(f"[Seed] {seed}")

    device = pick_device(prefer_mps=args.prefer_mps)
    print(f"[Device] {device}")

    print(f"Loading fold outer={args.outer} inner={args.inner} …")
    train_graphs, test_graphs, val_graphs = load_fold_data(
        args.outer,
        args.inner,
        data_dir=args.data_dir,
    )
    window_subsample = int(getattr(args, "window_subsample", 1))
    train_graphs = subsample_graph_windows(train_graphs, window_subsample)
    test_graphs = subsample_graph_windows(test_graphs, window_subsample)
    val_graphs = subsample_graph_windows(val_graphs, window_subsample)
    if window_subsample > 1:
        print(
            f"Uniform window subsampling: factor={window_subsample}, "
            f"retained_windows={train_graphs.shape[1]}"
        )
    omit_test_loader = bool(getattr(args, "omit_test_loader", False))

    strict_validation = bool(getattr(args, "strict_validation", False))
    refit_full_development = bool(getattr(args, "refit_full_development", False))
    disable_validation = bool(getattr(args, "disable_validation", False))
    if disable_validation and not bool(getattr(args, "no_early_stop", False)):
        raise ValueError("disable_validation requires no_early_stop")
    if refit_full_development:
        train_dataset = STGraphDataset(np.concatenate([train_graphs, val_graphs], axis=0))
    elif strict_validation:
        train_dataset = STGraphDataset(train_graphs)
    else:
        train_dataset = STGraphDataset(np.concatenate([train_graphs, val_graphs], axis=0))
    val_dataset = None if disable_validation else STGraphDataset(val_graphs)
    test_dataset = None if omit_test_loader else STGraphDataset(test_graphs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=padseq,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,
            num_workers=0,
            collate_fn=padseq,
            pin_memory=False,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=False,
            num_workers=0,
            collate_fn=padseq,
            pin_memory=False,
        )

    partition = [
        len(train_dataset),
        0 if val_dataset is None else len(val_dataset),
        0 if test_dataset is None else len(test_dataset),
    ]
    training_label = "train+val refit" if refit_full_development else (
        "train" if strict_validation else "train+val"
    )
    print(f"Partition ({training_label} / val / test): {partition}")

    dimension_graphs = (train_graphs, val_graphs) if omit_test_loader else (
        train_graphs,
        val_graphs,
        test_graphs,
    )
    inferred_dims = infer_model_dimensions(*dimension_graphs)
    print(
        "Inferred graph dimensions: "
        f"nodes={inferred_dims['num_nodes']}  "
        f"x_dim={inferred_dims['x_dim']}  "
        f"y_dim={inferred_dims['y_dim']}"
    )

    model_params = {
        "num_nodes": inferred_dims["num_nodes"],
        "num_classes": inferred_dims["num_classes"],
        "x_dim": inferred_dims["x_dim"],
        "y_dim": inferred_dims["y_dim"],
        "z_hidden_dim": 32,
        "z_dim": 16,
        "z_phi_dim": 8,
        "x_phi_dim": 64,
        "rnn_dim": 16,
        "y_hidden_dim": [32],
        "x_hidden_dim": 64,
        "layer_dims": [],
        "readout_mode": getattr(args, "readout_mode", "legacy"),
        "readout_projection_dim": getattr(args, "readout_projection_dim", 32),
        "stable_fc_skip": bool(getattr(args, "stable_fc_skip", False)),
        "stable_fc_dim": getattr(args, "stable_fc_dim", 32),
    }
    classifier_hidden_dim = getattr(args, "classifier_hidden_dim", None)
    if classifier_hidden_dim is not None:
        model_params["y_hidden_dim"] = [int(classifier_hidden_dim)]
    model_param_overrides = getattr(args, "model_params_override", None) or {}
    model_params.update(model_param_overrides)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = f"ST-BrainXDL_softmax_adv_fold{args.outer}{args.inner}_{args.version}.pth"
    save_path = os.path.join(args.save_dir, ckpt_name)
    rng_path = os.path.join(args.save_dir, "ST-BrainXDL_softmax_adv_fold11.pth")

    anneal_type = "ReduceLROnPlateau"
    setting = {
        "device": str(device),
        "prefer_mps": args.prefer_mps,
        "seed": seed,
        "sparse_on_cpu": not getattr(args, "no_sparse_on_cpu", False),
        "rngPATH": rng_path if os.path.exists(rng_path) else None,
        "model_params": model_params,
        "recurrent": not args.no_recurrent,
        "graphRNN": not args.no_graphRNN,
        "variational": not args.no_variational,
        "DAT": args.dat,
        "kl_weight": getattr(args, "kl_weight", 1.0),
        "kl_start_weight": getattr(args, "kl_start_weight", None),
        "kl_warmup_epochs": getattr(args, "kl_warmup_epochs", 0),
        "bce_weight": getattr(args, "bce_weight", 1.0),
        "mce_weight": getattr(args, "mce_weight", 1.0),
        "encoder_cls_weight": getattr(args, "encoder_cls_weight", 0.0),
        "learnRate": [args.learn_rate, args.learn_rate],
        "yBCEMultiplier": [1, 1],
        "l2factor": [args.l2factor, args.l2factor],
        "selection_metric": getattr(args, "selection_metric", "legacy_train_bce_acc"),
        "lr_annealType": [anneal_type, anneal_type],
        "lr_annealFactor": [0.8, 0.8],
        "lr_annealPatience": [30, 30],
        "partition": partition,
        "strict_validation": strict_validation,
        "refit_full_development": refit_full_development,
        "validation_enabled": not disable_validation,
        "test_during_training": not bool(getattr(args, "no_test_during_training", False)),
    }

    model, optimizers, schedulers, epoch_start, train_losses, val_losses, test_losses = (
        init_checkpoint(setting, save_path, save_path)
    )

    print("\nModel summary:")
    print(model)

    print("\nStarting training …")
    model, train_losses, val_losses, test_losses = train(
        model,
        optimizers,
        schedulers,
        setting,
        save_path,
        train_losses,
        val_losses,
        test_losses,
        train_loader,
        val_loader,
        test_loader,
        device=device,
        epoch_start=epoch_start,
        num_epochs=args.num_epochs,
        grad_threshold=args.grad_clip,
        gradient_clip=True,
        verbose_freq=1,
        verbose=not args.no_verbose,
        val_freq=args.val_freq,
        validation=not disable_validation,
        testing=not bool(getattr(args, "no_test_during_training", False)),
        early_stop_patience=args.patience,
        early_stop=not args.no_early_stop,
    )

    print(f"\nCheckpoint saved to: {save_path}")
    print("To inspect results: python -m brainxdl.st.visualize", save_path)

    return STWorkflowResult(
        model=model,
        setting=setting,
        save_path=save_path,
        partition=partition,
        train_losses=train_losses,
        val_losses=val_losses,
        test_losses=test_losses,
    )
