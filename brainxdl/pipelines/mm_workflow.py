"""Reusable MM training workflow orchestration.

This module starts after CLI parsing. Entry points provide an argparse-like
namespace, and the workflow handles model sequencing, cache/data loading,
split construction, training/evaluation, checkpointing, and optional XAI
orchestration without owning MM model logic.
"""

import argparse
import copy
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from brainxdl.common.checkpointing import to_cpu_state_dict
from brainxdl.common.logging import METRIC_DIRECTIONS, format_metric_value, metric_is_better
from brainxdl.common.seed import seed_everything
from brainxdl.mm.data.dataset import (
    build_weighted_sampler,
    compute_min_max_ratio,
    infer_pos_weight,
    prepare_data,
    resolve_cache_path,
    summarize_split_counts,
)
from brainxdl.xai.explainer import maybe_run_gnn_explanations
from brainxdl.mm.models.brainmgf import build_model
from brainxdl.mm.training.trainer import (
    analyze_fusion_gate_bias,
    evaluate,
    evaluate_metrics,
    run_kfold_cv,
    train_one_epoch,
)
from brainxdl.mm.utils.io import compute_run_slug, resolve_checkpoint_dir
from brainxdl.xai.latent import generate_latent_manifolds


def load_fold_aligned_st_embeddings(
    *,
    embedding_dir: str,
    manifest_path: str,
    labels_np: np.ndarray,
    groups_np: Optional[np.ndarray],
    feature_set: str = "readout",
    expected_seed: Optional[int] = None,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[int]]:
    """Load frozen ST embeddings into per-outer-fold MM sample order."""
    embedding_root = Path(embedding_dir)
    if not embedding_root.is_dir():
        raise FileNotFoundError(f"ST embedding directory not found: {embedding_root}")
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"ST fold manifest not found: {manifest_file}")

    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    manifest_seed = manifest.get("seed")
    if expected_seed is not None:
        if manifest_seed is None:
            raise ValueError("ST embedding manifest does not record a seed.")
        if int(manifest_seed) != int(expected_seed):
            raise ValueError(
                f"ST embedding seed mismatch: expected {expected_seed}, manifest has {manifest_seed}."
            )

    folds_meta = manifest.get("folds")
    if not isinstance(folds_meta, dict):
        folds_meta = {}
        fold_data_dir = manifest_file.parent
        for fold in range(1, 6):
            fold_file = fold_data_dir / f"graphs_outer{fold}_inner1.pkl"
            if not fold_file.is_file():
                raise ValueError(
                    "ST fold manifest does not contain a 'folds' mapping and "
                    f"fallback fold file is missing: {fold_file}"
                )
            fold_payload = torch.load(fold_file, map_location="cpu", weights_only=False)
            metadata = fold_payload.get("metadata", {})
            split_indices = metadata.get("split_indices")
            participant_groups = metadata.get("participant_groups")
            if split_indices is None or participant_groups is None:
                raise ValueError(f"{fold_file} metadata lacks split indices or participant groups.")
            folds_meta[f"outer{fold}_inner1"] = {
                split: {
                    "indices": indices,
                    "participant_groups": [participant_groups[index] for index in indices],
                }
                for split, indices in split_indices.items()
            }

    if groups_np is None:
        warnings.warn("MM participant groups are unavailable; ST embedding alignment will verify labels only.")

    n_samples = int(len(labels_np))
    manifest_n_samples = manifest.get("n_samples")
    if manifest_n_samples is not None and int(manifest_n_samples) != n_samples:
        raise ValueError(
            f"ST/MM sample-count mismatch: manifest has {manifest_n_samples}, MM has {n_samples}."
        )
    manifest_sample_ids = manifest.get("sample_ids")
    if manifest_sample_ids is not None:
        if len(manifest_sample_ids) != n_samples:
            raise ValueError("ST embedding manifest sample_ids length does not match MM samples.")
        if len(set(map(str, manifest_sample_ids))) != n_samples:
            raise ValueError("ST embedding manifest contains duplicate sample_ids.")

    fold_embeddings: list[np.ndarray] = []
    fold_splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    fold_embedding_dims: list[int] = []

    for fold in range(1, 6):
        key = f"outer{fold}_inner1"
        fold_meta = folds_meta.get(key)
        if not fold_meta:
            raise ValueError(f"Missing {key} in ST fold manifest.")
        npz_path = embedding_root / f"st_embedding_features_fold{fold}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing ST embedding NPZ: {npz_path}")
        data = np.load(npz_path, allow_pickle=False)

        fold_matrix = None
        fold_embedding_dim: Optional[int] = None
        seen_indices: set[int] = set()
        seen_sample_ids: set[str] = set()
        for split in ("train", "val", "test"):
            indices = np.asarray(fold_meta[split]["indices"], dtype=np.int64)
            split_features = np.asarray(data[f"{split}_{feature_set}"], dtype=np.float32)
            split_labels = np.asarray(data[f"{split}_labels"], dtype=np.int64)
            if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
                raise ValueError(f"Fold {fold} {split}: indices must be one-dimensional and unique.")
            if np.any(indices < 0) or np.any(indices >= n_samples):
                raise ValueError(f"Fold {fold} {split}: indices fall outside the MM sample range.")
            overlap = seen_indices.intersection(map(int, indices))
            if overlap:
                raise ValueError(f"Fold {fold}: split indices overlap: {sorted(overlap)[:10]}")
            seen_indices.update(map(int, indices))
            if split_features.ndim != 2 or split_features.shape[1] < 1:
                raise ValueError(f"Fold {fold} {split}: ST features must be a non-empty 2D matrix.")
            if not np.isfinite(split_features).all():
                raise ValueError(f"Fold {fold} {split}: ST features contain NaN or Inf.")
            if len(indices) != split_features.shape[0]:
                raise ValueError(
                    f"Fold {fold} {split}: manifest indices and embedding rows differ "
                    f"({len(indices)} vs {split_features.shape[0]})."
                )
            if not np.array_equal(labels_np[indices].astype(int), split_labels.astype(int)):
                raise ValueError(f"Fold {fold} {split}: ST labels do not match MM labels.")
            if groups_np is not None:
                split_groups = np.asarray(fold_meta[split]["participant_groups"], dtype=object)
                if not np.array_equal(groups_np[indices].astype(str), split_groups.astype(str)):
                    raise ValueError(f"Fold {fold} {split}: ST participant groups do not match MM groups.")
            subject_index_key = f"{split}_subject_index"
            if subject_index_key in data and not np.array_equal(
                np.asarray(data[subject_index_key], dtype=np.int64), indices
            ):
                raise ValueError(f"Fold {fold} {split}: NPZ subject indices do not match the manifest.")
            sample_id_key = f"{split}_sample_id"
            if sample_id_key in data:
                split_sample_ids = np.asarray(data[sample_id_key]).astype(str)
                expected_ids = fold_meta[split].get("sample_ids")
                if expected_ids is None and manifest_sample_ids is not None:
                    expected_ids = [manifest_sample_ids[int(index)] for index in indices]
                if expected_ids is not None and not np.array_equal(
                    split_sample_ids, np.asarray(expected_ids).astype(str)
                ):
                    raise ValueError(f"Fold {fold} {split}: NPZ sample IDs do not match the manifest.")
                duplicate_ids = seen_sample_ids.intersection(map(str, split_sample_ids))
                if duplicate_ids:
                    raise ValueError(f"Fold {fold}: duplicate sample IDs across splits.")
                seen_sample_ids.update(map(str, split_sample_ids))
            participant_key = f"{split}_participant_id"
            if participant_key in data and groups_np is not None:
                if not np.array_equal(
                    np.asarray(data[participant_key]).astype(str), groups_np[indices].astype(str)
                ):
                    raise ValueError(f"Fold {fold} {split}: NPZ participant IDs do not match MM groups.")
            seed_key = f"{split}_seed"
            if expected_seed is not None and seed_key in data:
                split_seeds = np.asarray(data[seed_key], dtype=np.int64)
                if not np.all(split_seeds == int(expected_seed)):
                    raise ValueError(f"Fold {fold} {split}: NPZ source seed is not {expected_seed}.")
            if fold_matrix is None:
                fold_embedding_dim = int(split_features.shape[1])
                fold_matrix = np.zeros((n_samples, fold_embedding_dim), dtype=np.float32)
            elif split_features.shape[1] != fold_embedding_dim:
                raise ValueError(f"Fold {fold}: ST embedding dimensions differ across splits.")
            fold_matrix[indices] = split_features

        if fold_matrix is None or fold_embedding_dim is None:
            raise ValueError(f"Fold {fold}: no embeddings loaded.")
        if seen_indices != set(range(n_samples)):
            missing = sorted(set(range(n_samples)) - seen_indices)
            raise ValueError(f"Fold {fold}: split indices do not cover all MM samples; missing={missing[:10]}.")
        if seen_sample_ids and len(seen_sample_ids) != n_samples:
            raise ValueError(f"Fold {fold}: sample IDs are not unique and complete.")
        train_idx = np.asarray(fold_meta["train"]["indices"], dtype=np.int64)
        val_idx = np.asarray(fold_meta["val"]["indices"], dtype=np.int64)
        test_idx = np.asarray(fold_meta["test"]["indices"], dtype=np.int64)
        if groups_np is not None:
            train_groups = set(groups_np[train_idx].astype(str))
            val_groups = set(groups_np[val_idx].astype(str))
            test_groups = set(groups_np[test_idx].astype(str))
            if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
                raise ValueError(f"Fold {fold}: participant leakage detected across manifest splits.")
        fold_embeddings.append(fold_matrix)
        fold_splits.append((train_idx, val_idx, test_idx))
        fold_embedding_dims.append(fold_embedding_dim)

    if not fold_embedding_dims:
        raise ValueError("No ST embeddings were loaded.")
    return fold_embeddings, fold_splits, fold_embedding_dims


def print_args(args: argparse.Namespace) -> None:
    command = getattr(args, "command", None)
    if command:
        print("Command:")
        print(f"  {command}")
    print("Arguments:")
    for key, value in sorted(vars(args).items()):
        if key == "command" or key.startswith("_"):
            continue
        print(f"  {key}: {value}")


def validate_participant_grouping(
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    *,
    n_folds: int,
    required: bool,
) -> Optional[np.ndarray]:
    """Validate the grouping contract for repeated-measure public workflows.

    The check prevents the public CLI from silently falling back to
    observation-level splits when participant identifiers are unavailable.
    It does not construct or otherwise alter the requested folds.
    """
    if not required:
        return None if groups is None else np.asarray(groups)
    if groups is None:
        raise ValueError(
            "Participant identifiers are required for this workflow; refusing "
            "an observation-level split that could leak participants."
        )
    checked = np.asarray(groups)
    if checked.ndim != 1 or len(checked) != len(labels):
        raise ValueError("Participant identifiers must be a one-dimensional label-aligned array.")
    if int(n_folds) < 2 and len(np.unique(checked)) < len(checked):
        raise ValueError(
            "Repeated participant observations require grouped cross-validation; "
            "set --n_folds to at least 2 (five folds are used by the thesis protocol)."
        )
    return checked


def run_mm_training_workflow(args: argparse.Namespace) -> None:
    args.model = args.model.lower()
    if args.model == "brainmgf_dynamic":
        args.model = "mm_brainxdl_dynamic"
    args.graph_mode = getattr(args, "graph_mode", "static").lower()

    if args.model == "all":
        if args.graph_mode == "dynamic":
            model_sequence = ["mm_brainxdl_dynamic"]
            fusion_models = {"mm_brainxdl_dynamic"}
            print("graph_mode=dynamic -> running only mm_brainxdl_dynamic within --model all.")
        else:
            model_sequence = [
                "mmodal_gcn",
                "mmodal_gcn_nogate",
                "fusion_mlp",
                "mm_brainxdl_dynamic",
                "gcn",
            ]
            fusion_models = {"mmodal_gcn", "mmodal_gcn_nogate", "fusion_mlp", "mm_brainxdl_dynamic"}
        for model_name in model_sequence:
            sub_args = copy.deepcopy(args)
            sub_args.model = model_name
            if model_name in fusion_models:
                sub_args.modality = "fusion"
            else:
                sub_args.modality = "fmri"
            print("\n" + "=" * 80)
            print(f"Running model: {model_name}")
            print("=" * 80)
            print_args(sub_args)
            run_mm_training_workflow(sub_args)
        return

    seed_everything(args.seed)

    args.loss_type = args.loss_type.lower()
    binary_mode = args.loss_type == "bce"

    if args.model == "gcn":
        if args.modality.lower() != "fmri":
            warnings.warn("Overriding modality to 'fmri' for the selected fMRI-only baseline.")
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

    if getattr(args, "plot_latents", False):
        args.latent_dir = os.path.abspath(args.latent_dir)

    tasks_np: Optional[np.ndarray] = None
    groups_np: Optional[np.ndarray] = None
    if args.use_cache and args.cache_path and os.path.isfile(args.cache_path):
        with open(args.cache_path, "rb") as cache_file:
            cache_payload = dill.load(cache_file)
        if isinstance(cache_payload, dict):
            corr_fmri = cache_payload.get("corr_fmri")
            pcorr_fmri = cache_payload.get("pcorr_fmri")
            corr_eeg = cache_payload.get("corr_eeg")
            pcorr_eeg = cache_payload.get("pcorr_eeg")
            labels_np = cache_payload.get("labels")
            pseudo_fmri = cache_payload.get("pseudo_fmri")
            pseudo_eeg = cache_payload.get("pseudo_eeg")
            cached_groups = cache_payload.get("groups")
            if cached_groups is not None:
                groups_np = np.asarray(cached_groups)
            cached_tasks = cache_payload.get("tasks")
            if cached_tasks is not None:
                tasks_np = np.asarray(cached_tasks)
        else:
            try:
                (
                    corr_fmri,
                    pcorr_fmri,
                    corr_eeg,
                    pcorr_eeg,
                    labels_np,
                    pseudo_fmri,
                    pseudo_eeg,
                    groups_np,
                    tasks_np,
                ) = cache_payload
            except ValueError:
                try:
                    (
                        corr_fmri,
                        pcorr_fmri,
                        corr_eeg,
                        pcorr_eeg,
                        labels_np,
                        pseudo_fmri,
                        pseudo_eeg,
                        tasks_np,
                    ) = cache_payload
                except ValueError:
                    (
                        corr_fmri,
                        pcorr_fmri,
                        corr_eeg,
                        pcorr_eeg,
                        labels_np,
                        pseudo_fmri,
                        pseudo_eeg,
                    ) = cache_payload
    else:
        prepared = prepare_data(args)
        if len(prepared) == 9:
            corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels_np, pseudo_fmri, pseudo_eeg, groups_np, tasks_np = prepared
        elif len(prepared) == 8:
            corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels_np, pseudo_fmri, pseudo_eeg, tasks_np = prepared
        else:
            corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels_np, pseudo_fmri, pseudo_eeg = prepared
        if args.cache_path:
            payload = {
                "corr_fmri": corr_fmri,
                "pcorr_fmri": pcorr_fmri,
                "corr_eeg": corr_eeg,
                "pcorr_eeg": pcorr_eeg,
                "labels": labels_np,
                "pseudo_fmri": pseudo_fmri,
                "pseudo_eeg": pseudo_eeg,
                "groups": groups_np,
                "tasks": tasks_np,
            }
            with open(args.cache_path, "wb") as f:
                dill.dump(payload, f)

    if tasks_np is None:
        tasks_np = np.array(["unknown"] * len(labels_np), dtype=object)
    groups_np = validate_participant_grouping(
        labels_np,
        groups_np,
        n_folds=getattr(args, "n_folds", 1),
        required=bool(getattr(args, "require_grouped_cv", False)),
    )

    num_subjects = int(labels_np.shape[0])
    print(f"Loaded subjects: {num_subjects}")

    if getattr(args, "plot_debug", False):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fmri_mat = corr_fmri[0] if corr_fmri.ndim == 3 else corr_fmri[0, 0]
        eeg_mat = corr_eeg[0] if corr_eeg.ndim == 3 else corr_eeg[0, 0]
        axs[0].imshow(fmri_mat, cmap="bwr", vmin=-1, vmax=1)
        axs[0].set_title("fMRI Pearson")
        axs[1].imshow(eeg_mat, cmap="bwr", vmin=-1, vmax=1)
        axs[1].set_title("EEG Pearson")
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    pearson_fmri = torch.tensor(corr_fmri, dtype=torch.float32)
    pearson_eeg = torch.tensor(corr_eeg, dtype=torch.float32)
    pcorr_fmri_t = torch.tensor(pcorr_fmri, dtype=torch.float32)
    pcorr_eeg_t = torch.tensor(pcorr_eeg, dtype=torch.float32)
    pseudo_fmri_t = torch.tensor(pseudo_fmri, dtype=torch.float32)
    pseudo_eeg_t = torch.tensor(pseudo_eeg, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)

    dataset = TensorDataset(
        pearson_fmri,
        pearson_eeg,
        pcorr_fmri_t,
        pcorr_eeg_t,
        labels,
        pseudo_fmri_t,
        pseudo_eeg_t,
    )

    if getattr(args, "st_embedding_dir", None):
        if not getattr(args, "st_fold_manifest", None):
            raise ValueError("--st_fold_manifest is required with --st_embedding_dir.")
        fold_embeddings, fold_splits, fold_embedding_dims = load_fold_aligned_st_embeddings(
            embedding_dir=args.st_embedding_dir,
            manifest_path=args.st_fold_manifest,
            labels_np=labels_np,
            groups_np=groups_np,
            feature_set=getattr(args, "st_embedding_feature_set", "readout"),
            expected_seed=getattr(args, "st_embedding_seed", None),
        )
        args._st_embeddings_by_fold = fold_embeddings
        args._st_embedding_dims_by_fold = fold_embedding_dims
        args._fold_splits = fold_splits
        args.st_embedding_dim = fold_embedding_dims[0]
        dimensions_text = ",".join(map(str, fold_embedding_dims))
        print(
            f"Loaded fold-aligned ST {args.st_embedding_feature_set} embeddings "
            f"from {args.st_embedding_dir} (fold dims={dimensions_text})."
        )

    if getattr(args, "n_folds", 1) and args.n_folds > 1:
        run_kfold_cv(dataset, labels_np, args, tasks_np, groups_np)
        return

    labels_np = labels_np.astype(int)
    split_seed = getattr(args, "split_seed", None)
    if split_seed is None:
        split_seed = args.seed
    train_idx, test_idx = train_test_split(
        np.arange(len(labels_np)),
        test_size=0.2,
        stratify=labels_np,
        random_state=split_seed,
    )
    train_labels = labels_np[train_idx]
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=getattr(args, "val_ratio", 0.2),
        stratify=train_labels,
        random_state=split_seed,
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    num_classes = int(labels_np.max()) + 1
    train_hist = summarize_split_counts("Train", train_idx, labels_np, num_classes=num_classes)
    summarize_split_counts("Val", val_idx, labels_np, num_classes=num_classes)
    summarize_split_counts("Test", test_idx, labels_np, num_classes=num_classes)

    imbalance_ratio = compute_min_max_ratio(train_hist)
    auto_balance = getattr(args, "balance_threshold", 0.0) > 0 and imbalance_ratio < args.balance_threshold
    use_sampler = getattr(args, "balance_batches", False) or auto_balance
    if use_sampler:
        sampler = build_weighted_sampler(labels_np[train_idx], train_hist)
        if auto_balance and not getattr(args, "balance_batches", False):
            print(
                f"Activating WeightedRandomSampler (ratio={imbalance_ratio:.3f} < "
                f"threshold={args.balance_threshold:.3f})"
            )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _infer_num_nodes(t: torch.Tensor, label: str) -> int:
        if t.dim() < 3:
            raise ValueError(f"Unexpected shape for {label}: {tuple(t.shape)}")
        last, second_last = int(t.shape[-1]), int(t.shape[-2])
        if last == second_last:
            return last
        if t.dim() >= 4 and int(t.shape[-3]) == second_last:
            return second_last
        return max(last, second_last)

    n_fmri = _infer_num_nodes(pearson_fmri, "pearson_fmri")
    n_eeg = _infer_num_nodes(pearson_eeg, "pearson_eeg")
    model_out_dim = 1 if binary_mode else args.out_channels
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
        print(f"Using BCEWithLogitsLoss (pos_weight={pos_weight_value:.3f})")
    else:
        if getattr(args, "class_weighted", False):
            class_counts = np.bincount(labels_np[train_idx], minlength=num_classes)
            class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
            print(f"Using class-weighted CrossEntropyLoss with weights={class_weights.detach().cpu().tolist()}")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    save_best = getattr(args, "save_best_model", False)
    monitored_metric = getattr(args, "save_metric", "acc")
    if save_best and monitored_metric not in METRIC_DIRECTIONS:
        warnings.warn(
            f"Metric '{monitored_metric}' is not supported for checkpointing; disabling --save_best_model."
        )
        save_best = False
    checkpoint_path: Optional[str] = None
    if save_best:
        identifier = f"train_{compute_run_slug(args)}"
        checkpoint_root = resolve_checkpoint_dir(args)
        checkpoint_dir = os.path.join(checkpoint_root, identifier)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{identifier}_best.pt")
    best_checkpoint_metric: Optional[float] = None

    best_metric_val = float("inf") if args.es_metric == "loss" else -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

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
            f"Epoch {epoch:03d} | Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        curr = val_loss if args.es_metric == "loss" else val_acc
        improved = (
            (best_metric_val - curr) > args.es_min_delta
            if args.es_metric == "loss"
            else (curr - best_metric_val) > args.es_min_delta
        )
        if improved:
            best_metric_val = curr
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if (not getattr(args, "no_early_stop", False)) and epoch >= args.min_epochs and no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    metrics = evaluate_metrics(
        test_loader,
        model,
        criterion,
        device,
        args,
        binary_mode=binary_mode,
    )
    print("Test metrics (% where applicable):")
    for key, value in metrics.items():
        if key == "loss":
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value * 100:.2f}")

    analyze_fusion_gate_bias(
        model,
        test_dataset,
        args,
        device,
        tasks_np,
        split_name="test",
    )

    split_map = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    maybe_run_gnn_explanations(
        model,
        split_map,
        args,
        device,
        binary_mode,
        labels_np,
    )

    if save_best and checkpoint_path:
        metric_value = metrics.get(monitored_metric)
        if metric_value is None:
            warnings.warn(
                f"Metric '{monitored_metric}' not found in evaluation outputs; "
                "skipping checkpoint saving."
            )
        elif metric_is_better(monitored_metric, metric_value, best_checkpoint_metric):
            best_checkpoint_metric = float(metric_value)
            metrics_snapshot = {key: float(val) for key, val in metrics.items()}
            payload = {
                "state_dict": to_cpu_state_dict(best_state),
                "metric_name": monitored_metric,
                "metric_value": best_checkpoint_metric,
                "metrics": metrics_snapshot,
                "command": getattr(args, "command", None),
            }
            torch.save(payload, checkpoint_path)
            print(
                f"Saved best checkpoint to {checkpoint_path} "
                f"({monitored_metric}={format_metric_value(monitored_metric, metric_value)})"
            )

    if getattr(args, "plot_latents", False):
        if args.latent_split == "train":
            target_dataset = train_dataset
            split_tag = "train"
        elif args.latent_split == "val":
            target_dataset = val_dataset
            split_tag = "val"
        elif args.latent_split == "train_test":
            target_dataset = ConcatDataset([train_dataset, test_dataset])
            split_tag = "train-test"
        elif args.latent_split == "all":
            target_dataset = dataset
            split_tag = "all"
        else:
            target_dataset = test_dataset
            split_tag = "test"
        generate_latent_manifolds(model, target_dataset, args, device, split_tag)

main = run_mm_training_workflow
