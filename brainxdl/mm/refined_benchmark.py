"""Controlled participant-grouped refinements of static MM-BrainXDL.

The saved participant manifest remains the sole split authority.  This module
adds representation diagnostics and a small, named ablation family without
changing the completed participant-grouped benchmark outputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from brainxdl.common.metrics import choose_binary_threshold, compute_classification_metrics
from brainxdl.common.seed import seed_everything
from brainxdl.mm.models.brainmgf import MMBrainXDL
from brainxdl.mm.participant_grouped_benchmark import (
    METRICS,
    SPLITS,
    BenchmarkData,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    fold_indices,
    metric_dict,
    read_manifest,
    validate_manifest,
)


@dataclass(frozen=True)
class RefinedVariant:
    name: str
    fusion_mode: str
    gate_type: str
    connectivity_skip_dim: int
    residual_connectivity_dim: int
    residual_lambda: float
    classifier_type: str
    output_dim: int
    loss: str
    description: str
    graph_encoder_type: str = "published"
    graph_reconstruction_weight: float = 0.0


VARIANTS = {
    "mm_a0_original": RefinedVariant(
        "mm_a0_original", "adaptive_sum", "scalar", 0, 0, 1.0, "mlp", 2, "cross_entropy",
        "Original scalar adaptive-sum fusion and original MLP classifier.",
    ),
    "mm_a1_gated_concat": RefinedVariant(
        "mm_a1_gated_concat", "gated_concat", "scalar", 0, 0, 1.0, "mlp", 2, "cross_entropy",
        "Scalar gated concatenation with the original MLP classifier.",
    ),
    "mm_a2_gated_concat_linear": RefinedVariant(
        "mm_a2_gated_concat_linear", "gated_concat", "scalar", 0, 0, 1.0, "linear", 1, "bce_with_logits",
        "Scalar gated concatenation with a one-logit linear classifier.",
    ),
    "mm_a3_gated_concat_skip": RefinedVariant(
        "mm_a3_gated_concat_skip", "gated_concat", "scalar", 32, 0, 1.0, "mlp", 2, "cross_entropy",
        "Scalar gated concatenation plus 32-dimensional per-modality FC skips and the original MLP head.",
    ),
    "mm_a4_gated_concat_skip_linear": RefinedVariant(
        "mm_a4_gated_concat_skip_linear", "gated_concat", "scalar", 32, 0, 1.0, "linear", 1, "bce_with_logits",
        "Scalar gated concatenation plus 32-dimensional per-modality FC skips and a one-logit linear head.",
    ),
    "mm_a5_featurewise_gate": RefinedVariant(
        "mm_a5_featurewise_gate", "gated_concat", "featurewise", 0, 0, 1.0, "mlp", 2, "cross_entropy",
        "Bounded feature-wise gated concatenation with the original MLP classifier.",
    ),
    "mm_r1_residual_equal": RefinedVariant(
        "mm_r1_residual_equal", "average", "scalar", 0, 32, 1.0, "mlp", 2, "cross_entropy",
        "Additive raw-connectivity residuals with fixed lambda=1 and equal fusion.",
    ),
    "mm_r2_residual_adaptive": RefinedVariant(
        "mm_r2_residual_adaptive", "adaptive_sum", "scalar", 0, 32, 1.0, "mlp", 2, "cross_entropy",
        "Additive raw-connectivity residuals with fixed lambda=1 and scalar adaptive fusion.",
    ),
    "mm_g1_self_preserving": RefinedVariant(
        "mm_g1_self_preserving", "adaptive_sum", "scalar", 0, 0, 1.0, "mlp", 2,
        "cross_entropy",
        "Self-preserving graph updates with separate learned self and neighbourhood paths.",
        "self_preserving",
    ),
    "mm_g2_sign_separated": RefinedVariant(
        "mm_g2_sign_separated", "adaptive_sum", "scalar", 0, 0, 1.0, "mlp", 2,
        "cross_entropy",
        "Sign-separated row-normalized propagation in the first graph block only.",
        "sign_separated_first",
    ),
    "mm_g3_graph_autoencoder": RefinedVariant(
        "mm_g3_graph_autoencoder", "adaptive_sum", "scalar", 0, 0, 1.0, "mlp", 2,
        "cross_entropy",
        "MM-G2 plus ST-BrainXDL-inspired graph-support reconstruction from final node embeddings.",
        "sign_separated_first",
        0.1,
    ),
}

PROBE_REPRESENTATIONS = (
    "raw_fmri", "raw_eeg", "raw_combined",
    "graph_block0", "graph_block1", "graph_block2",
    "z_fmri", "z_eeg", "z_concat", "adaptive_sum",
)


def build_refined_model(data: BenchmarkData, variant: RefinedVariant) -> MMBrainXDL:
    return MMBrainXDL(
        in_channels_fmri=data.corr_fmri.shape[1],
        in_channels_eeg=data.corr_eeg.shape[1],
        embed_dim=128,
        fc_out=variant.output_dim,
        nrois_fmri=data.corr_fmri.shape[1],
        nrois_eeg=data.corr_eeg.shape[1],
        k_fmri=40,
        k_eeg=40,
        dropout=0.3,
        use_fmri=True,
        use_eeg=True,
        fusion_mode=variant.fusion_mode,
        gate_type=variant.gate_type,
        connectivity_skip_dim=variant.connectivity_skip_dim,
        residual_connectivity_dim=variant.residual_connectivity_dim,
        residual_lambda=variant.residual_lambda,
        classifier_type=variant.classifier_type,
        graph_encoder_type=variant.graph_encoder_type,
        graph_reconstruction=variant.graph_reconstruction_weight > 0,
    )


def _arrays(data: BenchmarkData) -> dict[str, torch.Tensor]:
    return {
        "corr_fmri": torch.from_numpy(data.corr_fmri),
        "pcorr_fmri": torch.from_numpy(data.pcorr_fmri),
        "corr_eeg": torch.from_numpy(data.corr_eeg),
        "pcorr_eeg": torch.from_numpy(data.pcorr_eeg),
        "pseudo_fmri": torch.from_numpy(data.pseudo_fmri),
        "pseudo_eeg": torch.from_numpy(data.pseudo_eeg),
        "labels": torch.from_numpy(data.labels.astype(np.int64)),
    }


def _model_inputs(arrays: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        arrays["corr_fmri"][indices].to(device),
        arrays["corr_eeg"][indices].to(device),
        arrays["pcorr_fmri"][indices].to(device),
        arrays["pcorr_eeg"][indices].to(device),
        arrays["pseudo_fmri"][indices].to(device),
        arrays["pseudo_eeg"][indices].to(device),
    )


def _loss(logits: torch.Tensor, labels: torch.Tensor, variant: RefinedVariant) -> torch.Tensor:
    if variant.loss == "bce_with_logits":
        return nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float())
    return nn.functional.cross_entropy(logits, labels)


def _probabilities(logits: torch.Tensor, variant: RefinedVariant) -> torch.Tensor:
    if variant.loss == "bce_with_logits":
        return torch.sigmoid(logits.reshape(-1))
    return torch.softmax(logits, dim=1)[:, 1]


def _predict(
    model: MMBrainXDL,
    arrays: dict[str, torch.Tensor],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    variant: RefinedVariant,
    *,
    components: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    collected: dict[str, list[np.ndarray]] = {}
    loader = DataLoader(TensorDataset(torch.as_tensor(indices, dtype=torch.long)), batch_size=batch_size)
    with torch.no_grad():
        for (batch_indices,) in loader:
            inputs = _model_inputs(arrays, batch_indices, device)
            logits = model(*inputs)
            all_logits.append(logits.detach().cpu().numpy())
            all_scores.append(_probabilities(logits, variant).detach().cpu().numpy())
            if components:
                latest = model.get_latest_components()
                if latest is None:
                    raise RuntimeError("Model did not expose representation components")
                for name, value in latest.items():
                    collected.setdefault(name, []).append(value.detach().cpu().numpy())
    return (
        np.concatenate(all_logits),
        np.concatenate(all_scores),
        {name: np.concatenate(values) for name, values in collected.items()},
    )


def train_refined_model(
    model: MMBrainXDL,
    data: BenchmarkData,
    split: dict[str, np.ndarray],
    device: torch.device,
    variant: RefinedVariant,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[MMBrainXDL, list[dict[str, Any]], int]:
    arrays = _arrays(data)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(split["train"], dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    logged_shapes = False
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        classification_total = 0.0
        reconstruction_total = 0.0
        seen = 0
        for (batch_indices,) in loader:
            optimizer.zero_grad(set_to_none=True)
            labels = arrays["labels"][batch_indices].to(device)
            inputs = _model_inputs(arrays, batch_indices, device)
            logits = model(*inputs)
            if not logged_shapes:
                components = model.get_latest_components() or {}
                shapes = {"input_fmri": tuple(inputs[0].shape), "input_eeg": tuple(inputs[1].shape),
                          "logits": tuple(logits.shape),
                          **{name: tuple(value.shape) for name, value in components.items()}}
                norms = {
                    name: float(value.detach().norm(dim=-1).mean().cpu())
                    for name, value in components.items()
                    if value.ndim == 2
                }
                print(f"[refined-mm shapes] {variant.name}: {shapes}")
                print(f"[refined-mm mean-l2-norms] {variant.name}: {norms}")
            if not torch.isfinite(logits).all():
                raise RuntimeError("Classifier logits contain NaN or Inf")
            classification_loss = _loss(logits, labels, variant)
            reconstruction_losses = model.get_latest_reconstruction_losses()
            if variant.graph_reconstruction_weight > 0:
                if not reconstruction_losses:
                    raise RuntimeError(
                        "Graph-autoencoder variant produced no reconstruction loss"
                    )
                reconstruction_loss = torch.stack(
                    list(reconstruction_losses.values())
                ).mean()
            else:
                reconstruction_loss = classification_loss.new_zeros(())
            batch_loss = (
                classification_loss
                + variant.graph_reconstruction_weight * reconstruction_loss
            )
            batch_loss.backward()
            if not logged_shapes:
                gradients = {
                    name: float(parameter.grad.detach().norm().cpu())
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None and (
                        name.startswith("residual_")
                        or name.startswith("fusion.")
                        or ".self" in name
                        or ".positive0" in name
                        or ".negative0" in name
                    )
                }
                print(f"[refined-mm gradient-norms] {variant.name}: {gradients}")
                if reconstruction_losses:
                    reconstruction_values = {
                        name: float(value.detach().cpu())
                        for name, value in reconstruction_losses.items()
                    }
                    print(
                        f"[refined-mm reconstruction-losses] {variant.name}: "
                        f"{reconstruction_values}"
                    )
                logged_shapes = True
            # Preserve the original participant-grouped neural training protocol.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(batch_loss.item()) * len(batch_indices)
            classification_total += float(classification_loss.item()) * len(batch_indices)
            reconstruction_total += float(reconstruction_loss.item()) * len(batch_indices)
            seen += len(batch_indices)
        _, val_scores, _ = _predict(
            model, arrays, split["val"], device, batch_size, variant, components=False
        )
        val_labels = data.labels[split["val"]]
        val_loss = float(
            compute_classification_metrics(
                val_labels, (val_scores >= 0.5).astype(int), val_scores
            )["log_loss"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / seen,
                "train_classification_loss": classification_total / seen,
                "train_reconstruction_loss": reconstruction_total / seen,
                "val_loss": val_loss,
            }
        )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _raw_representations(data: BenchmarkData) -> dict[str, np.ndarray]:
    fmri_tri = np.triu_indices(data.corr_fmri.shape[1], 1)
    eeg_tri = np.triu_indices(data.corr_eeg.shape[1], 1)
    fmri = data.corr_fmri[:, fmri_tri[0], fmri_tri[1]]
    eeg = data.corr_eeg[:, eeg_tri[0], eeg_tri[1]]
    return {"raw_fmri": fmri, "raw_eeg": eeg, "raw_combined": np.concatenate([fmri, eeg], axis=1)}


def _probe_rows(
    data: BenchmarkData,
    split: dict[str, np.ndarray],
    learned: dict[str, dict[str, np.ndarray]],
    *,
    fold: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = _raw_representations(data)
    for split_name in SPLITS:
        for block in range(3):
            learned[split_name][f"graph_block{block}"] = np.concatenate(
                [
                    learned[split_name][f"fmri_block{block}_summary"],
                    learned[split_name][f"eeg_block{block}_summary"],
                ],
                axis=1,
            )
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for representation in PROBE_REPRESENTATIONS:
        if representation.startswith("raw_"):
            values = {name: raw[representation][indices] for name, indices in split.items()}
        else:
            values = {name: learned[name][representation] for name in SPLITS}
        probe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed)),
            ]
        )
        probe.fit(values["train"], data.labels[split["train"]])
        scores = {name: probe.predict_proba(values[name])[:, 1] for name in SPLITS}
        threshold, objective = choose_binary_threshold(
            data.labels[split["val"]], scores["val"], "balanced_accuracy"
        )
        for split_name in SPLITS:
            metrics = metric_dict(data.labels[split[split_name]], scores[split_name], threshold)
            metric_rows.append(
                {"fold": fold, "representation": representation, "split": split_name,
                 "threshold": threshold, "validation_threshold_objective": objective, **metrics}
            )
        for split_name in ("val", "test"):
            for index, score in zip(split[split_name], scores[split_name], strict=True):
                prediction_rows.append(
                    {"fold": fold, "representation": representation, "split": split_name,
                     "sample_id": data.sample_ids[index], "participant_id": data.groups[index],
                     "session": data.sessions[index], "true_label": int(data.labels[index]),
                     "score": float(score), "threshold": threshold,
                     "predicted_class": int(score >= threshold)}
                )
    return metric_rows, prediction_rows


def diagnose_refined_checkpoint_fold(
    *,
    checkpoint_path: Path,
    data: BenchmarkData,
    manifest_path: Path,
    output_root: Path,
    fold: int,
    seed: int,
    device: torch.device,
    batch_size: int = 8,
) -> Path:
    """Probe a saved MM-A0 checkpoint without retraining or touching its outputs."""
    manifest = read_manifest(manifest_path)
    validate_manifest(manifest)
    split = fold_indices(manifest, fold)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(data.groups[split[left]]) & set(data.groups[split[right]])
        if overlap:
            raise RuntimeError(f"Participant leakage {left}/{right}: {sorted(overlap)}")
    variant = VARIANTS["mm_a0_original"]
    model = build_refined_model(data, variant).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    arrays = _arrays(data)
    extracted: dict[str, dict[str, np.ndarray]] = {}
    for split_name in SPLITS:
        _, _, extracted[split_name] = _predict(
            model, arrays, split[split_name], device, batch_size, variant, components=True
        )
    metric_rows, prediction_rows = _probe_rows(
        data, split, extracted, fold=fold, seed=seed + fold
    )
    destination = output_root / f"fold{fold:02d}"
    atomic_write_csv(destination / "linear_probe_metrics.csv", metric_rows)
    atomic_write_csv(destination / "linear_probe_predictions.csv", prediction_rows)
    atomic_write_json(
        destination / "protocol.json",
        {
            "fit": "training split only",
            "threshold": "validation balanced accuracy only",
            "test_role": "single held-out evaluation",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "manifest_sha256": file_sha256(manifest_path),
            "representations": list(PROBE_REPRESENTATIONS),
            "graph_block_summary": "node-wise mean, population SD, and maximum concatenated across fMRI and EEG",
            "retrained": False,
        },
    )
    return destination


def aggregate_probe_diagnostics(output_root: Path) -> Path:
    """Aggregate five checkpoint-only probe folds and their pooled OOF predictions."""
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = output_root / f"fold{fold:02d}"
        with (fold_dir / "linear_probe_metrics.csv").open(newline="", encoding="utf-8") as handle:
            metric_rows.extend(csv.DictReader(handle))
        with (fold_dir / "linear_probe_predictions.csv").open(newline="", encoding="utf-8") as handle:
            prediction_rows.extend(csv.DictReader(handle))
    summary: list[dict[str, Any]] = []
    from sklearn.metrics import balanced_accuracy_score
    key_map = {"accuracy": "acc", "roc_auc": "roc", "f1": "f1", "precision": "pre",
               "sensitivity": "sen", "specificity": "spe", "log_loss": "log_loss"}
    for representation in PROBE_REPRESENTATIONS:
        tests = [
            row for row in metric_rows
            if row["representation"] == representation and row["split"] == "test"
        ]
        predictions = [
            row for row in prediction_rows
            if row["representation"] == representation and row["split"] == "test"
        ]
        if len(tests) != 5 or len(predictions) != 120:
            raise RuntimeError(
                f"Incomplete diagnostic {representation}: {len(tests)} folds/{len(predictions)} OOF"
            )
        row: dict[str, Any] = {"representation": representation, "n_folds": 5, "n_oof": 120}
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in tests])
            row[f"fold_mean_{metric}"] = float(values.mean())
            row[f"fold_sd_{metric}"] = float(values.std(ddof=1))
        labels = np.asarray([int(item["true_label"]) for item in predictions])
        scores = np.asarray([float(item["score"]) for item in predictions])
        classes = np.asarray([int(item["predicted_class"]) for item in predictions])
        pooled = compute_classification_metrics(labels, classes, np.clip(scores, 1e-7, 1 - 1e-7))
        for metric, source in key_map.items():
            row[f"pooled_{metric}"] = float(pooled[source])
        row["pooled_balanced_accuracy"] = float(balanced_accuracy_score(labels, classes))
        summary.append(row)
    atomic_write_csv(output_root / "representation_diagnostics.csv", summary)
    atomic_write_csv(output_root / "linear_probe_fold_metrics.csv", metric_rows)
    atomic_write_csv(output_root / "linear_probe_predictions.csv", prediction_rows)
    return output_root / "representation_diagnostics.csv"


def _gate_rows(
    data: BenchmarkData,
    split_name: str,
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    fold: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flattened = weights.reshape(len(weights), 2, -1)
    for position, index in enumerate(indices):
        fmri = flattened[position, 0]
        eeg = flattened[position, 1]
        rows.append(
            {"fold": fold, "split": split_name, "sample_id": data.sample_ids[index],
             "participant_id": data.groups[index], "session": data.sessions[index],
             "true_label": int(data.labels[index]), "fmri_gate_mean": float(fmri.mean()),
             "fmri_gate_min": float(fmri.min()), "fmri_gate_max": float(fmri.max()),
             "eeg_gate_mean": float(eeg.mean()), "eeg_gate_min": float(eeg.min()),
             "eeg_gate_max": float(eeg.max())}
        )
    return rows


def run_refined_fold(
    *,
    variant_name: str,
    seed: int,
    fold: int,
    data: BenchmarkData,
    manifest_path: Path,
    output_root: Path,
    device: torch.device,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    diagnostics: bool = False,
    connectivity_skip_dim: int | None = None,
) -> Path:
    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown variant {variant_name}; choose from {sorted(VARIANTS)}")
    if diagnostics and variant_name != "mm_a0_original":
        raise ValueError("Representation probes are the Phase-1 MM-A0 diagnostic and may only run for mm_a0_original")
    variant = VARIANTS[variant_name]
    if connectivity_skip_dim is not None:
        if variant.connectivity_skip_dim == 0:
            raise ValueError("--connectivity-skip-dim is only valid for MM-A3/MM-A4")
        if connectivity_skip_dim not in {16, 32, 64}:
            raise ValueError("Connectivity skip dim must be one of 16, 32, or 64")
        variant = replace(variant, connectivity_skip_dim=connectivity_skip_dim)
    manifest = read_manifest(manifest_path)
    audit = validate_manifest(manifest)
    saved_manifest = output_root / "manifests" / "participant_grouped_folds.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    if saved_manifest.exists() and saved_manifest.read_text(encoding="utf-8") != manifest_text:
        raise RuntimeError(f"Refusing to replace a different refined manifest: {saved_manifest}")
    atomic_write_text(saved_manifest, manifest_text)
    split = fold_indices(manifest, fold)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(data.groups[split[left]]) & set(data.groups[split[right]])
        if overlap:
            raise RuntimeError(f"Participant leakage {left}/{right}: {sorted(overlap)}")
    seed_everything(seed + fold)
    fold_dir = output_root / variant_name / f"seed{seed}" / f"fold{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    initial_model = build_refined_model(data, variant)
    config = {
        "variant": asdict(variant), "seed": seed, "fold": fold, "epochs_requested": epochs,
        "patience": patience, "batch_size": batch_size, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "device": str(device), "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path), "split_indices": {k: v.tolist() for k, v in split.items()},
        "participant_split_audit": audit["folds"][f"outer{fold}_inner1"],
        "parameter_count": sum(parameter.numel() for parameter in initial_model.parameters()),
        "parameter_count_by_top_level_module": {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in initial_model.named_children()
        },
    }
    atomic_write_json(fold_dir / "config.json", config)
    started = time.time()
    model, history, best_epoch = train_refined_model(
        initial_model, data, split, device, variant, seed=seed + fold,
        epochs=epochs, patience=patience, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay,
    )
    temporary_checkpoint = fold_dir / ".checkpoint.pt.tmp"
    torch.save(model.state_dict(), temporary_checkpoint)
    os.replace(temporary_checkpoint, fold_dir / "checkpoint.pt")
    atomic_write_csv(fold_dir / "training_history.csv", history)
    arrays = _arrays(data)
    extracted: dict[str, dict[str, np.ndarray]] = {}
    predictions: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    gate_vectors: list[np.ndarray] = []
    gate_vector_splits: list[np.ndarray] = []
    gate_vector_indices: list[np.ndarray] = []
    scores_by_split: dict[str, np.ndarray] = {}
    for split_name in SPLITS:
        indices = split[split_name]
        logits, scores, components = _predict(
            model, arrays, indices, device, batch_size, variant, components=True
        )
        extracted[split_name] = components
        scores_by_split[split_name] = scores
        payload = {
            "indices": indices, "sample_ids": data.sample_ids[indices],
            "participant_ids": data.groups[indices], "session_ids": data.sessions[indices],
            "fold": np.repeat(fold, len(indices)), "seed": np.repeat(seed, len(indices)),
            "split": np.repeat(split_name, len(indices)), "y_true": data.labels[indices],
            "logits": logits, "probabilities": scores, **components,
        }
        _atomic_save_npz(fold_dir / "representations" / f"{split_name}.npz", **payload)
        if "gate_weights" in components:
            gates.extend(_gate_rows(data, split_name, indices, components["gate_weights"], fold=fold))
            gate_vectors.append(components["gate_weights"])
            gate_vector_splits.append(np.repeat(split_name, len(indices)))
            gate_vector_indices.append(indices)
    threshold, objective = choose_binary_threshold(
        data.labels[split["val"]], scores_by_split["val"], "balanced_accuracy"
    )
    fold_metrics: dict[str, Any] = {
        "variant": variant_name, "seed": seed, "fold": fold, "best_epoch": best_epoch,
        "epochs_completed": len(history), "threshold": threshold,
        "validation_threshold_objective": objective,
    }
    for split_name in SPLITS:
        metrics = metric_dict(data.labels[split[split_name]], scores_by_split[split_name], threshold)
        fold_metrics.update({f"{split_name}_{key}": value for key, value in metrics.items()})
        for index, score in zip(split[split_name], scores_by_split[split_name], strict=True):
            predictions.append(
                {"variant": variant_name, "seed": seed, "fold": fold, "split": split_name,
                 "sample_id": data.sample_ids[index], "participant_id": data.groups[index],
                 "session": data.sessions[index], "true_label": int(data.labels[index]),
                 "score": float(score), "threshold": threshold,
                 "predicted_class": int(score >= threshold)}
            )
    atomic_write_json(fold_dir / "fold_metrics.json", fold_metrics)
    atomic_write_csv(fold_dir / "fold_metrics.csv", [fold_metrics])
    atomic_write_csv(fold_dir / "predictions.csv", predictions)
    atomic_write_csv(fold_dir / "gate_values.csv", gates)
    atomic_write_json(
        fold_dir / "run_metadata.json",
        {"elapsed_seconds": time.time() - started, "python": sys.version,
         "platform": platform.platform(), "numpy": np.__version__, "sklearn": sklearn.__version__,
         "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
         "participant_intersections": {"train_val": [], "train_test": [], "val_test": []}},
    )
    if gates:
        collapse = {
            "n": len(gates),
            "fmri_gate_mean": float(np.mean([row["fmri_gate_mean"] for row in gates])),
            "fmri_gate_std": float(np.std([row["fmri_gate_mean"] for row in gates])),
            "eeg_gate_mean": float(np.mean([row["eeg_gate_mean"] for row in gates])),
            "eeg_gate_std": float(np.std([row["eeg_gate_mean"] for row in gates])),
            "fraction_near_boundary": float(np.mean([
                row["fmri_gate_mean"] < 0.05 or row["fmri_gate_mean"] > 0.95 for row in gates
            ])),
            "fraction_fmri_one_eeg_zero": float(np.mean([
                row["fmri_gate_mean"] > 0.95 and row["eeg_gate_mean"] < 0.05 for row in gates
            ])),
            "fraction_fmri_zero_eeg_one": float(np.mean([
                row["fmri_gate_mean"] < 0.05 and row["eeg_gate_mean"] > 0.95 for row in gates
            ])),
        }
        atomic_write_json(fold_dir / "gate_summary.json", collapse)
        condition_rows: list[dict[str, Any]] = []
        for split_name in SPLITS:
            for label in (0, 1):
                selected = [row for row in gates if row["split"] == split_name and row["true_label"] == label]
                if not selected:
                    continue
                condition_rows.append(
                    {"fold": fold, "split": split_name, "condition": "psilocybin" if label else "control",
                     "n": len(selected),
                     "fmri_gate_mean": float(np.mean([row["fmri_gate_mean"] for row in selected])),
                     "fmri_gate_sd": float(np.std([row["fmri_gate_mean"] for row in selected], ddof=1)) if len(selected) > 1 else 0.0,
                     "eeg_gate_mean": float(np.mean([row["eeg_gate_mean"] for row in selected])),
                     "eeg_gate_sd": float(np.std([row["eeg_gate_mean"] for row in selected], ddof=1)) if len(selected) > 1 else 0.0}
                )
        atomic_write_csv(fold_dir / "gate_by_condition.csv", condition_rows)
        if variant.gate_type == "featurewise":
            vectors = np.concatenate(gate_vectors)
            vector_indices = np.concatenate(gate_vector_indices)
            _atomic_save_npz(
                fold_dir / "featurewise_gate_vectors.npz",
                gate_vectors=vectors,
                gate_mean_vector=vectors.mean(axis=0),
                gate_variance_vector=vectors.var(axis=0),
                split=np.concatenate(gate_vector_splits),
                indices=vector_indices,
                sample_ids=data.sample_ids[vector_indices],
                participant_ids=data.groups[vector_indices],
                labels=data.labels[vector_indices],
            )
    if diagnostics:
        metric_rows, probe_predictions = _probe_rows(data, split, extracted, fold=fold, seed=seed + fold)
        diagnostics_dir = output_root / "diagnostics" / variant_name / f"seed{seed}" / f"fold{fold:02d}"
        atomic_write_csv(diagnostics_dir / "linear_probe_metrics.csv", metric_rows)
        atomic_write_csv(diagnostics_dir / "linear_probe_predictions.csv", probe_predictions)
        atomic_write_json(
            diagnostics_dir / "protocol.json",
            {"fit": "training split only", "threshold": "validation balanced accuracy only",
             "test_role": "single held-out evaluation", "representations": list(PROBE_REPRESENTATIONS)},
        )
    return fold_dir


def aggregate_refined_variant(output_root: Path, variant_name: str, seed: int) -> Path:
    base = output_root / variant_name / f"seed{seed}"
    fold_rows: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    all_gates: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = base / f"fold{fold:02d}"
        if not (fold_dir / "fold_metrics.json").is_file():
            raise FileNotFoundError(f"Cannot aggregate incomplete experiment; missing {fold_dir}")
        fold_rows.append(json.loads((fold_dir / "fold_metrics.json").read_text(encoding="utf-8")))
        with (fold_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
            oof.extend(row for row in csv.DictReader(handle) if row["split"] == "test")
        if (fold_dir / "gate_values.csv").is_file():
            with (fold_dir / "gate_values.csv").open(newline="", encoding="utf-8") as handle:
                all_gates.extend(csv.DictReader(handle))
    sample_ids = [row["sample_id"] for row in oof]
    if len(oof) != 120 or len(set(sample_ids)) != 120:
        raise RuntimeError(f"Expected exactly 120 unique OOF predictions, got {len(oof)}/{len(set(sample_ids))}")
    atomic_write_csv(base / "fold_metrics.csv", fold_rows)
    atomic_write_csv(base / "oof_predictions.csv", sorted(oof, key=lambda row: row["sample_id"]))
    summary: dict[str, Any] = {"variant": variant_name, "seed": seed, "n_folds": 5, "n_oof": 120}
    for metric in METRICS:
        values = np.asarray([float(row[f"test_{metric}"]) for row in fold_rows])
        summary[f"fold_mean_{metric}"] = float(values.mean())
        summary[f"fold_sd_{metric}"] = float(values.std(ddof=1))
    labels = np.asarray([int(row["true_label"]) for row in oof])
    scores = np.asarray([float(row["score"]) for row in oof])
    predictions = np.asarray([int(row["predicted_class"]) for row in oof])
    # Per-fold validation-selected thresholds are already reflected in predictions.
    from sklearn.metrics import balanced_accuracy_score
    pooled = compute_classification_metrics(labels, predictions, np.clip(scores, 1e-7, 1 - 1e-7))
    key_map = {"accuracy": "acc", "roc_auc": "roc", "f1": "f1", "precision": "pre",
               "sensitivity": "sen", "specificity": "spe", "log_loss": "log_loss"}
    for metric, source in key_map.items():
        summary[f"pooled_{metric}"] = float(pooled[source])
    summary["pooled_balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))
    atomic_write_json(base / "summary.json", summary)
    atomic_write_csv(base / "summary.csv", [summary])
    if all_gates:
        atomic_write_csv(base / "gate_values.csv", all_gates)
        gate_summary: list[dict[str, Any]] = []
        for split_name in SPLITS:
            for label in (0, 1):
                selected = [
                    row for row in all_gates
                    if row["split"] == split_name and int(row["true_label"]) == label
                ]
                if not selected:
                    continue
                fmri = np.asarray([float(row["fmri_gate_mean"]) for row in selected])
                eeg = np.asarray([float(row["eeg_gate_mean"]) for row in selected])
                gate_summary.append(
                    {"split": split_name, "condition": "psilocybin" if label else "control", "n": len(selected),
                     "fmri_gate_mean": float(fmri.mean()), "fmri_gate_sd": float(fmri.std(ddof=1)),
                     "eeg_gate_mean": float(eeg.mean()), "eeg_gate_sd": float(eeg.std(ddof=1)),
                     "fraction_near_boundary": float(np.mean((fmri < 0.05) | (fmri > 0.95)))}
                )
        atomic_write_csv(base / "gate_summary_by_condition.csv", gate_summary)
    diagnostics_base = output_root / "diagnostics" / variant_name / f"seed{seed}"
    if diagnostics_base.is_dir():
        probe_metrics: list[dict[str, Any]] = []
        probe_predictions: list[dict[str, Any]] = []
        for fold in range(1, 6):
            fold_base = diagnostics_base / f"fold{fold:02d}"
            with (fold_base / "linear_probe_metrics.csv").open(newline="", encoding="utf-8") as handle:
                probe_metrics.extend(csv.DictReader(handle))
            with (fold_base / "linear_probe_predictions.csv").open(newline="", encoding="utf-8") as handle:
                probe_predictions.extend(csv.DictReader(handle))
        atomic_write_csv(diagnostics_base / "linear_probe_fold_metrics.csv", probe_metrics)
        atomic_write_csv(diagnostics_base / "linear_probe_predictions.csv", probe_predictions)
        probe_summary: list[dict[str, Any]] = []
        for representation in PROBE_REPRESENTATIONS:
            test_rows = [
                row for row in probe_metrics
                if row["representation"] == representation and row["split"] == "test"
            ]
            test_predictions = [
                row for row in probe_predictions
                if row["representation"] == representation and row["split"] == "test"
            ]
            row: dict[str, Any] = {"representation": representation, "n_folds": len(test_rows)}
            for metric in METRICS:
                values = np.asarray([float(item[metric]) for item in test_rows])
                row[f"fold_mean_{metric}"] = float(values.mean())
                row[f"fold_sd_{metric}"] = float(values.std(ddof=1))
            labels_probe = np.asarray([int(item["true_label"]) for item in test_predictions])
            scores_probe = np.asarray([float(item["score"]) for item in test_predictions])
            predictions_probe = np.asarray([int(item["predicted_class"]) for item in test_predictions])
            from sklearn.metrics import balanced_accuracy_score
            pooled_probe = compute_classification_metrics(
                labels_probe, predictions_probe, np.clip(scores_probe, 1e-7, 1 - 1e-7)
            )
            for metric, source in key_map.items():
                row[f"pooled_{metric}"] = float(pooled_probe[source])
            row["pooled_balanced_accuracy"] = float(balanced_accuracy_score(labels_probe, predictions_probe))
            probe_summary.append(row)
        atomic_write_csv(diagnostics_base / "linear_probe_summary.csv", probe_summary)
    return base
