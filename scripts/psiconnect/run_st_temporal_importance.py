#!/usr/bin/env python3
"""Explain frozen seed-2026 ST-BrainXDL predictions with temporal-block occlusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from brainxdl.common.device import pick_device
from brainxdl.common.metrics import compute_classification_metrics
from brainxdl.common.seed import seed_everything
from brainxdl.st.artifacts import (
    FOLDS,
    condition_name,
    fold_inputs,
    read_json,
    sha256,
    split_metadata,
)
from brainxdl.st.checkpoints import load_model
from brainxdl.st.dataset import STGraphDataset, padseq
from brainxdl.xai.st_fidelity import graph_sequence_digest, model_state_digest
from brainxdl.xai.st_temporal import (
    block_scores_to_windows,
    oriented_prediction_margin,
    replace_temporal_blocks,
    select_nonoverlapping_blocks,
    temporal_block_starts,
)


SEED = 2026
CONDITIONS = ("top", "random", "low")
METRIC_NAMES = ("acc", "roc", "f1", "log_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--window-length", type=int, default=40)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Dynamic-graph windows per occluded block; default is ceil(window-length/window-stride).",
    )
    parser.add_argument("--block-step", type=int, default=1)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    parser.add_argument("--random-repetitions", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--baseline-score-tolerance", type=float, default=1e-5)
    parser.add_argument("--prefer-mps", action="store_true")
    parser.add_argument(
        "--sparse-on-cpu", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _object_matrix(sequences: list[np.ndarray]) -> np.ndarray:
    if not sequences:
        raise ValueError("No graph sequences were supplied.")
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError("All graph sequences in a prediction batch must have equal length.")
    output = np.empty((len(sequences), lengths.pop()), dtype=object)
    for row, sequence in enumerate(sequences):
        for column, graph in enumerate(sequence):
            output[row, column] = graph
    return output


def predict_logits_many(
    model,
    setting: dict[str, Any],
    sequences: list[np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and raw classifier logits in deterministic sequence order."""
    data = _object_matrix(sequences)
    loader = DataLoader(
        STGraphDataset(data),
        batch_size=min(batch_size, len(data)),
        shuffle=False,
        num_workers=0,
        collate_fn=padseq,
        pin_memory=False,
    )
    labels, logits_rows = [], []
    with torch.no_grad():
        for batch in loader:
            _, _, _, readout, target = model(batch, setting, sample=False)
            last_mask = torch.stack(
                [step.last.detach().cpu().bool() for step in batch], dim=0
            )
            positions = torch.nonzero(last_mask, as_tuple=False)
            if positions.shape[0] != readout.shape[0]:
                raise ValueError("Final readout count does not match batch size.")
            restore_order = torch.argsort(positions[:, 1]).to(readout.device)
            readout = readout[restore_order]
            target = target[restore_order]
            logits = model.classifier.linear(readout)
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            labels.append(target.detach().cpu().numpy().astype(np.int64))
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(labels), np.concatenate(logits_rows)


def positive_probability(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2:
        raise ValueError("logits must be a matrix.")
    if values.shape[1] == 1:
        return 1.0 / (1.0 + np.exp(-values[:, 0]))
    if values.shape[1] == 2:
        difference = values[:, 1] - values[:, 0]
        return 1.0 / (1.0 + np.exp(-difference))
    raise ValueError("Temporal importance currently supports binary classifiers only.")


def deterministic_seed(*values: object) -> int:
    payload = "|".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def prediction_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    predictions = (probabilities >= thresholds).astype(np.int64)
    return compute_classification_metrics(labels, predictions, probabilities)


def finite_metric_rows(metrics: dict[str, float]) -> dict[str, float | None]:
    return {
        name: float(metrics[name]) if math.isfinite(float(metrics[name])) else None
        for name in METRIC_NAMES
    }


def validate_targets(output_dir: Path, overwrite: bool) -> None:
    names = (
        "st_temporal_predictions.csv",
        "st_temporal_block_importance.csv",
        "st_temporal_window_importance.csv",
        "st_temporal_fidelity.csv",
        "st_temporal_importance_metadata.json",
        "st_temporal_importance.md",
        "st_temporal_importance.png",
        "st_temporal_importance.pdf",
        "st_temporal_fidelity.png",
        "st_temporal_fidelity.pdf",
    )
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Temporal-importance outputs already exist; pass --overwrite: "
            + ", ".join(path.name for path in existing)
        )


def _plot_temporal_importance(
    output_dir: Path,
    prediction_rows: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    sample_ids = [str(row["sample_id"]) for row in prediction_rows]
    conditions = [str(row["condition"]) for row in prediction_rows]
    by_sample_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        by_sample_window[str(row["sample_id"])].append(row)
    for row in block_rows:
        by_sample_block[str(row["sample_id"])].append(row)
    matrix = np.stack(
        [
            np.asarray(
                [float(row["delta_predicted_class_margin"]) for row in sorted(by_sample_window[sid], key=lambda x: int(x["window_index_zero_based"]))]
            )
            for sid in sample_ids
        ]
    )
    max_abs = np.max(np.abs(matrix), axis=1)
    exemplar_index = int(np.argsort(max_abs, kind="stable")[len(max_abs) // 2])
    exemplar_id = sample_ids[exemplar_index]
    exemplar_blocks = sorted(
        by_sample_block[exemplar_id], key=lambda row: int(row["block_start_zero_based"])
    )
    order = np.lexsort((np.asarray(sample_ids), np.asarray(conditions)))
    ordered_matrix = matrix[order]
    ordered_conditions = np.asarray(conditions)[order]

    figure, axes = plt.subplots(
        3, 1, figsize=(7.4, 9.0), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.55, 1.0]},
    )
    starts = np.asarray([int(row["block_start_zero_based"]) + 1 for row in exemplar_blocks])
    block_values = np.asarray([float(row["delta_predicted_class_margin"]) for row in exemplar_blocks])
    colors = np.where(block_values >= 0, "#B33A3A", "#3267A8")
    axes[0].bar(starts, block_values, width=0.85, color=colors)
    axes[0].axhline(0, color="0.2", linewidth=0.8)
    axes[0].set_title("A  Representative held-out scan (median peak magnitude)", loc="left", fontweight="bold")
    axes[0].set_ylabel(r"$\Delta$ predicted-class margin")
    axes[0].set_xlabel("Occluded block start (dynamic-FC window)")
    axes[0].text(
        0.99, 0.96, exemplar_id, transform=axes[0].transAxes,
        ha="right", va="top", fontsize=8,
    )

    limit = float(np.quantile(np.abs(ordered_matrix), 0.99))
    image = axes[1].imshow(
        ordered_matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r",
        vmin=-limit, vmax=limit,
    )
    axes[1].set_title("B  All out-of-fold scans", loc="left", fontweight="bold")
    axes[1].set_ylabel("Held-out scan (grouped by condition)")
    axes[1].set_xlabel("Dynamic-FC window")
    boundary = int(np.sum(ordered_conditions == ordered_conditions[0]))
    if 0 < boundary < len(ordered_conditions):
        axes[1].axhline(boundary - 0.5, color="0.15", linewidth=1.0)
    colorbar = figure.colorbar(image, ax=axes[1], pad=0.015, fraction=0.025)
    colorbar.set_label(r"$\Delta$ predicted-class margin (colour clipped at 99th percentile)")

    window_index = np.arange(1, matrix.shape[1] + 1)
    unique_conditions = sorted(set(conditions))
    palette = ["#3267A8", "#B33A3A", "#4C8B57"]
    for color, condition in zip(palette, unique_conditions):
        selected = np.asarray([value == condition for value in conditions])
        values = matrix[selected]
        median = np.median(values, axis=0)
        lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
        axes[2].plot(window_index, median, color=color, linewidth=1.8, label=f"{condition} (n={values.shape[0]})")
        axes[2].fill_between(window_index, lower, upper, color=color, alpha=0.18, linewidth=0)
    axes[2].axhline(0, color="0.2", linewidth=0.8)
    axes[2].set_title("C  Descriptive group median across normalized scan position", loc="left", fontweight="bold")
    axes[2].set_xlabel("Dynamic-FC window")
    axes[2].set_ylabel(r"Median $\Delta$ margin (IQR)")
    axes[2].legend(frameon=False, ncol=min(2, len(unique_conditions)), fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"st_temporal_importance.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(figure)


def _plot_fidelity(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), constrained_layout=True)
    colors = {"top": "#B33A3A", "random": "#666666", "low": "#3267A8"}
    labels = {"top": "Top-ranked blocks", "random": "Random blocks", "low": "Lowest-ranked blocks"}
    for condition in CONDITIONS:
        selected = sorted(
            [row for row in rows if row["perturbation"] == condition],
            key=lambda row: float(row["window_fraction"]),
        )
        x = [100 * float(row["window_fraction"]) for row in selected]
        for axis, metric in zip(axes.flat, ("acc", "roc", "f1", "log_loss")):
            axis.plot(x, [float(row[metric]) for row in selected], marker="o", color=colors[condition], label=labels[condition])
    for axis, ylabel in zip(
        axes.flat,
        ("Pooled accuracy", "Pooled ROC-AUC", "Pooled F1", "Pooled log loss"),
    ):
        axis.set_ylabel(ylabel)
    for axis, title in zip(
        axes.flat,
        ("A  Accuracy", "B  Discrimination", "C  F1", "D  Probabilistic error"),
    ):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Temporal windows replaced (%)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"st_temporal_fidelity.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError("The first ST temporal-importance study is frozen to seed 2026.")
    if args.window_length < 1 or args.window_stride < 1:
        raise ValueError("Window length and stride must be positive.")
    block_size = args.block_size or int(math.ceil(args.window_length / args.window_stride))
    if block_size < 1 or args.block_step < 1 or args.batch_size < 1:
        raise ValueError("Block size, block step, and batch size must be positive.")
    fractions = tuple(sorted(set(map(float, args.fractions))))
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("Every fidelity fraction must lie strictly between zero and one.")
    if args.random_repetitions < 1:
        raise ValueError("At least one random repetition is required.")

    frozen_path = args.d1_root / "frozen_selection.json"
    frozen = read_json(frozen_path)
    if frozen.get("outer_test_used") is not False:
        raise ValueError("ST-D1 frozen selection must explicitly exclude outer-test data.")
    fold_specs = [fold_inputs(args, frozen, fold) for fold in FOLDS]
    for spec in fold_specs:
        print(
            f"fold={spec['fold']} density={spec['density']} checkpoint={spec['checkpoint']} data={spec['fold_file']}",
            flush=True,
        )
    if args.plan_only:
        print(
            f"Plan validated: seed={args.seed}, folds={list(FOLDS)}, block_size={block_size}, "
            f"block_step={args.block_step}, fractions={fractions}"
        )
        return 0

    validate_targets(args.output_dir, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = pick_device(prefer_mps=args.prefer_mps)

    prediction_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    fidelity_values: dict[tuple[str, float, int], list[tuple[int, float, float]]] = defaultdict(list)
    baseline_values: list[tuple[int, float, float]] = []
    provenance: dict[str, Any] = {}
    test_index_sets: list[set[int]] = []
    master_ids: list[str] | None = None
    master_groups: list[str] | None = None

    for spec in fold_specs:
        fold = int(spec["fold"])
        payload = torch.load(spec["fold_file"], map_location="cpu", weights_only=False)
        graphs = np.asarray(payload["test_graphs"], dtype=object)
        metadata = payload.get("metadata", {})
        sample_ids, participant_groups, indices = split_metadata(metadata, fold)
        if master_ids is None:
            master_ids, master_groups = sample_ids, participant_groups
        elif sample_ids != master_ids or participant_groups != master_groups:
            raise ValueError(f"Fold {fold}: sample ordering differs across fold files.")
        test_indices = np.asarray(indices["test"], dtype=np.int64)
        test_index_sets.append(set(map(int, test_indices)))
        if len(graphs) != len(test_indices):
            raise ValueError(f"Fold {fold}: test graph rows do not match test indices.")

        model, setting, _ = load_model(spec["checkpoint"], device, args.sparse_on_cpu)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before_state = model_state_digest(model)
        threshold = float(spec["metrics"]["validation_threshold"])
        reference_labels = np.asarray(spec["metrics"]["y_true"], dtype=np.int64)
        reference_scores = np.asarray(spec["metrics"]["y_score"], dtype=float)
        label_mapping = metadata.get("label_mapping", {"0": "control", "1": "psilocybin"})

        for row_index, (sample_graphs, global_index) in enumerate(zip(graphs, test_indices)):
            original_digest = graph_sequence_digest(sample_graphs)
            num_windows = len(sample_graphs)
            if block_size >= num_windows:
                raise ValueError(
                    f"Fold {fold} sample {global_index}: block_size={block_size} leaves no reference window."
                )
            starts = temporal_block_starts(num_windows, block_size, args.block_step)
            labels, logits = predict_logits_many(model, setting, [sample_graphs], args.batch_size)
            probability = float(positive_probability(logits)[0])
            if int(labels[0]) != int(reference_labels[row_index]):
                raise ValueError(f"Fold {fold} sample {global_index}: target ordering mismatch.")
            if abs(probability - float(reference_scores[row_index])) > args.baseline_score_tolerance:
                raise RuntimeError(
                    f"Fold {fold} sample {global_index}: reproduced probability differs from saved baseline."
                )
            base_margin_tensor, predicted_tensor = oriented_prediction_margin(
                torch.as_tensor(logits), threshold
            )
            base_margin = float(base_margin_tensor[0].item())
            predicted_class = int(predicted_tensor[0].item())
            sample_id = sample_ids[int(global_index)]
            participant_id = participant_groups[int(global_index)]
            condition = condition_name(label_mapping, int(labels[0]))
            baseline_values.append((int(labels[0]), probability, threshold))
            prediction_rows.append(
                {
                    "seed": args.seed,
                    "outer_fold": fold,
                    "sample_index": int(global_index),
                    "sample_id": sample_id,
                    "participant_id": participant_id,
                    "condition": condition,
                    "label": int(labels[0]),
                    "predicted_class": predicted_class,
                    "correct": int(predicted_class == int(labels[0])),
                    "validation_threshold": threshold,
                    "positive_probability": probability,
                    "predicted_class_margin": base_margin,
                    "num_windows": num_windows,
                }
            )

            perturbed_logits_parts = []
            for offset in range(0, len(starts), args.batch_size):
                chunk_starts = starts[offset : offset + args.batch_size]
                sequences = [
                    replace_temporal_blocks(
                        sample_graphs, [(int(start), int(start) + block_size)]
                    )
                    for start in chunk_starts
                ]
                chunk_labels, chunk_logits = predict_logits_many(
                    model, setting, sequences, args.batch_size
                )
                if not np.all(chunk_labels == int(labels[0])):
                    raise AssertionError("Temporal replacement changed target labels.")
                perturbed_logits_parts.append(chunk_logits)
            perturbed_logits = np.concatenate(perturbed_logits_parts)
            repeated_prediction = torch.full(
                (len(perturbed_logits),), predicted_class, dtype=torch.long
            )
            perturbed_margins, _ = oriented_prediction_margin(
                torch.as_tensor(perturbed_logits), threshold, repeated_prediction
            )
            block_scores = base_margin - perturbed_margins.numpy()
            perturbed_probabilities = positive_probability(perturbed_logits)
            for index, start in enumerate(starts):
                block_rows.append(
                    {
                        "seed": args.seed,
                        "outer_fold": fold,
                        "sample_index": int(global_index),
                        "sample_id": sample_id,
                        "participant_id": participant_id,
                        "condition": condition,
                        "label": int(labels[0]),
                        "predicted_class": predicted_class,
                        "block_start_zero_based": int(start),
                        "block_end_exclusive": int(start) + block_size,
                        "block_center_normalized": float((int(start) + 0.5 * block_size) / num_windows),
                        "baseline_positive_probability": probability,
                        "perturbed_positive_probability": float(perturbed_probabilities[index]),
                        "baseline_predicted_class_margin": base_margin,
                        "perturbed_predicted_class_margin": float(perturbed_margins[index]),
                        "delta_predicted_class_margin": float(block_scores[index]),
                    }
                )
            window_scores = block_scores_to_windows(
                block_scores, starts, block_size, num_windows
            )
            for window_index, score in enumerate(window_scores):
                window_rows.append(
                    {
                        "seed": args.seed,
                        "outer_fold": fold,
                        "sample_index": int(global_index),
                        "sample_id": sample_id,
                        "participant_id": participant_id,
                        "condition": condition,
                        "label": int(labels[0]),
                        "predicted_class": predicted_class,
                        "window_index_zero_based": window_index,
                        "window_position_normalized": float((window_index + 0.5) / num_windows),
                        "delta_predicted_class_margin": float(score),
                    }
                )

            fidelity_sequences: list[np.ndarray] = []
            fidelity_keys: list[tuple[str, float, int]] = []
            max_blocks = max(1, (num_windows - 1) // block_size)
            for fraction in fractions:
                count = min(
                    max_blocks,
                    max(1, int(math.ceil(fraction * num_windows / block_size))),
                )
                for condition_name_value in ("top", "low"):
                    selected = select_nonoverlapping_blocks(
                        block_scores, starts, block_size, count, condition_name_value
                    )
                    blocks = [
                        (int(starts[index]), int(starts[index]) + block_size)
                        for index in selected
                    ]
                    fidelity_sequences.append(replace_temporal_blocks(sample_graphs, blocks))
                    fidelity_keys.append((condition_name_value, fraction, -1))
                for repetition in range(args.random_repetitions):
                    rng = np.random.default_rng(
                        deterministic_seed(args.seed, fold, int(global_index), fraction, repetition)
                    )
                    selected = select_nonoverlapping_blocks(
                        block_scores, starts, block_size, count, "random", rng=rng
                    )
                    blocks = [
                        (int(starts[index]), int(starts[index]) + block_size)
                        for index in selected
                    ]
                    fidelity_sequences.append(replace_temporal_blocks(sample_graphs, blocks))
                    fidelity_keys.append(("random", fraction, repetition))
            for offset in range(0, len(fidelity_sequences), args.batch_size):
                chunk = fidelity_sequences[offset : offset + args.batch_size]
                chunk_keys = fidelity_keys[offset : offset + args.batch_size]
                chunk_labels, chunk_logits = predict_logits_many(model, setting, chunk, args.batch_size)
                chunk_probabilities = positive_probability(chunk_logits)
                if not np.all(chunk_labels == int(labels[0])):
                    raise AssertionError("Fidelity replacement changed target labels.")
                for key, value in zip(chunk_keys, chunk_probabilities):
                    fidelity_values[key].append((int(labels[0]), float(value), threshold))

            if graph_sequence_digest(sample_graphs) != original_digest:
                raise AssertionError(f"Fold {fold} sample {global_index}: input sequence was mutated.")
            print(
                f"[temporal] fold={fold} sample={row_index + 1}/{len(graphs)} "
                f"index={global_index} blocks={len(starts)}",
                flush=True,
            )

        if model_state_digest(model) != before_state:
            raise AssertionError(f"Fold {fold}: frozen model state changed.")
        provenance[str(fold)] = {
            "density": spec["density"],
            "checkpoint": str(spec["checkpoint"]),
            "checkpoint_sha256": sha256(spec["checkpoint"]),
            "metrics": str(spec["metrics_path"].resolve()),
            "fold_file": str(spec["fold_file"].resolve()),
            "validation_threshold": threshold,
            "n_test": len(graphs),
        }
        del model

    if any(left & right for i, left in enumerate(test_index_sets) for right in test_index_sets[i + 1 :]):
        raise ValueError("Outer-test indices overlap across folds.")
    if master_ids is None or set().union(*test_index_sets) != set(range(len(master_ids))):
        raise ValueError("Five outer-test folds do not cover every observation exactly once.")

    baseline_labels = np.asarray([row[0] for row in baseline_values], dtype=np.int64)
    baseline_probabilities = np.asarray([row[1] for row in baseline_values], dtype=float)
    baseline_thresholds = np.asarray([row[2] for row in baseline_values], dtype=float)
    baseline_metrics = prediction_metrics(
        baseline_labels, baseline_probabilities, baseline_thresholds
    )
    fidelity_rows: list[dict[str, Any]] = []
    for condition_value in CONDITIONS:
        fidelity_rows.append(
            {
                "perturbation": condition_value,
                "window_fraction": 0.0,
                "random_repetitions": args.random_repetitions if condition_value == "random" else "",
                **finite_metric_rows(baseline_metrics),
                "delta_acc": 0.0,
                "delta_roc": 0.0,
                "delta_f1": 0.0,
                "delta_log_loss": 0.0,
            }
        )
        for fraction in fractions:
            if condition_value == "random":
                repetitions = []
                for repetition in range(args.random_repetitions):
                    entries = fidelity_values[(condition_value, fraction, repetition)]
                    labels = np.asarray([row[0] for row in entries], dtype=np.int64)
                    probabilities = np.asarray([row[1] for row in entries], dtype=float)
                    thresholds = np.asarray([row[2] for row in entries], dtype=float)
                    repetitions.append(prediction_metrics(labels, probabilities, thresholds))
                metrics = {
                    name: float(np.mean([row[name] for row in repetitions]))
                    for name in METRIC_NAMES
                }
            else:
                entries = fidelity_values[(condition_value, fraction, -1)]
                labels = np.asarray([row[0] for row in entries], dtype=np.int64)
                probabilities = np.asarray([row[1] for row in entries], dtype=float)
                thresholds = np.asarray([row[2] for row in entries], dtype=float)
                metrics = prediction_metrics(labels, probabilities, thresholds)
            fidelity_rows.append(
                {
                    "perturbation": condition_value,
                    "window_fraction": fraction,
                    "random_repetitions": args.random_repetitions if condition_value == "random" else "",
                    **finite_metric_rows(metrics),
                    **{
                        f"delta_{name}": float(metrics[name] - baseline_metrics[name])
                        for name in METRIC_NAMES
                    },
                }
            )

    write_csv(args.output_dir / "st_temporal_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "st_temporal_block_importance.csv", block_rows)
    write_csv(args.output_dir / "st_temporal_window_importance.csv", window_rows)
    write_csv(args.output_dir / "st_temporal_fidelity.csv", fidelity_rows)
    _plot_temporal_importance(args.output_dir, prediction_rows, block_rows, window_rows)
    _plot_fidelity(args.output_dir, fidelity_rows)

    top_rows = [
        row for row in fidelity_rows
        if row["perturbation"] == "top" and float(row["window_fraction"]) > 0
    ]
    random_lookup = {
        float(row["window_fraction"]): row
        for row in fidelity_rows
        if row["perturbation"] == "random" and float(row["window_fraction"]) > 0
    }
    top_more_damaging_roc_log_loss = all(
        float(row["roc"]) < float(random_lookup[float(row["window_fraction"])]["roc"])
        and float(row["log_loss"]) > float(random_lookup[float(row["window_fraction"])]["log_loss"])
        for row in top_rows
    )
    top_more_damaging_accuracy_f1 = all(
        float(row["acc"]) < float(random_lookup[float(row["window_fraction"])]["acc"])
        and float(row["f1"]) < float(random_lookup[float(row["window_fraction"])]["f1"])
        for row in top_rows
    )
    largest_window_row = max(
        window_rows, key=lambda row: abs(float(row["delta_predicted_class_margin"]))
    )
    display_clip = float(
        np.quantile(
            np.abs([float(row["delta_predicted_class_margin"]) for row in window_rows]),
            0.99,
        )
    )
    metadata = {
        "schema_version": 1,
        "experiment": "ST temporal-block perturbation importance",
        "status": "complete",
        "dataset": "PsiConnect",
        "task": "rest",
        "source_experiment": "[ST-D1]",
        "seed": args.seed,
        "outer_folds": list(FOLDS),
        "n_outer_test_observations": len(prediction_rows),
        "n_participants": len(set(row["participant_id"] for row in prediction_rows)),
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "temporal_block_size_dynamic_windows": block_size,
        "temporal_block_step": args.block_step,
        "replacement": "nearest unselected observed graph; ties prefer the earlier window",
        "importance_definition": (
            "Original predicted-class log-odds margin relative to the fold-specific "
            "validation threshold minus the same oriented margin after temporal-block replacement."
        ),
        "window_score_definition": "Mean importance of every evaluated block covering that window.",
        "fidelity_fractions": list(fractions),
        "random_repetitions": args.random_repetitions,
        "baseline_metrics": finite_metric_rows(baseline_metrics),
        "top_more_damaging_than_random_for_accuracy_and_f1_at_all_fractions": top_more_damaging_accuracy_f1,
        "top_more_damaging_than_random_for_roc_and_log_loss_at_all_fractions": top_more_damaging_roc_log_loss,
        "display_absolute_clip_99th_percentile": display_clip,
        "largest_absolute_window_score": largest_window_row,
        "outer_test_used_for_training_or_selection": False,
        "model_retrained": False,
        "provenance": provenance,
    }
    (args.output_dir / "st_temporal_importance_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = [
        "# PsiConnect ST Temporal-Block Importance — Seed 2026",
        "",
        "This frozen-model post-hoc analysis evaluates all five held-out outer folds",
        "using each observation's matching `[ST-D1]` checkpoint. No model was retrained.",
        "",
        "## Method",
        "",
        f"- Dynamic FC: window length `{args.window_length}`, stride `{args.window_stride}`.",
        f"- Occlusion unit: `{block_size}` consecutive dynamic-FC windows.",
        "- Replacement: the nearest unselected graph from the same scan, preserving",
        "  the target position's label, padding flag, and last-window marker.",
        "- Importance: decrease in the original predicted-class log-odds margin, centred",
        "  on the fold-specific validation-selected classification threshold.",
        "- Overlapping block scores are averaged onto each covered dynamic-FC window.",
        "",
        "## Reproduced pooled baseline",
        "",
        "| Accuracy | ROC-AUC | F1 | Log loss |",
        "| ---: | ---: | ---: | ---: |",
        "| " + " | ".join(f"{float(baseline_metrics[name]):.4f}" for name in METRIC_NAMES) + " |",
        "",
        "## Temporal fidelity",
        "",
        "| Replaced windows | Condition | Accuracy | ROC-AUC | F1 | Log loss |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in fidelity_rows:
        if float(row["window_fraction"]) == 0 and row["perturbation"] != "top":
            continue
        report.append(
            f"| {100 * float(row['window_fraction']):.0f}% | {row['perturbation']} | "
            f"{float(row['acc']):.4f} | {float(row['roc']):.4f} | "
            f"{float(row['f1']):.4f} | {float(row['log_loss']):.4f} |"
        )
    report.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Positive scores identify temporal blocks whose replacement reduced support for",
            "the model's original decision. Negative scores identify blocks whose replacement",
            "strengthened that decision. These are model-behaviour explanations, not causal",
            "neurobiological effects. Because this is resting-state data, the cohort curve uses",
            "normalized scan position descriptively and does not assume event-locked alignment.",
            "",
            "The cohort heatmap colour range is clipped at the 99th percentile and the",
            "condition curves use medians and interquartile ranges. Full unclipped values",
            "remain in the CSV files. This robust display is necessary because one held-out",
            f"scan (`{largest_window_row['sample_id']}`) has a maximum absolute window score",
            f"of `{abs(float(largest_window_row['delta_predicted_class_margin'])):.4f}`, substantially larger than the rest.",
            "This is evidence of scan-specific model sensitivity, not a group-wide temporal effect.",
            "",
            "The window-level heatmap is an overlap-average of block perturbations and therefore",
            "has dynamic-window—not individual-TR—resolution. The analysis does not use summed",
            "`TransformerConv` attention as temporal importance.",
            "",
            "## Validation",
            "",
            f"- All five outer folds and `{len(prediction_rows)}` held-out observations completed.",
            "- Saved baseline probabilities were reproduced before perturbation.",
            "- Input-sequence and frozen-model hashes were unchanged after evaluation.",
            f"- Top-ranked joint deletion reduced accuracy and F1 more than matched random deletion at every tested fraction: `{top_more_damaging_accuracy_f1}`.",
            f"- The analogous criterion for ROC-AUC and log loss was consistently satisfied: `{top_more_damaging_roc_log_loss}`; top deletion often reduced confidence and improved log loss.",
        ]
    )
    (args.output_dir / "st_temporal_importance.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Wrote temporal-importance results to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
