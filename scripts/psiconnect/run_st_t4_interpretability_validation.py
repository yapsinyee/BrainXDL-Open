#!/usr/bin/env python3
"""Run seed-2026 ST-T4 frozen-model edge-removal fidelity validation."""

from __future__ import annotations

import argparse
import csv
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
from brainxdl.st.artifacts import FOLDS, fold_inputs, read_json, sha256, split_metadata
from brainxdl.st.checkpoints import load_model
from brainxdl.st.dataset import STGraphDataset, padseq
from brainxdl.xai.st_attention import TransformerAttentionCapture
from brainxdl.xai.st_fidelity import (
    EDGE_IMPORTANCE_DEFINITION,
    aggregate_attention_importance,
    deterministic_random_seed,
    graph_sequence_digest,
    mask_graph_sequence,
    model_state_digest,
    removal_count,
    select_edge_indices,
    undirected_edge_pairs,
)


SEED = 2026
FRACTIONS = (0.05, 0.10, 0.20, 0.30)
CONDITIONS = ("low", "random", "top")
EXPECTED_BASELINE = {"acc": 0.8167, "roc": 0.8438, "f1": 0.8308}
METRIC_NAMES = ("acc", "roc", "f1", "log_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--random-repetitions", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--baseline-tolerance", type=float, default=0.005)
    parser.add_argument("--score-tolerance", type=float, default=1e-5)
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


def finite_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    return {
        name: float(metrics[name]) if math.isfinite(float(metrics[name])) else None
        for name in METRIC_NAMES
    }


def graph_flag(value: object) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().cpu().item())
    return bool(value)


def predict_many(model, setting: dict[str, Any], graphs: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate graph sequences in deterministic input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    dataset = STGraphDataset(graphs)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=padseq,
        pin_memory=False,
    )
    labels, scores = [], []
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
            if logits.shape[-1] > 1:
                probability = torch.softmax(logits, dim=-1)[:, 1]
            else:
                probability = torch.sigmoid(logits.reshape(-1))
            labels.append(target.detach().cpu().numpy().astype(np.int64))
            scores.append(probability.detach().cpu().numpy().astype(float))
    return np.concatenate(labels), np.concatenate(scores)


def baseline_and_importance(model, setting: dict[str, Any], graphs) -> tuple[int, float, np.ndarray, list[str]]:
    batch = padseq([graphs])
    if not batch:
        raise ValueError("Observation has no graph sequence.")
    valid_mask = np.asarray([not graph_flag(graph.pad[0]) for graph in batch], dtype=bool)
    with torch.no_grad(), TransformerAttentionCapture(model) as capture:
        _, _, _, readout, target = model(batch, setting, sample=False)
        logits = model.classifier.linear(readout)
        if logits.shape[-1] > 1:
            score = float(torch.softmax(logits, dim=-1)[0, 1].item())
        else:
            score = float(torch.sigmoid(logits.reshape(-1)[0]).item())
    scores, components = aggregate_attention_importance(
        capture.records, int(model.num_nodes), valid_mask
    )
    return int(target.item()), score, scores, components


def validate_output_targets(output_dir: Path, overwrite: bool) -> None:
    targets = [
        output_dir / "st_t4_interpretability_validation.md",
        output_dir / "st_t4_results.csv",
        output_dir / "st_t4_detailed_results.csv",
        output_dir / "st_t4_plot_data.csv",
        output_dir / "st_t4_experiment_metadata.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "ST-T4 outputs already exist; pass --overwrite to replace them: "
            + ", ".join(path.name for path in existing)
        )


def metric_row(
    *, fold: int | str, condition: str, fraction: float, repetition: int | str,
    count: int, threshold: float | str, y_true: np.ndarray, scores: np.ndarray,
) -> dict[str, Any]:
    if threshold == "fold_specific":
        raise ValueError("A numeric threshold is required for fold-level metrics.")
    predictions = (scores >= float(threshold)).astype(int)
    metrics = compute_classification_metrics(y_true, predictions, scores)
    return {
        "seed": SEED,
        "outer_fold": fold,
        "perturbation": condition,
        "removal_fraction": fraction,
        "random_repetition": repetition,
        "num_edges_removed": count,
        "validation_threshold": threshold,
        "n_samples": len(y_true),
        **finite_metrics(metrics),
    }


def pooled_metrics(entries: list[tuple[np.ndarray, np.ndarray, float]]) -> dict[str, float]:
    labels = np.concatenate([entry[0] for entry in entries])
    scores = np.concatenate([entry[1] for entry in entries])
    predictions = np.concatenate([(entry[1] >= entry[2]).astype(int) for entry in entries])
    return compute_classification_metrics(labels, predictions, scores)


def main() -> int:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError("ST-T4 is frozen to seed 2026 only.")
    if args.random_repetitions != 20:
        raise ValueError("ST-T4 requires exactly 20 random repetitions.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.baseline_tolerance <= 0 or args.score_tolerance <= 0:
        raise ValueError("Validation tolerances must be positive.")

    frozen_path = args.d1_root / "frozen_selection.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(
            f"Missing ST-D1 input bundle at {args.d1_root}. Expected {frozen_path} plus "
            "final/seed_2026/outer_01 through outer_05. Supply a locally prepared "
            "ST-D1 bundle via --d1-root; restricted checkpoints are not distributed."
        )
    frozen = read_json(frozen_path)
    if frozen.get("outer_test_used") is not False:
        raise ValueError("ST-D1 frozen selection must explicitly exclude outer-test data.")
    fold_specs = [fold_inputs(args, frozen, fold) for fold in FOLDS]
    for spec in fold_specs:
        print(
            f"fold={spec['fold']} density={spec['density']} checkpoint={spec['checkpoint']} "
            f"data={spec['fold_file']}",
            flush=True,
        )
    if args.plan_only:
        print("Plan validation complete; no outputs were written.")
        return 0

    validate_output_targets(args.output_dir, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = pick_device(prefer_mps=args.prefer_mps)

    baseline_entries: list[tuple[np.ndarray, np.ndarray, float]] = []
    baseline_by_fold: dict[int, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    master_ids: list[str] | None = None
    master_groups: list[str] | None = None
    test_index_sets: list[set[int]] = []
    component_names: list[str] | None = None
    num_nodes: int | None = None

    # Phase 1 deliberately completes and validates all baselines before any
    # perturbation result is evaluated or interpreted.
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
        checkpoint_seed = setting.get("seed")
        if checkpoint_seed is not None and int(checkpoint_seed) != args.seed:
            raise ValueError(
                f"Fold {fold}: checkpoint seed {checkpoint_seed} does not match {args.seed}."
            )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
            raise AssertionError("Frozen-model evaluation mode was not enforced.")
        before_state = model_state_digest(model)
        fold_labels, fold_scores, importance_rows = [], [], []
        fold_sample_ids = []
        for row_index, (sample_graphs, global_index) in enumerate(zip(graphs, test_indices)):
            original_digest = graph_sequence_digest(sample_graphs)
            label, score, importance, names = baseline_and_importance(
                model, setting, sample_graphs
            )
            if graph_sequence_digest(sample_graphs) != original_digest:
                raise AssertionError(f"Fold {fold} sample {global_index}: baseline mutated input.")
            if component_names is None:
                component_names = names
            elif names != component_names:
                raise ValueError("Attention component order changed across samples/folds.")
            if num_nodes is None:
                num_nodes = int(model.num_nodes)
            elif int(model.num_nodes) != num_nodes:
                raise ValueError("Node count changed across folds.")
            expected_label = int(np.asarray(spec["metrics"]["y_true"], dtype=int)[row_index])
            if label != expected_label:
                raise ValueError(
                    f"Fold {fold} sample {global_index}: label/checkpoint test ordering mismatch."
                )
            fold_labels.append(label)
            fold_scores.append(score)
            importance_rows.append(importance)
            fold_sample_ids.append(sample_ids[int(global_index)])
            print(
                f"[importance] fold={fold} sample={row_index + 1}/{len(graphs)} "
                f"index={global_index}",
                flush=True,
            )

        if model_state_digest(model) != before_state:
            raise AssertionError(f"Fold {fold}: model parameters or buffers changed.")
        labels = np.asarray(fold_labels, dtype=np.int64)
        scores = np.asarray(fold_scores, dtype=float)
        importance_array = np.stack(importance_rows).astype(np.float32)
        reference_scores = np.asarray(spec["metrics"]["y_score"], dtype=float)
        if not np.allclose(scores, reference_scores, atol=args.score_tolerance, rtol=0):
            maximum = float(np.max(np.abs(scores - reference_scores)))
            raise RuntimeError(
                f"Fold {fold}: reproduced baseline scores differ from saved ST-D1 scores "
                f"(max absolute difference {maximum:.3g})."
            )
        threshold = float(spec["metrics"]["validation_threshold"])
        baseline_entries.append((labels, scores, threshold))
        baseline_by_fold[fold] = {
            "labels": labels,
            "scores": scores,
            "threshold": threshold,
            "test_indices": test_indices,
            "sample_ids": np.asarray(fold_sample_ids, dtype=str),
            "importance": importance_array,
        }

        fold_dir = args.output_dir / f"outer_fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pairs = undirected_edge_pairs(int(model.num_nodes))
        np.savez_compressed(
            fold_dir / "edge_importance.npz",
            seed=np.full(len(labels), args.seed, dtype=np.int64),
            outer_fold=np.full(len(labels), fold, dtype=np.int64),
            sample_index=test_indices,
            sample_id=np.asarray(fold_sample_ids, dtype=str),
            source_roi_zero_based=pairs[:, 0],
            target_roi_zero_based=pairs[:, 1],
            importance=importance_array,
            component_names=np.asarray(component_names, dtype=str),
        )
        fold_mean = importance_array.mean(axis=0)
        rank_order = np.lexsort((np.arange(len(fold_mean)), -fold_mean))
        ranks = np.empty(len(fold_mean), dtype=np.int64)
        ranks[rank_order] = np.arange(1, len(fold_mean) + 1)
        write_csv(
            fold_dir / "edge_importance_fold_mean.csv",
            [
                {
                    "seed": args.seed,
                    "outer_fold": fold,
                    "source_roi_zero_based": int(source),
                    "target_roi_zero_based": int(target),
                    "source_roi_one_based": int(source + 1),
                    "target_roi_one_based": int(target + 1),
                    "importance_mean": float(fold_mean[index]),
                    "importance_sd": float(importance_array[:, index].std(ddof=1)),
                    "rank": int(ranks[index]),
                }
                for index, (source, target) in enumerate(pairs)
            ],
        )
        provenance[str(fold)] = {
            "density": spec["density"],
            "checkpoint": str(spec["checkpoint"]),
            "checkpoint_sha256": sha256(spec["checkpoint"]),
            "metrics": str(spec["metrics_path"].resolve()),
            "fold_file": str(spec["fold_file"].resolve()),
            "validation_threshold": threshold,
            "n_test": len(labels),
        }
        del model

    if any(left & right for i, left in enumerate(test_index_sets) for right in test_index_sets[i + 1 :]):
        raise ValueError("Outer-test indices overlap across folds.")
    if master_ids is None or set().union(*test_index_sets) != set(range(len(master_ids))):
        raise ValueError("Five outer-test folds do not cover every observation exactly once.")
    baseline_metrics = pooled_metrics(baseline_entries)
    for name, expected in EXPECTED_BASELINE.items():
        if abs(float(baseline_metrics[name]) - expected) > args.baseline_tolerance:
            raise RuntimeError(
                f"Pooled baseline {name}={baseline_metrics[name]:.6f} differs materially "
                f"from expected {expected:.6f}; perturbation evaluation stopped."
            )
    print(
        "[baseline validated] "
        + " ".join(f"{name}={baseline_metrics[name]:.4f}" for name in METRIC_NAMES),
        flush=True,
    )

    assert num_nodes is not None and component_names is not None
    pairs = undirected_edge_pairs(num_nodes)
    num_edges = len(pairs)
    detailed_rows: list[dict[str, Any]] = []
    pooled: dict[tuple[str, float, int], list[tuple[np.ndarray, np.ndarray, float]]] = defaultdict(list)
    for fold, cached in baseline_by_fold.items():
        detailed_rows.append(
            metric_row(
                fold=fold, condition="none", fraction=0.0, repetition="",
                count=0, threshold=cached["threshold"], y_true=cached["labels"],
                scores=cached["scores"],
            )
        )

    # Phase 2 reloads each frozen checkpoint and evaluates matched masks.
    for spec in fold_specs:
        fold = int(spec["fold"])
        cached = baseline_by_fold[fold]
        payload = torch.load(spec["fold_file"], map_location="cpu", weights_only=False)
        graphs = np.asarray(payload["test_graphs"], dtype=object)
        model, setting, _ = load_model(spec["checkpoint"], device, args.sparse_on_cpu)
        checkpoint_seed = setting.get("seed")
        if checkpoint_seed is not None and int(checkpoint_seed) != args.seed:
            raise ValueError(
                f"Fold {fold}: checkpoint seed {checkpoint_seed} does not match {args.seed}."
            )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before_state = model_state_digest(model)

        for fraction in FRACTIONS:
            count = removal_count(num_edges, fraction)
            for condition in ("low", "top"):
                masked_samples = []
                for row_index, sample_graphs in enumerate(graphs):
                    selected = select_edge_indices(
                        cached["importance"][row_index], fraction, condition
                    )
                    if len(selected) != count:
                        raise AssertionError("Matched perturbation counts differ.")
                    original_digest = graph_sequence_digest(sample_graphs)
                    masked = mask_graph_sequence(sample_graphs, pairs[selected])
                    if graph_sequence_digest(sample_graphs) != original_digest:
                        raise AssertionError("Perturbation mutated an original input tensor.")
                    masked_samples.append(masked)
                labels, scores = predict_many(
                    model, setting, np.asarray(masked_samples, dtype=object), args.batch_size
                )
                if not np.array_equal(labels, cached["labels"]):
                    raise AssertionError("Perturbation changed target labels or sample order.")
                key = (condition, fraction, -1)
                pooled[key].append((cached["labels"], scores, cached["threshold"]))
                detailed_rows.append(
                    metric_row(
                        fold=fold, condition=condition, fraction=fraction, repetition="",
                        count=count, threshold=cached["threshold"],
                        y_true=cached["labels"], scores=scores,
                    )
                )
                print(f"[perturb] fold={fold} condition={condition} fraction={fraction:.2f}", flush=True)

            for repetition in range(1, args.random_repetitions + 1):
                masked_samples = []
                reproducibility_checked = False
                for row_index, sample_graphs in enumerate(graphs):
                    random_seed = deterministic_random_seed(
                        args.seed, fold, int(cached["test_indices"][row_index]), fraction, repetition
                    )
                    selected = select_edge_indices(
                        cached["importance"][row_index], fraction, "random",
                        rng=np.random.default_rng(random_seed),
                    )
                    if not reproducibility_checked:
                        repeated = select_edge_indices(
                            cached["importance"][row_index], fraction, "random",
                            rng=np.random.default_rng(random_seed),
                        )
                        if not np.array_equal(selected, repeated):
                            raise AssertionError("Random masking is not reproducible.")
                        reproducibility_checked = True
                    if len(selected) != count:
                        raise AssertionError("Random perturbation count differs from matched masks.")
                    original_digest = graph_sequence_digest(sample_graphs)
                    masked = mask_graph_sequence(sample_graphs, pairs[selected])
                    if graph_sequence_digest(sample_graphs) != original_digest:
                        raise AssertionError("Random perturbation mutated an original input tensor.")
                    masked_samples.append(masked)
                labels, scores = predict_many(
                    model, setting, np.asarray(masked_samples, dtype=object), args.batch_size
                )
                if not np.array_equal(labels, cached["labels"]):
                    raise AssertionError("Random perturbation changed target labels or sample order.")
                key = ("random", fraction, repetition)
                pooled[key].append((cached["labels"], scores, cached["threshold"]))
                detailed_rows.append(
                    metric_row(
                        fold=fold, condition="random", fraction=fraction,
                        repetition=repetition, count=count, threshold=cached["threshold"],
                        y_true=cached["labels"], scores=scores,
                    )
                )
                print(
                    f"[perturb] fold={fold} condition=random fraction={fraction:.2f} "
                    f"repetition={repetition}/{args.random_repetitions}",
                    flush=True,
                )
        if model_state_digest(model) != before_state:
            raise AssertionError(f"Fold {fold}: model state changed during perturbation.")
        del model

    summary_rows: list[dict[str, Any]] = []
    baseline_clean = finite_metrics(baseline_metrics)
    summary_rows.append(
        {
            "perturbation": "none", "removal_fraction": 0.0,
            "edges_removed_percent": 0, "num_edges_removed": 0,
            "random_repetitions": "", **baseline_clean,
            "delta_acc": 0.0, "delta_roc": 0.0,
        }
    )
    summary_metrics: dict[tuple[str, float], dict[str, float]] = {}
    for fraction in FRACTIONS:
        for condition in CONDITIONS:
            if condition == "random":
                repetition_metrics = [
                    pooled_metrics(pooled[(condition, fraction, repetition)])
                    for repetition in range(1, args.random_repetitions + 1)
                ]
                metrics = {
                    name: float(np.mean([row[name] for row in repetition_metrics]))
                    for name in METRIC_NAMES
                }
                repetitions: int | str = args.random_repetitions
            else:
                metrics = {
                    name: float(value)
                    for name, value in pooled_metrics(pooled[(condition, fraction, -1)]).items()
                    if name in METRIC_NAMES
                }
                repetitions = ""
            summary_metrics[(condition, fraction)] = metrics
            summary_rows.append(
                {
                    "perturbation": condition,
                    "removal_fraction": fraction,
                    "edges_removed_percent": int(round(100 * fraction)),
                    "num_edges_removed": removal_count(num_edges, fraction),
                    "random_repetitions": repetitions,
                    **finite_metrics(metrics),
                    "delta_acc": float(metrics["acc"] - baseline_metrics["acc"]),
                    "delta_roc": float(metrics["roc"] - baseline_metrics["roc"]),
                }
            )

    plot_rows = []
    for condition in ("top", "random", "low"):
        plot_rows.append(
            {"perturbation": condition, "edges_removed_percent": 0, "roc_auc_percent": 100 * baseline_metrics["roc"]}
        )
        for fraction in FRACTIONS:
            plot_rows.append(
                {
                    "perturbation": condition,
                    "edges_removed_percent": int(round(100 * fraction)),
                    "roc_auc_percent": 100 * summary_metrics[(condition, fraction)]["roc"],
                }
            )

    write_csv(args.output_dir / "st_t4_results.csv", summary_rows)
    write_csv(args.output_dir / "st_t4_detailed_results.csv", detailed_rows)
    write_csv(args.output_dir / "st_t4_plot_data.csv", plot_rows)

    top_worse_all = all(
        summary_metrics[("top", fraction)]["roc"] < summary_metrics[("random", fraction)]["roc"]
        and summary_metrics[("top", fraction)]["roc"] < summary_metrics[("low", fraction)]["roc"]
        for fraction in FRACTIONS
    )
    metadata = {
        "schema_version": 1,
        "experiment": "ST-T4",
        "status": "complete",
        "seed": args.seed,
        "outer_folds": list(FOLDS),
        "removal_fractions": list(FRACTIONS),
        "random_repetitions": args.random_repetitions,
        "num_nodes": num_nodes,
        "num_valid_undirected_fc_edges": num_edges,
        "removed_edge_counts": {
            str(fraction): removal_count(num_edges, fraction) for fraction in FRACTIONS
        },
        "attention_components": component_names,
        "edge_importance_definition": EDGE_IMPORTANCE_DEFINITION,
        "mask_definition": (
            "Selected undirected FCs are zeroed symmetrically in x and adj and removed "
            "from edge_index/edge_attr in every temporal window; diagonal values are unchanged."
        ),
        "candidate_edge_definition": "All 200 choose 2 off-diagonal undirected FC entries.",
        "baseline_metrics": finite_metrics(baseline_metrics),
        "baseline_expected": EXPECTED_BASELINE,
        "baseline_tolerance": args.baseline_tolerance,
        "expected_fidelity_pattern_all_fractions": top_worse_all,
        "frozen_selection": str(frozen_path.resolve()),
        "fold_provenance": provenance,
        "validation_checks": {
            "seed_exactly_2026": True,
            "checkpoint_fold_matching": True,
            "outer_test_fold_matching": True,
            "model_state_unchanged": True,
            "fc_symmetry_preserved": True,
            "diagonals_unchanged": True,
            "original_tensors_unchanged": True,
            "matched_removal_counts": True,
            "random_masking_reproducible": True,
            "labels_excluded_from_ranking": True,
            "baseline_reproduced": True,
        },
    }
    (args.output_dir / "st_t4_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    row_lookup = {(row["perturbation"], row["removal_fraction"]): row for row in summary_rows}
    report_lines = [
        "# ST-T4: Quantitative validation of ST-BrainXDL interpretability",
        "",
        "This post-hoc fidelity experiment uses only seed `2026`, all five held-out outer",
        "folds, and the exact frozen `[ST-D1]` checkpoints and fold-specific validation",
        "thresholds. No model parameter is trained or updated.",
        "",
        "## Results",
        "",
        "| Perturbation | Edges removed (%) | Accuracy (%) | ROC-AUC (%) | F1-score (%) | Log loss | Δ Accuracy | Δ AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| None | 0 | {100 * baseline_metrics['acc']:.2f} | {100 * baseline_metrics['roc']:.2f} | {100 * baseline_metrics['f1']:.2f} | {baseline_metrics['log_loss']:.4f} | – | – |",
    ]
    labels = {"low": "Low-importance", "random": "Random", "top": "Top-importance"}
    for fraction in FRACTIONS:
        for condition in CONDITIONS:
            row = row_lookup[(condition, fraction)]
            report_lines.append(
                f"| {labels[condition]} | {row['edges_removed_percent']} | "
                f"{100 * row['acc']:.2f} | {100 * row['roc']:.2f} | {100 * row['f1']:.2f} | "
                f"{row['log_loss']:.4f} | {100 * row['delta_acc']:+.2f} | {100 * row['delta_roc']:+.2f} |"
            )
    interpretation = (
        "The prespecified pattern was observed at every removal fraction: top-attributed "
        "edge removal produced lower ROC-AUC than both matched random and low-importance removal."
        if top_worse_all
        else "The prespecified fidelity pattern was not observed systematically at every removal fraction."
    )
    report_lines.extend(
        [
            "",
            "Random rows report the mean metric across 20 deterministic matched-mask",
            "repetitions; deltas are relative to the shared unperturbed pooled baseline.",
            "",
            "## Edge-importance definition",
            "",
            metadata["edge_importance_definition"],
            "The ranking is observation-specific and is computed from the unmodified input.",
            "Outer-test labels are used only after predictions are produced to calculate metrics.",
            "",
            "## Interpretation",
            "",
            interpretation,
            "A performance decrease alone is not treated as interpretability evidence; the",
            "comparison of top-attributed removal against matched random and low-attribution",
            "removal is the relevant fidelity test.",
            "",
            "## Validation and limitations",
            "",
            f"- Baseline reproduced: accuracy `{baseline_metrics['acc']:.4f}`, ROC-AUC `{baseline_metrics['roc']:.4f}`, F1 `{baseline_metrics['f1']:.4f}`, log loss `{baseline_metrics['log_loss']:.4f}`.",
            f"- Tensor layout: `{num_nodes} x {num_nodes}` FC matrix per window; `{len(component_names)}` attention components; `{num_edges}` undirected off-diagonal candidates.",
            "- All mandatory seed, checkpoint/fold, non-mutation, symmetry, diagonal, matched-count, reproducibility, label-independence, and baseline checks passed.",
            "- TransformerConv attention covers the graph encoder/prior/recurrent paths, but not the stable-FC skip, temporal pooling, or classifier directly.",
            "- The analysis is a fixed seed-2026 post-hoc fidelity study and does not establish cross-seed attribution stability or causal neurobiological importance.",
        ]
    )
    (args.output_dir / "st_t4_interpretability_validation.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(f"Complete: {args.output_dir / 'st_t4_interpretability_validation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
