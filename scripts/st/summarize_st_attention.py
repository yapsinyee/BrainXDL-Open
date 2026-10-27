#!/usr/bin/env python3
"""Summarize held-out ST-BrainXDL attention across five outer folds."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from brainxdl.xai.st_attention import symmetrize_attention, top_undirected_edges


SCORE_KEYS = {
    "mean_with_absent_zero": "mean_attention",
    "conditional_mean": "conditional_mean_attention",
    "edge_presence": "edge_presence",
}
CONDITIONS = {0: "control", 1: "psilocybin"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--expected-samples", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def edge_vector(matrix: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(matrix.shape[0], k=1)
    return matrix[upper]


def top_edge_set(matrix: np.ndarray, top_k: int) -> set[tuple[int, int]]:
    return {(source, target) for source, target, _ in top_undirected_edges(matrix, top_k)}


def pairwise_stability(matrices: list[np.ndarray], top_k: int) -> tuple[float, float, int]:
    jaccards = []
    correlations = []
    for left, right in itertools.combinations(matrices, 2):
        left_set = top_edge_set(left, top_k)
        right_set = top_edge_set(right, top_k)
        union = left_set | right_set
        jaccards.append(len(left_set & right_set) / len(union) if union else float("nan"))
        correlation = spearmanr(edge_vector(left), edge_vector(right)).statistic
        correlations.append(float(correlation))
    return (
        float(np.nanmean(jaccards)),
        float(np.nanmean(correlations)),
        len(jaccards),
    )


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")
    if len(args.fold_files) != 5:
        raise ValueError("The primary ST attention summary requires exactly five fold files.")

    output_md = args.output_dir / "st_attention_summary.md"
    output_json = args.output_dir / "st_attention_summary.json"
    output_npz = args.output_dir / "st_attention_group_matrices.npz"
    output_top = args.output_dir / "st_attention_group_top_edges.csv"
    output_stability = args.output_dir / "st_attention_fold_stability.csv"
    output_subjects = args.output_dir / "st_attention_subject_index.csv"
    targets = (output_md, output_json, output_npz, output_top, output_stability, output_subjects)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Summary outputs already exist: {[path.name for path in existing]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_payloads = []
    component_names: list[str] | None = None
    num_nodes: int | None = None
    seen_folds: set[int] = set()
    seen_samples: set[str] = set()
    seen_participants: set[str] = set()
    subject_rows: list[dict[str, Any]] = []
    for path in args.fold_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=False)
        folds = np.unique(np.asarray(data["outer_fold"], dtype=np.int64))
        seeds = np.unique(np.asarray(data["seed"], dtype=np.int64))
        if len(folds) != 1 or int(folds[0]) not in range(1, 6):
            raise ValueError(f"{path}: expected exactly one valid outer fold.")
        if len(seeds) != 1 or int(seeds[0]) != args.seed:
            raise ValueError(f"{path}: expected seed {args.seed}.")
        fold = int(folds[0])
        if fold in seen_folds:
            raise ValueError(f"Duplicate outer fold {fold}.")
        seen_folds.add(fold)

        names = np.asarray(data["component_names"]).astype(str).tolist()
        if component_names is None:
            component_names = names
            num_nodes = int(data["mean_attention"].shape[-1])
        elif names != component_names or int(data["mean_attention"].shape[-1]) != num_nodes:
            raise ValueError("Attention components or ROI dimensions differ across folds.")

        sample_ids = np.asarray(data["sample_id"]).astype(str)
        participants = np.asarray(data["participant_id"]).astype(str)
        labels = np.asarray(data["labels"], dtype=np.int64)
        predictions = np.asarray(data["predictions"], dtype=np.int64)
        probabilities = np.asarray(data["positive_probability"], dtype=float)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"{path}: duplicate observation IDs within fold.")
        if seen_samples.intersection(sample_ids):
            raise ValueError(f"{path}: observation appears in more than one outer-test fold.")
        if seen_participants.intersection(participants):
            raise ValueError(f"{path}: participant appears in more than one outer-test fold.")
        seen_samples.update(map(str, sample_ids))
        seen_participants.update(map(str, participants))
        for index in range(len(sample_ids)):
            subject_rows.append(
                {
                    "outer_fold": fold,
                    "seed": args.seed,
                    "sample_id": sample_ids[index],
                    "participant_id": participants[index],
                    "label": int(labels[index]),
                    "condition": CONDITIONS[int(labels[index])],
                    "prediction": int(predictions[index]),
                    "positive_probability": float(probabilities[index]),
                    "source": str(path.resolve()),
                }
            )
        fold_payloads.append(
            {
                "fold": fold,
                "labels": labels,
                **{
                    score_type: np.asarray(data[array_key], dtype=np.float32)
                    for score_type, array_key in SCORE_KEYS.items()
                },
            }
        )

    if seen_folds != {1, 2, 3, 4, 5}:
        raise ValueError(f"Missing outer folds: {sorted({1, 2, 3, 4, 5} - seen_folds)}")
    if args.expected_samples > 0 and len(seen_samples) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} unique held-out observations, found {len(seen_samples)}."
        )
    assert component_names is not None and num_nodes is not None

    top_rows: list[dict[str, Any]] = []
    group_arrays: dict[str, np.ndarray] = {}
    contrast_arrays: dict[str, np.ndarray] = {}
    for score_type in SCORE_KEYS:
        condition_matrices = []
        for label, condition in CONDITIONS.items():
            fold_matrices = []
            all_subject_matrices = []
            for payload in sorted(fold_payloads, key=lambda item: item["fold"]):
                mask = payload["labels"] == label
                if not mask.any():
                    raise ValueError(f"Fold {payload['fold']} has no {condition} test observations.")
                subject_matrices = symmetrize_attention(payload[score_type][mask])
                fold_matrices.append(subject_matrices.mean(axis=0))
                all_subject_matrices.append(subject_matrices)
            group_matrix = np.concatenate(all_subject_matrices, axis=0).mean(axis=0)
            condition_matrices.append(group_matrix)
        group_arrays[score_type] = np.stack(condition_matrices)
        contrast_arrays[score_type] = condition_matrices[1] - condition_matrices[0]

    stability_rows: list[dict[str, Any]] = []
    for score_type in SCORE_KEYS:
        for label, condition in CONDITIONS.items():
            for component_index, component in enumerate(component_names):
                fold_matrices = []
                for payload in sorted(fold_payloads, key=lambda item: item["fold"]):
                    mask = payload["labels"] == label
                    matrices = symmetrize_attention(
                        payload[score_type][mask, component_index]
                    )
                    fold_matrices.append(matrices.mean(axis=0))
                jaccard, correlation, n_pairs = pairwise_stability(fold_matrices, args.top_k)
                stability_rows.append(
                    {
                        "seed": args.seed,
                        "score_type": score_type,
                        "condition": condition,
                        "component": component,
                        "top_k": args.top_k,
                        "fold_pairs": n_pairs,
                        "mean_top_k_jaccard": jaccard,
                        "mean_spearman_all_edges": correlation,
                    }
                )

    for score_type in SCORE_KEYS:
        for condition_index, condition in enumerate(CONDITIONS.values()):
            for component_index, component in enumerate(component_names):
                matrix = group_arrays[score_type][condition_index, component_index]
                for rank, (source, target, score) in enumerate(
                    top_undirected_edges(matrix, args.top_k), start=1
                ):
                    top_rows.append(
                        {
                            "seed": args.seed,
                            "score_type": score_type,
                            "analysis": condition,
                            "component": component,
                            "rank": rank,
                            "source_roi_zero_based": source,
                            "target_roi_zero_based": target,
                            "source_roi_one_based": source + 1,
                            "target_roi_one_based": target + 1,
                            "score": score,
                            "direction": (
                                "higher edge presence"
                                if score_type == "edge_presence"
                                else "higher attention"
                            ),
                        }
                    )
        for component_index, component in enumerate(component_names):
            contrast = contrast_arrays[score_type][component_index]
            upper = np.triu_indices(num_nodes, k=1)
            values = contrast[upper]
            order = np.argsort(np.abs(values))[::-1][: args.top_k]
            for rank, position in enumerate(order, start=1):
                score = float(values[position])
                top_rows.append(
                    {
                        "seed": args.seed,
                        "score_type": score_type,
                        "analysis": "psilocybin_minus_control",
                        "component": component,
                        "rank": rank,
                        "source_roi_zero_based": int(upper[0][position]),
                        "target_roi_zero_based": int(upper[1][position]),
                        "source_roi_one_based": int(upper[0][position]) + 1,
                        "target_roi_one_based": int(upper[1][position]) + 1,
                        "score": score,
                        "direction": "psilocybin higher" if score > 0 else "control higher",
                    }
                )

    np.savez_compressed(
        output_npz,
        component_names=np.asarray(component_names, dtype=str),
        condition_names=np.asarray(list(CONDITIONS.values()), dtype=str),
        **{name: matrix for name, matrix in group_arrays.items()},
        **{f"contrast_{name}": matrix for name, matrix in contrast_arrays.items()},
        seed=np.asarray([args.seed], dtype=np.int64),
    )
    write_csv(output_top, top_rows)
    write_csv(output_stability, stability_rows)
    write_csv(output_subjects, sorted(subject_rows, key=lambda row: (row["outer_fold"], row["sample_id"])))

    result = {
        "status": "complete",
        "scope": "held-out outer-test attention; five folds; seed 2026 only",
        "seed": args.seed,
        "n_samples": len(seen_samples),
        "n_participants": len(seen_participants),
        "folds": sorted(seen_folds),
        "components": component_names,
        "score_types": list(SCORE_KEYS),
        "top_k": args.top_k,
        "interpretation_warning": (
            "These are descriptive attention summaries, not causal or validated feature "
            "attributions, and they do not cover the stable-FC skip, temporal pooling, or "
            "classifier. Perturbation faithfulness and randomization sanity checks remain required."
        ),
        "outputs": {path.stem: str(path.resolve()) for path in targets if path != output_json},
    }
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# ST-BrainXDL Attention Summary",
        "",
        f"Held-out outer-test attention across five participant-grouped folds, seed `{args.seed}`.",
        "",
        "> Attention coefficients are descriptive internal weights. They are not yet",
        "> validated feature attributions; perturbation faithfulness and model/label",
        "> randomization checks remain required. They cover only TransformerConv paths,",
        "> not the stable-FC skip, temporal pooling, or classifier.",
        "",
        "## Fold stability",
        "",
        "| Score | Condition | Component | Top-k Jaccard | All-edge Spearman |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in stability_rows:
        lines.append(
            f"| {row['score_type']} | {row['condition']} | {row['component']} | "
            f"{row['mean_top_k_jaccard']:.4f} | {row['mean_spearman_all_edges']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Group matrices: `{output_npz}`",
            f"- Ranked group/contrast edges: `{output_top}`",
            f"- Fold stability: `{output_stability}`",
            f"- Held-out observation index: `{output_subjects}`",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
