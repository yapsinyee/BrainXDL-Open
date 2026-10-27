#!/usr/bin/env python3
"""Extract raw TransformerConv attention from a frozen ST-BrainXDL checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brainxdl.common.device import pick_device
from brainxdl.common.seed import seed_everything
from brainxdl.st.checkpoints import load_model
from brainxdl.st.dataset import padseq
from brainxdl.xai.st_attention import (
    TransformerAttentionCapture,
    records_to_component_series,
    summarize_attention_series,
    symmetrize_attention,
    top_undirected_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold-file", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional diagnostic limit; 0 extracts the complete selected split.",
    )
    parser.add_argument("--prefer-mps", action="store_true")
    parser.add_argument(
        "--sparse-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_flag(value: object) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().cpu().item())
    return bool(value)


def sample_metadata(
    metadata: dict[str, Any], split: str, n_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_indices = metadata.get("split_indices", {}).get(split)
    all_sample_ids = metadata.get("sample_ids")
    all_groups = metadata.get("participant_groups")
    if raw_indices is None or all_sample_ids is None or all_groups is None:
        raise ValueError(
            "Fold metadata must contain split_indices, sample_ids, and participant_groups."
        )
    indices = np.asarray(raw_indices, dtype=np.int64)
    if len(indices) != n_rows:
        raise ValueError(f"{split} indices and graph rows differ ({len(indices)} != {n_rows}).")
    sample_ids = np.asarray([str(all_sample_ids[int(index)]) for index in indices], dtype=str)
    participant_ids = np.asarray([str(all_groups[int(index)]) for index in indices], dtype=str)
    if len(set(map(str, sample_ids))) != len(sample_ids):
        raise ValueError(f"{split} observation IDs are not unique.")
    return indices, sample_ids, participant_ids


def prediction_from_readout(model, readout: torch.Tensor) -> tuple[int, float]:
    logits = model.classifier.linear(readout)
    if logits.shape[-1] > 1:
        probabilities = torch.softmax(logits, dim=-1)
        prediction = int(probabilities.argmax(dim=-1).item())
        positive_probability = float(probabilities[0, 1].item())
    else:
        positive_probability = float(torch.sigmoid(logits.reshape(-1)[0]).item())
        prediction = int(positive_probability >= 0.5)
    return prediction, positive_probability


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.outer_fold not in range(1, 6):
        raise ValueError("--outer-fold must be from 1 through 5.")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.fold_file.is_file():
        raise FileNotFoundError(args.fold_file)

    output_npz = args.output_dir / f"st_attention_fold{args.outer_fold}_{args.split}.npz"
    output_json = args.output_dir / f"st_attention_fold{args.outer_fold}_{args.split}.json"
    output_csv = args.output_dir / f"st_attention_fold{args.outer_fold}_{args.split}_top_edges.csv"
    existing = [path for path in (output_npz, output_json, output_csv) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Attention outputs already exist: {[path.name for path in existing]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = pick_device(prefer_mps=args.prefer_mps)
    model, setting, _ = load_model(args.checkpoint, device, args.sparse_on_cpu)
    checkpoint_seed = setting.get("seed")
    if checkpoint_seed is not None and int(checkpoint_seed) != args.seed:
        raise ValueError(
            f"Checkpoint seed mismatch: expected {args.seed}, found {checkpoint_seed}."
        )
    model.eval()

    fold_payload = torch.load(args.fold_file, map_location="cpu", weights_only=False)
    graphs = np.asarray(fold_payload[f"{args.split}_graphs"], dtype=object)
    metadata = fold_payload.get("metadata", {})
    indices, sample_ids, participant_ids = sample_metadata(metadata, args.split, len(graphs))
    n_extract = len(graphs) if args.max_samples <= 0 else min(args.max_samples, len(graphs))
    num_nodes = int(setting["model_params"]["num_nodes"])

    component_names: list[str] | None = None
    mean_rows: list[np.ndarray] = []
    conditional_rows: list[np.ndarray] = []
    presence_rows: list[np.ndarray] = []
    temporal_strength_rows: list[list[np.ndarray]] = []
    temporal_top_rows: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    sequence_lengths: list[int] = []
    labels: list[int] = []
    predictions: list[int] = []
    positive_probabilities: list[float] = []
    top_edge_rows: list[dict[str, Any]] = []

    for row_index in range(n_extract):
        subject_graphs = graphs[row_index]
        batch = padseq([subject_graphs])
        if not batch:
            raise ValueError(f"Observation {sample_ids[row_index]} has no graph sequence.")
        valid_mask = np.asarray([not graph_flag(graph.pad[0]) for graph in batch], dtype=bool)
        if not valid_mask.any():
            raise ValueError(f"Observation {sample_ids[row_index]} has no valid windows.")

        with torch.no_grad(), TransformerAttentionCapture(model) as capture:
            _, _, _, readout, target = model(batch, setting, sample=False)
            prediction, positive_probability = prediction_from_readout(model, readout)
        observed_label = int(target.item())
        expected_index = int(indices[row_index])
        label_mapping = metadata.get("label_mapping", {})
        if observed_label not in (0, 1):
            raise ValueError(f"Unexpected label {observed_label} for sample index {expected_index}.")

        series_by_component = records_to_component_series(capture.records, num_nodes)
        current_components = list(series_by_component)
        if component_names is None:
            component_names = current_components
        elif current_components != component_names:
            raise ValueError("Attention component order changed across observations.")

        sample_means = []
        sample_conditionals = []
        sample_presence = []
        sample_strengths = []
        sample_temporal_top = []
        for name in component_names:
            series = series_by_component[name]
            if len(series) != len(valid_mask):
                raise ValueError(
                    f"{name}: captured {len(series)} windows but graph sequence has {len(valid_mask)}."
                )
            series = series[valid_mask]
            mean, conditional, presence = summarize_attention_series(series)
            sample_means.append(mean)
            sample_conditionals.append(conditional)
            sample_presence.append(presence)

            symmetric = symmetrize_attention(series)
            node_strength = symmetric.sum(axis=-1) / max(num_nodes - 1, 1)
            sample_strengths.append(node_strength.astype(np.float32))

            time_sources = np.full((len(series), args.top_k), -1, dtype=np.int16)
            time_targets = np.full((len(series), args.top_k), -1, dtype=np.int16)
            time_weights = np.full((len(series), args.top_k), np.nan, dtype=np.float32)
            for time_index, matrix in enumerate(series):
                for rank, (source, target_index, weight) in enumerate(
                    top_undirected_edges(matrix, args.top_k)
                ):
                    time_sources[time_index, rank] = source
                    time_targets[time_index, rank] = target_index
                    time_weights[time_index, rank] = weight
            sample_temporal_top.append((time_sources, time_targets, time_weights))

            for score_type, matrix in (
                ("mean_with_absent_zero", mean),
                ("conditional_mean", conditional),
                ("edge_presence", presence),
            ):
                for rank, (source, target_index, weight) in enumerate(
                    top_undirected_edges(matrix, args.top_k), start=1
                ):
                    top_edge_rows.append(
                        {
                            "outer_fold": args.outer_fold,
                            "seed": args.seed,
                            "split": args.split,
                            "sample_index": expected_index,
                            "sample_id": str(sample_ids[row_index]),
                            "participant_id": str(participant_ids[row_index]),
                            "label": observed_label,
                            "condition": label_mapping.get(str(observed_label), str(observed_label)),
                            "prediction": prediction,
                            "positive_probability": positive_probability,
                            "component": name,
                            "score_type": score_type,
                            "rank": rank,
                            "source_roi_zero_based": source,
                            "target_roi_zero_based": target_index,
                            "source_roi_one_based": source + 1,
                            "target_roi_one_based": target_index + 1,
                            "score": weight,
                        }
                    )

        mean_rows.append(np.stack(sample_means))
        conditional_rows.append(np.stack(sample_conditionals))
        presence_rows.append(np.stack(sample_presence))
        temporal_strength_rows.append(sample_strengths)
        temporal_top_rows.append(sample_temporal_top)
        sequence_lengths.append(int(valid_mask.sum()))
        labels.append(observed_label)
        predictions.append(prediction)
        positive_probabilities.append(positive_probability)
        print(
            f"[{row_index + 1}/{n_extract}] {sample_ids[row_index]} "
            f"label={observed_label} pred={prediction} windows={valid_mask.sum()}",
            flush=True,
        )

    if component_names is None:
        raise RuntimeError("No attention was extracted.")
    n_components = len(component_names)
    max_windows = max(sequence_lengths)
    temporal_node_strength = np.full(
        (n_extract, n_components, max_windows, num_nodes), np.nan, dtype=np.float32
    )
    temporal_top_source = np.full(
        (n_extract, n_components, max_windows, args.top_k), -1, dtype=np.int16
    )
    temporal_top_target = np.full_like(temporal_top_source, -1)
    temporal_top_weight = np.full(
        (n_extract, n_components, max_windows, args.top_k), np.nan, dtype=np.float32
    )
    for sample_index, (strengths, top_components) in enumerate(
        zip(temporal_strength_rows, temporal_top_rows)
    ):
        for component_index, (strength, top_values) in enumerate(
            zip(strengths, top_components)
        ):
            n_windows = len(strength)
            temporal_node_strength[sample_index, component_index, :n_windows] = strength
            temporal_top_source[sample_index, component_index, :n_windows] = top_values[0]
            temporal_top_target[sample_index, component_index, :n_windows] = top_values[1]
            temporal_top_weight[sample_index, component_index, :n_windows] = top_values[2]

    np.savez_compressed(
        output_npz,
        component_names=np.asarray(component_names, dtype=str),
        sample_index=indices[:n_extract],
        sample_id=sample_ids[:n_extract],
        participant_id=participant_ids[:n_extract],
        labels=np.asarray(labels, dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.int64),
        positive_probability=np.asarray(positive_probabilities, dtype=np.float32),
        sequence_length=np.asarray(sequence_lengths, dtype=np.int64),
        mean_attention=np.stack(mean_rows).astype(np.float32),
        conditional_mean_attention=np.stack(conditional_rows).astype(np.float32),
        edge_presence=np.stack(presence_rows).astype(np.float32),
        temporal_node_strength=temporal_node_strength,
        temporal_top_source=temporal_top_source,
        temporal_top_target=temporal_top_target,
        temporal_top_weight=temporal_top_weight,
        outer_fold=np.full(n_extract, args.outer_fold, dtype=np.int64),
        seed=np.full(n_extract, args.seed, dtype=np.int64),
        source_checkpoint=np.asarray([str(args.checkpoint.resolve())] * n_extract, dtype=str),
    )
    write_rows(output_csv, top_edge_rows)

    report = {
        "schema_version": 1,
        "status": "complete" if n_extract == len(graphs) else "diagnostic_partial",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_fold_file": str(args.fold_file.resolve()),
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "split": args.split,
        "n_available": int(len(graphs)),
        "n_extracted": n_extract,
        "num_nodes": num_nodes,
        "components": component_names,
        "top_k": args.top_k,
        "attention_semantics": (
            "Directed TransformerConv coefficients averaged across heads; self-loops removed. "
            "Undirected rankings average source-to-target and target-to-source coefficients."
        ),
        "interpretation_warning": (
            "Attention is a descriptive model-internal quantity, not a validated explanation. "
            "It covers TransformerConv paths, not the stable-FC skip, temporal pooling, or "
            "classifier. Stability, perturbation faithfulness, and randomization sanity checks "
            "are required."
        ),
        "outputs": {
            "npz": str(output_npz.resolve()),
            "top_edges_csv": str(output_csv.resolve()),
        },
    }
    output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_npz}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
