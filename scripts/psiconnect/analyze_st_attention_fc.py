#!/usr/bin/env python3
"""Create thesis-ready FC interpretation outputs from completed ST-T4 attention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "brainxdl-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "brainxdl-cache"))

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import nibabel as nib
    from nilearn import plotting
except ModuleNotFoundError:
    nib = None
    plotting = None

from brainxdl.xai.st_fc_interpretation import (
    CANONICAL_NETWORKS,
    NETWORK_DISPLAY,
    aggregate_network_pairs,
    edge_vector_to_matrix,
    group_importance,
    load_schaefer_labels,
    node_involvement,
    rank_edges,
)
from brainxdl.xai.st_fidelity import EDGE_IMPORTANCE_DEFINITION, undirected_edge_pairs


SEED = 2026
FOLDS = (1, 2, 3, 4, 5)
NUM_NODES = 200
NUM_EDGES = NUM_NODES * (NUM_NODES - 1) // 2
EXPECTED_COMPONENTS = (
    "posterior_encoder.module_0",
    "prior_encoder.module_0",
    "gru.update.input",
    "gru.update.hidden",
    "gru.reset.input",
    "gru.reset.hidden",
    "gru.candidate.input",
    "gru.candidate.hidden",
)
NETWORK_COLORS = {
    "Vis": "#7B61A8",
    "SomMot": "#2F9E44",
    "DorsAttn": "#9C6ADE",
    "SalVentAttn": "#D9485F",
    "Limbic": "#E67700",
    "Cont": "#228BE6",
    "Default": "#E03131",
}


def parse_args() -> argparse.Namespace:
    cache = Path.home() / "nilearn_data" / "schaefer_2018"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st-t4-dir",
        type=Path,
        default=Path("outputs/psiconnect/st_t4_interpretability_validation_seed2026"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/folds_psiconnect_nested")
    )
    parser.add_argument(
        "--atlas-img",
        type=Path,
        default=cache / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz",
    )
    parser.add_argument(
        "--atlas-labels",
        type=Path,
        default=cache / "Schaefer2018_200Parcels_7Networks_order.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/psiconnect/st_attention_fc_interpretation_seed2026"),
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifests(data_root: Path) -> tuple[dict[str, Any], list[str]]:
    manifests = []
    sample_id_orders = []
    for density in ("p0p1", "p0p2"):
        density_root = data_root / f"w40_s10_{density}"
        manifest = read_json(density_root / "psiconnect_st_manifest.json")
        if manifest.get("atlas") != "Schaefer-200":
            raise ValueError(f"Unexpected atlas in {density} manifest: {manifest.get('atlas')}")
        if int(manifest.get("n_samples", -1)) != 120:
            raise ValueError(f"Unexpected sample count in {density} manifest.")
        manifests.append(manifest)
        fold_payload = torch.load(
            density_root / "graphs_outer1_inner1.pkl",
            map_location="cpu",
            weights_only=False,
        )
        sample_id_orders.append(
            [str(value) for value in fold_payload.get("metadata", {}).get("sample_ids", [])]
        )
    sample_ids = sample_id_orders[0]
    if len(sample_ids) != 120 or sample_ids != sample_id_orders[1]:
        raise ValueError("Graph files do not contain one identical 120-observation ordering.")
    source_root = Path(str(manifests[0]["source_root"]))
    if not source_root.is_dir():
        raise FileNotFoundError(f"Manifest source directory is unavailable: {source_root}")
    source_files = sorted(source_root.glob("*task-rest*atlas-Schaefer200Parcels7Networks*.mat"))
    source_names = {path.name for path in source_files}
    unmatched_ids = [
        sample_id
        for sample_id in sample_ids
        if not any(name.startswith(f"{sample_id}_") for name in source_names)
    ]
    if unmatched_ids:
        raise ValueError(
            "Could not match every selected observation to a source derivative named for "
            f"Schaefer200Parcels7Networks: {unmatched_ids[:5]}"
        )
    return manifests[0], sample_ids


def load_subject_importance(
    st_t4_dir: Path, metadata: dict[str, Any], master_sample_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    expected_pairs = undirected_edge_pairs(NUM_NODES)
    importance_rows, subject_ids, sample_indices, folds = [], [], [], []
    component_names: list[str] | None = None
    for fold in FOLDS:
        path = st_t4_dir / f"outer_fold_{fold:02d}" / "edge_importance.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        first = np.load(path)
        second = np.load(path)
        importance = np.asarray(first["importance"], dtype=np.float32)
        if not np.array_equal(importance, second["importance"]):
            raise AssertionError(f"Fold {fold}: saved importance is not deterministic on reload.")
        source = np.asarray(first["source_roi_zero_based"], dtype=np.int64)
        target = np.asarray(first["target_roi_zero_based"], dtype=np.int64)
        pairs = np.column_stack((source, target))
        if not np.array_equal(pairs, expected_pairs):
            raise ValueError(f"Fold {fold}: edge ordering differs from canonical upper triangle.")
        indices = np.asarray(first["sample_index"], dtype=np.int64)
        ids = np.asarray(first["sample_id"], dtype=str)
        current_folds = np.asarray(first["outer_fold"], dtype=np.int64)
        seeds = np.asarray(first["seed"], dtype=np.int64)
        names = [str(value) for value in first["component_names"]]
        if importance.shape != (len(indices), NUM_EDGES):
            raise ValueError(f"Fold {fold}: unexpected importance shape {importance.shape}.")
        if not np.all(seeds == SEED) or not np.all(current_folds == fold):
            raise ValueError(f"Fold {fold}: seed/fold provenance mismatch.")
        expected_ids = np.asarray([master_sample_ids[int(index)] for index in indices], dtype=str)
        if not np.array_equal(ids, expected_ids):
            raise ValueError(f"Fold {fold}: sample IDs do not match manifest ordering.")
        if component_names is None:
            component_names = names
        elif names != component_names:
            raise ValueError("Attention component order differs across folds.")
        importance_rows.append(importance)
        subject_ids.extend(ids.tolist())
        sample_indices.extend(indices.tolist())
        folds.extend(current_folds.tolist())

        provenance = metadata.get("fold_provenance", {}).get(str(fold), {})
        checkpoint = Path(str(provenance.get("checkpoint", "")))
        if not checkpoint.is_file() or sha256(checkpoint) != provenance.get("checkpoint_sha256"):
            raise ValueError(f"Fold {fold}: checkpoint path/hash provenance failed.")

    assert component_names is not None
    if component_names != list(EXPECTED_COMPONENTS):
        raise ValueError(f"Unexpected attention components: {component_names}")
    indices_array = np.asarray(sample_indices, dtype=np.int64)
    if len(indices_array) != 120 or sorted(indices_array.tolist()) != list(range(120)):
        raise ValueError("Held-out observations are not covered exactly once across five folds.")
    if len(set(subject_ids)) != 120:
        raise ValueError("Held-out subject/session identifiers are not unique.")
    values = np.concatenate(importance_rows, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("Subject importance contains NaN or Inf.")
    return (
        values,
        np.asarray(subject_ids, dtype=str),
        indices_array,
        np.asarray(folds, dtype=np.int64),
        expected_pairs,
        component_names,
    )


def load_atlas(atlas_img_path: Path, atlas_labels_path: Path):
    if not atlas_img_path.is_file() or not atlas_labels_path.is_file():
        raise FileNotFoundError(
            "Canonical Schaefer atlas files are missing. Install them with: "
            "python -c \"from nilearn import datasets; "
            "datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)\""
        )
    atlas_rows = load_schaefer_labels(atlas_labels_path, expected_rois=NUM_NODES)
    atlas_img = nib.load(str(atlas_img_path))
    data = np.asanyarray(atlas_img.dataobj)
    unique_labels = np.unique(data.astype(np.int64))
    if not np.array_equal(unique_labels, np.arange(NUM_NODES + 1)):
        raise ValueError("Atlas image must contain background 0 and parcel labels 1..200.")
    coordinates = np.asarray(
        plotting.find_parcellation_cut_coords(labels_img=atlas_img), dtype=float
    )
    if coordinates.shape != (NUM_NODES, 3) or not np.isfinite(coordinates).all():
        raise ValueError(f"Expected exactly 200 finite MNI coordinates, got {coordinates.shape}.")
    return atlas_rows, atlas_img, coordinates


def top_k_matrix(matrix: np.ndarray, count: int) -> np.ndarray:
    pairs = undirected_edge_pairs(len(matrix))
    values = matrix[pairs[:, 0], pairs[:, 1]]
    order = np.lexsort((np.arange(len(values)), -values))[:count]
    output = np.zeros_like(matrix)
    chosen = pairs[order]
    output[chosen[:, 0], chosen[:, 1]] = values[order]
    output[chosen[:, 1], chosen[:, 0]] = values[order]
    return output


def save_connectome(
    matrix: np.ndarray,
    coordinates: np.ndarray,
    node_colors: list[str],
    node_size,
    title: str,
    stem: Path,
) -> None:
    positive = matrix[np.triu_indices_from(matrix, k=1)]
    positive = positive[positive > 0]
    display = plotting.plot_connectome(
        matrix,
        coordinates,
        node_color=node_colors,
        node_size=node_size,
        edge_cmap="viridis",
        edge_vmin=float(positive.min()) if positive.size else None,
        edge_vmax=float(positive.max()) if positive.size else None,
        edge_kwargs={"linewidth": 1.4, "alpha": 0.75},
        colorbar=True,
        display_mode="lyrz",
        title=title,
        annotate=True,
    )
    display.savefig(str(stem.with_suffix(".png")), dpi=350)
    display.savefig(str(stem.with_suffix(".pdf")), dpi=350)
    display.close()


def save_matrix_heatmap(
    group_matrix: np.ndarray,
    atlas_rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    networks = np.asarray([str(row["network"]) for row in atlas_rows], dtype=str)
    order = np.concatenate([np.flatnonzero(networks == network) for network in CANONICAL_NETWORKS])
    ordered = group_matrix[np.ix_(order, order)]
    counts = [int(np.sum(networks == network)) for network in CANONICAL_NETWORKS]
    boundaries = np.cumsum(counts)
    centres = np.r_[0, boundaries[:-1]] + np.asarray(counts) / 2
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    image = ax.imshow(ordered, cmap="magma", interpolation="nearest", aspect="equal")
    for boundary in boundaries[:-1]:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.8, alpha=0.8)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.8, alpha=0.8)
    labels = [NETWORK_DISPLAY[value] for value in CANONICAL_NETWORKS]
    ax.set_xticks(centres, labels=labels, rotation=40, ha="right")
    ax.set_yticks(centres, labels=labels)
    ax.set_xlabel("Schaefer-200 functional network")
    ax.set_ylabel("Schaefer-200 functional network")
    ax.set_title("Mean ST-BrainXDL attention importance across held-out observations")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean attention importance")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"attention_fc_importance_matrix.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_network_figures(rows: list[dict[str, object]], output_dir: Path) -> list[dict[str, object]]:
    lookup = {(row["network_1_key"], row["network_2_key"]): row for row in rows}
    matrix = np.zeros((len(CANONICAL_NETWORKS), len(CANONICAL_NETWORKS)), dtype=float)
    for i, network_1 in enumerate(CANONICAL_NETWORKS):
        for j, network_2 in enumerate(CANONICAL_NETWORKS):
            key = (network_1, network_2) if i <= j else (network_2, network_1)
            matrix[i, j] = float(lookup[key]["mean_attention_importance"])
    labels = [NETWORK_DISPLAY[value] for value in CANONICAL_NETWORKS]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(matrix, cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("Mean attention importance by functional-network pair")
    fig.colorbar(image, ax=ax, label="Mean attention importance")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"network_pair_importance_heatmap.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)

    ranked = sorted(rows, key=lambda row: float(row["mean_attention_importance"]), reverse=True)
    top = ranked[:10]
    pair_labels = [f"{row['network_1']} – {row['network_2']}" for row in top]
    values = [float(row["mean_attention_importance"]) for row in top]
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    positions = np.arange(len(top))
    ax.barh(positions, values, color="#3B82B8", edgecolor="#173F5F", linewidth=0.6)
    ax.set_yticks(positions, labels=pair_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean attention importance per possible edge")
    ax.set_title("Top functional-network pairs by mean ST-BrainXDL attention")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"top_network_pair_importance.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    return ranked


def report_markdown(
    output_dir: Path,
    metadata: dict[str, Any],
    ranking: list[dict[str, object]],
    network_ranking: list[dict[str, object]],
    validation: dict[str, bool],
) -> None:
    lines = [
        "# ST-BrainXDL Attention-Based Functional-Connectivity Interpretation",
        "",
        "## Analysis setup",
        "",
        f"- Seed: `{metadata['seed']}`.",
        f"- Outer folds: `{metadata['outer_folds']}`.",
        f"- Held-out observations: `{metadata['n_subjects']}` with fold counts `{metadata['observations_per_fold']}`.",
        f"- Atlas: `{metadata['atlas']}`; `{metadata['num_nodes']}` ROIs and `{metadata['num_edges']}` undirected off-diagonal FC candidates.",
        f"- Attention components: `{metadata['num_attention_components']}`.",
        f"- Aggregation: {metadata['edge_importance_definition']}",
        "- The analysis reuses the exact subject-level edge scores saved by `[ST-T4]`; no model was retrained or updated.",
        "",
        "### Frozen checkpoint provenance",
        "",
    ]
    for fold, row in metadata["checkpoints"].items():
        lines.append(f"- Fold `{fold}`: `{row['path']}` (SHA-256 `{row['sha256']}`).")
    lines.extend(
        [
            "",
            "## Top functional connections",
            "",
            "| Rank | Region i | Region j | Network i | Network j | Mean attention importance |",
            "| ---: | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in ranking[:20]:
        lines.append(
            f"| {row['rank']} | {row['region_i']} | {row['region_j']} | "
            f"{row['network_i']} | {row['network_j']} | "
            f"{float(row['mean_attention_importance']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Top network pairs",
            "",
            "| Rank | Network 1 | Network 2 | Type | Mean importance | Number of edges |",
            "| ---: | --- | --- | --- | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(network_ranking[:10], start=1):
        lines.append(
            f"| {rank} | {row['network_1']} | {row['network_2']} | "
            f"{row['connection_type']} | {float(row['mean_attention_importance']):.8f} | "
            f"{row['n_edges']} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `attention_top20_connectome`: anatomical locations of the 20 highest-ranked group-level attention FCs.",
            "- `attention_top1pct_connectome`: anatomical locations of the highest-ranked 1% (199) of FCs; retained as a denser supplementary view.",
            "- `attention_node_importance_connectome`: the top-20 FCs with ROI marker size proportional to summed attention involvement.",
            "- `attention_fc_importance_matrix`: the complete symmetric mean-importance matrix ordered by Schaefer-7 network.",
            "- `network_pair_importance_heatmap`: mean edge importance for all 28 within/between-network pairs.",
            "- `top_network_pair_importance`: the ten network pairs with highest mean importance per possible edge.",
            "- A circular connectome was not generated because 200 ROI labels and 199 edges were less readable than the anatomical and network-level views.",
            "",
            "## Validation",
            "",
            "| Check | Result |",
            "| --- | --- |",
        ]
    )
    for name, passed in validation.items():
        lines.append(f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The tables and figures describe which functional connections receive the",
            "largest attention weights inside the frozen seed-2026 ST-BrainXDL graph",
            "representation. `[ST-T4]` independently links this same ranking to predictive",
            "fidelity: top-ranked removal was more damaging than matched random and",
            "low-importance removal at every tested fraction. High attention or node",
            "involvement does not establish causal regional or neurobiological importance.",
            "The analysis does not establish cross-seed attribution stability.",
        ]
    )
    (output_dir / "attention_fc_interpretability_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if nib is None or plotting is None:
        raise ModuleNotFoundError(
            "This visualization command requires the optional dependencies; "
            "install BrainXDL-Open with the 'xai' extra."
        )
    if args.top_n < 20:
        raise ValueError("--top-n must be at least 20.")
    targets = [
        args.output_dir / "attention_fc_interpretability_results.md",
        args.output_dir / "all_functional_connection_importance.csv",
        args.output_dir / "attention_top20_connectome.png",
    ]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Interpretability outputs already exist; pass --overwrite.")

    st_t4_metadata = read_json(args.st_t4_dir / "st_t4_experiment_metadata.json")
    if st_t4_metadata.get("status") != "complete" or int(st_t4_metadata.get("seed", -1)) != SEED:
        raise ValueError("A complete seed-2026 ST-T4 result is required.")
    if st_t4_metadata.get("outer_folds") != list(FOLDS):
        raise ValueError("ST-T4 metadata must contain exactly outer folds 1..5.")
    if st_t4_metadata.get("edge_importance_definition") != EDGE_IMPORTANCE_DEFINITION:
        raise ValueError("ST-T4 edge-importance definition differs from the shared implementation.")
    required_t4_checks = (
        "checkpoint_fold_matching",
        "outer_test_fold_matching",
        "model_state_unchanged",
        "labels_excluded_from_ranking",
    )
    if not all(st_t4_metadata.get("validation_checks", {}).get(key) is True for key in required_t4_checks):
        raise ValueError("ST-T4 provenance/frozen-model validation is incomplete.")

    manifest, master_sample_ids = load_manifests(args.data_root)
    values, subject_ids, sample_indices, folds, pairs, components = load_subject_importance(
        args.st_t4_dir, st_t4_metadata, master_sample_ids
    )
    atlas_rows, atlas_img, coordinates = load_atlas(args.atlas_img, args.atlas_labels)
    mean, standard_deviation, group_matrix = group_importance(values, NUM_NODES, pairs)
    subject_structure_valid = True
    for subject_values in values:
        subject_matrix = edge_vector_to_matrix(subject_values, NUM_NODES, pairs)
        subject_structure_valid = subject_structure_valid and bool(
            np.array_equal(subject_matrix, subject_matrix.T)
            and np.array_equal(np.diag(subject_matrix), np.zeros(NUM_NODES))
        )
    repeat_mean, repeat_sd, repeat_matrix = group_importance(values.copy(), NUM_NODES, pairs)
    deterministic = (
        np.array_equal(mean, repeat_mean)
        and np.array_equal(standard_deviation, repeat_sd)
        and np.array_equal(group_matrix, repeat_matrix)
    )
    if not deterministic:
        raise AssertionError("Group aggregation is not deterministic.")
    if not np.allclose(group_matrix, group_matrix.T, atol=0.0, rtol=0.0):
        raise AssertionError("Group importance matrix is not symmetric.")
    if not np.array_equal(np.diag(group_matrix), np.zeros(NUM_NODES)):
        raise AssertionError("Group importance matrix diagonal is not zero.")

    ranking = rank_edges(mean, standard_deviation, pairs, atlas_rows, len(values))
    network_rows = aggregate_network_pairs(group_matrix, atlas_rows)
    node_score, node_normalised = node_involvement(group_matrix)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranking_fields = [
        "rank", "node_i", "node_j", "region_i", "region_j", "network_i", "network_j",
        "mean_attention_importance", "std_attention_importance", "n_subjects",
    ]
    write_csv(args.output_dir / "all_functional_connection_importance.csv", ranking, ranking_fields)
    write_csv(args.output_dir / "top_functional_connections.csv", ranking[: args.top_n], ranking_fields)
    network_fields = [
        "network_1", "network_2", "connection_type", "n_edges",
        "mean_attention_importance", "sum_attention_importance",
    ]
    write_csv(args.output_dir / "network_pair_importance.csv", network_rows, network_fields)
    write_csv(
        args.output_dir / "node_attention_involvement.csv",
        [
            {
                "node": index + 1,
                "region": atlas_rows[index]["label"],
                "network": atlas_rows[index]["network_display"],
                "attention_involvement": float(node_score[index]),
                "normalised_attention_involvement": float(node_normalised[index]),
                "mni_x": float(coordinates[index, 0]),
                "mni_y": float(coordinates[index, 1]),
                "mni_z": float(coordinates[index, 2]),
            }
            for index in range(NUM_NODES)
        ],
    )
    np.savez_compressed(
        args.output_dir / "subject_level_edge_importance.npz",
        subject_id=subject_ids,
        sample_index=sample_indices,
        outer_fold=folds,
        node_i=pairs[:, 0] + 1,
        node_j=pairs[:, 1] + 1,
        attention_importance=values,
        seed=np.full(len(values), SEED, dtype=np.int64),
    )
    np.savez_compressed(
        args.output_dir / "group_attention_importance.npz",
        mean_matrix=group_matrix.astype(np.float32),
        mean_edge_importance=mean.astype(np.float32),
        std_edge_importance=standard_deviation.astype(np.float32),
        node_i=pairs[:, 0] + 1,
        node_j=pairs[:, 1] + 1,
        coordinates_mni=coordinates.astype(np.float32),
        roi_labels=np.asarray([row["label"] for row in atlas_rows], dtype=str),
        network_labels=np.asarray([row["network_display"] for row in atlas_rows], dtype=str),
    )

    node_colors = [NETWORK_COLORS[str(row["network"])] for row in atlas_rows]
    save_connectome(
        top_k_matrix(group_matrix, 20), coordinates, node_colors, 8,
        "Top 20 ST-BrainXDL attention-derived functional connections",
        args.output_dir / "attention_top20_connectome",
    )
    save_connectome(
        top_k_matrix(group_matrix, max(1, int(round(NUM_EDGES * 0.01)))),
        coordinates, node_colors, 6,
        "Top 1% ST-BrainXDL attention-derived functional connections",
        args.output_dir / "attention_top1pct_connectome",
    )
    save_connectome(
        top_k_matrix(group_matrix, 20), coordinates, node_colors,
        12 + 88 * node_normalised,
        "ROI involvement in top ST-BrainXDL attention-derived connections",
        args.output_dir / "attention_node_importance_connectome",
    )
    save_matrix_heatmap(group_matrix, atlas_rows, args.output_dir)
    network_ranking = save_network_figures(network_rows, args.output_dir)

    fold_counts = {str(fold): int(np.sum(folds == fold)) for fold in FOLDS}
    validation = {
        "exactly_five_outer_folds": sorted(set(folds.tolist())) == list(FOLDS),
        "seed_exactly_2026": int(st_t4_metadata["seed"]) == SEED,
        "each_observation_processed_once": sorted(sample_indices.tolist()) == list(range(120)),
        "matching_outer_fold_checkpoint": True,
        "model_parameters_frozen_and_unchanged": True,
        "labels_excluded_from_importance": True,
        "subject_matrices_symmetric": subject_structure_valid,
        "subject_matrix_diagonals_zero": subject_structure_valid,
        "exactly_19900_candidate_edges": len(pairs) == NUM_EDGES,
        "importance_values_finite": bool(np.isfinite(values).all()),
        "results_deterministic": deterministic,
        "aggregation_identical_to_st_t4": st_t4_metadata["edge_importance_definition"] == EDGE_IMPORTANCE_DEFINITION,
        "exactly_200_valid_mni_coordinates": coordinates.shape == (NUM_NODES, 3) and bool(np.isfinite(coordinates).all()),
        "roi_order_matches_atlas": [int(row["node"]) for row in atlas_rows] == list(range(1, NUM_NODES + 1)),
    }
    if not all(validation.values()):
        raise AssertionError(f"Validation failed: {[key for key, value in validation.items() if not value]}")
    checkpoints = {
        key: {
            "path": value["checkpoint"],
            "sha256": value["checkpoint_sha256"],
        }
        for key, value in st_t4_metadata["fold_provenance"].items()
    }
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "analysis": "ST-BrainXDL attention-based FC interpretation",
        "seed": SEED,
        "outer_folds": list(FOLDS),
        "n_subjects": len(values),
        "observations_per_fold": fold_counts,
        "atlas": "Schaefer2018 200 Parcels, 7 Networks, canonical ordered FSL MNI152 1 mm label map",
        "atlas_image": str(args.atlas_img.resolve()),
        "atlas_image_sha256": sha256(args.atlas_img),
        "atlas_labels": str(args.atlas_labels.resolve()),
        "atlas_labels_sha256": sha256(args.atlas_labels),
        "coordinate_source": "nilearn.plotting.find_parcellation_cut_coords on the canonical ordered label map",
        "network_label_source": "canonical Schaefer2018_200Parcels_7Networks_order.txt",
        "roi_order_provenance": (
            "Fold manifests identify Schaefer-200; all 120 source derivative filenames identify "
            "Schaefer200Parcels7Networks; brainxdl.st.psiconnect preserves matrix column order; "
            "atlas image labels and label table are exactly 1..200 in canonical order."
        ),
        "num_nodes": NUM_NODES,
        "num_edges": NUM_EDGES,
        "num_attention_components": len(components),
        "attention_components": components,
        "edge_importance_definition": EDGE_IMPORTANCE_DEFINITION,
        "source_st_t4": str(args.st_t4_dir.resolve()),
        "source_fold_manifest": str(
            (args.data_root / "w40_s10_p0p1" / "psiconnect_st_manifest.json").resolve()
        ),
        "source_dataset_root": manifest["source_root"],
        "checkpoints": checkpoints,
        "validation": validation,
    }
    (args.output_dir / "attention_fc_interpretability_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_markdown(args.output_dir, metadata, ranking, network_ranking, validation)
    print(f"Complete: {args.output_dir / 'attention_fc_interpretability_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
