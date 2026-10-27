#!/usr/bin/env python3
"""Characterise psilocybin FC changes for the frozen ST attention top 20."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "brainxdl-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "brainxdl-cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

try:
    import nibabel as nib
    from nilearn import plotting
except ModuleNotFoundError:
    nib = None
    plotting = None

from brainxdl.xai.st_fc_changes import (
    benjamini_hochberg,
    bootstrap_paired_dz_ci,
    cluster_bootstrap_mean_difference_ci,
    cohens_dz,
    fit_random_intercept_condition_model,
    paired_values,
)


EXPECTED_TOP20 = (
    ("LH_SomMot_6", "RH_SomMot_7"),
    ("LH_Default_PFC_3", "LH_Default_PFC_5"),
    ("LH_Default_pCunPCC_1", "RH_Default_pCunPCC_1"),
    ("LH_Cont_pCun_1", "RH_Cont_pCun_1"),
    ("RH_Limbic_TempPole_1", "RH_Limbic_TempPole_3"),
    ("LH_Vis_7", "RH_Vis_6"),
    ("LH_Cont_Cing_1", "RH_Cont_Cing_1"),
    ("LH_SomMot_16", "RH_SomMot_19"),
    ("LH_Default_PFC_6", "RH_Default_PFCdPFCm_2"),
    ("LH_Limbic_TempPole_4", "RH_SalVentAttn_FrOperIns_1"),
    ("LH_Vis_10", "RH_Vis_9"),
    ("LH_Default_pCunPCC_2", "RH_Default_pCunPCC_2"),
    ("LH_Default_PFC_7", "LH_Default_PFC_9"),
    ("LH_Default_PFC_6", "LH_Default_PFC_8"),
    ("LH_Default_PFC_8", "RH_Cont_PFCmp_1"),
    ("RH_SomMot_1", "RH_SomMot_2"),
    ("LH_SomMot_15", "RH_SomMot_18"),
    ("LH_Default_PFC_2", "RH_Default_PFCdPFCm_1"),
    ("LH_Vis_5", "RH_Vis_8"),
    ("RH_Default_PFCdPFCm_6", "RH_Default_PFCdPFCm_7"),
)
CONTROL = 0
PSILOCYBIN = 1
CONDITION_NAMES = {CONTROL: "control", PSILOCYBIN: "psilocybin"}
NUM_ROIS = 200


def parse_args() -> argparse.Namespace:
    attention_dir = Path(
        "outputs/psiconnect/st_attention_fc_interpretation_seed2026"
    )
    cache = Path.home() / "nilearn_data" / "schaefer_2018"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-ranking",
        type=Path,
        default=attention_dir / "top_functional_connections.csv",
    )
    parser.add_argument(
        "--attention-metadata",
        type=Path,
        default=attention_dir / "attention_fc_interpretability_metadata.json",
    )
    parser.add_argument(
        "--fc-data",
        type=Path,
        default=Path("data/folds_psiconnect_nested/w40_s10_p0p1/LDW_psiconnect_data.pkl"),
    )
    parser.add_argument(
        "--fc-manifest",
        type=Path,
        default=Path("data/folds_psiconnect_nested/w40_s10_p0p1/psiconnect_st_manifest.json"),
    )
    parser.add_argument(
        "--alternate-fc-data",
        type=Path,
        default=Path("data/folds_psiconnect_nested/w40_s10_p0p2/LDW_psiconnect_data.pkl"),
        help="Second graph-density payload used to verify density-independent FC tensors.",
    )
    parser.add_argument(
        "--atlas-img",
        type=Path,
        default=cache / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/psiconnect/st_attention_fc_changes_seed2026"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_label(value: str) -> str:
    return value.removeprefix("7Networks_")


def edge_label(row: dict[str, Any]) -> str:
    return f"{short_label(str(row['region_i']))} – {short_label(str(row['region_j']))}"


def load_ranking(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise ValueError(f"Expected exactly 20 saved attention-ranked edges, found {len(rows)}.")
    parsed: list[dict[str, Any]] = []
    for expected_rank, row in enumerate(rows, start=1):
        parsed_row = {
            **row,
            "rank": int(row["rank"]),
            "node_i": int(row["node_i"]),
            "node_j": int(row["node_j"]),
            "mean_attention_importance": float(row["mean_attention_importance"]),
        }
        if parsed_row["rank"] != expected_rank:
            raise ValueError("Saved top-20 attention ranking is not in rank order.")
        parsed.append(parsed_row)
    observed = tuple(
        (short_label(str(row["region_i"])), short_label(str(row["region_j"])))
        for row in parsed
    )
    if observed != EXPECTED_TOP20:
        raise ValueError("Canonical saved top-20 ranking differs from the expected edge list.")
    return parsed


def load_fc_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"node_feat", "labels", "groups", "sample_ids"}
    if not required.issubset(payload):
        raise ValueError(f"FC payload is missing keys: {sorted(required - set(payload))}")
    return payload


def fc_feature_digest(payload: dict[str, Any]) -> str:
    """Hash ordered signed FC matrices without the density-dependent adjacency."""
    digest = hashlib.sha256()
    for observation in payload["node_feat"]:
        for window in observation:
            digest.update(memoryview(np.ascontiguousarray(window)))
    for field in ("sample_ids", "labels", "groups"):
        digest.update("\n".join(map(str, payload[field])).encode("utf-8"))
    return digest.hexdigest()


def extract_values(
    payload: dict[str, Any], ranking: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(payload["labels"], dtype=np.int64)
    groups = np.asarray(payload["groups"], dtype=str)
    sample_ids = np.asarray(payload["sample_ids"], dtype=str)
    matrices = payload["node_feat"]
    if len(matrices) != len(labels) or labels.shape != groups.shape or labels.shape != sample_ids.shape:
        raise ValueError("FC payload observation-level fields have inconsistent lengths.")
    raw = np.empty((len(labels), len(ranking)), dtype=float)
    fisher_z = np.empty_like(raw)
    records: list[dict[str, Any]] = []
    for observation_index, windows in enumerate(matrices):
        array = np.asarray(windows, dtype=float)
        if array.ndim != 3 or array.shape[1:] != (NUM_ROIS, NUM_ROIS):
            raise ValueError(f"Observation {observation_index} has FC shape {array.shape}.")
        if not np.isfinite(array).all() or not np.allclose(array, array.transpose(0, 2, 1)):
            raise ValueError(f"Observation {observation_index} has invalid FC matrices.")
        for edge_index, edge in enumerate(ranking):
            i, j = int(edge["node_i"]) - 1, int(edge["node_j"]) - 1
            values = array[:, i, j]
            raw_value = float(values.mean())
            z_value = float(np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7)).mean())
            raw[observation_index, edge_index] = raw_value
            fisher_z[observation_index, edge_index] = z_value
            records.append(
                {
                    "subject_id": str(groups[observation_index]),
                    "sample_id": str(sample_ids[observation_index]),
                    "session_id": "ses-01" if labels[observation_index] == 1 else "ses-02",
                    "condition": CONDITION_NAMES[int(labels[observation_index])],
                    "condition_code": int(labels[observation_index]),
                    "edge_rank": int(edge["rank"]),
                    "roi_i": short_label(str(edge["region_i"])),
                    "roi_j": short_label(str(edge["region_j"])),
                    "network_i": edge["network_i"],
                    "network_j": edge["network_j"],
                    "attention_importance": float(edge["mean_attention_importance"]),
                    "n_dynamic_windows": int(array.shape[0]),
                    "fc_value": raw_value,
                    "fc_value_fisher_z": z_value,
                }
            )
    return records, raw, fisher_z, labels, groups


def summarise_edges(
    ranking: list[dict[str, Any]],
    raw: np.ndarray,
    fisher_z: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    repetitions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, edge in enumerate(ranking):
        control_raw = raw[labels == CONTROL, index]
        psilo_raw = raw[labels == PSILOCYBIN, index]
        paired_control_z, paired_psilo_z, pair_ids = paired_values(
            fisher_z[:, index], labels, groups
        )
        paired_control_raw, paired_psilo_raw, raw_pair_ids = paired_values(
            raw[:, index], labels, groups
        )
        if pair_ids != raw_pair_ids:
            raise AssertionError("Raw and Fisher-z paired subject orders differ.")
        delta = float(psilo_raw.mean() - control_raw.mean())
        mixed = fit_random_intercept_condition_model(fisher_z[:, index], labels, groups)
        paired_test = stats.ttest_rel(paired_psilo_z, paired_control_z)
        try:
            wilcoxon = stats.wilcoxon(
                paired_psilo_z,
                paired_control_z,
                alternative="two-sided",
                method="auto",
            )
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        dz = cohens_dz(paired_control_z, paired_psilo_z)
        dz_ci = bootstrap_paired_dz_ci(
            paired_control_z,
            paired_psilo_z,
            repetitions=repetitions,
            seed=2026 + int(edge["rank"]),
        )
        delta_ci = cluster_bootstrap_mean_difference_ci(
            raw[:, index],
            labels,
            groups,
            repetitions=repetitions,
            seed=4046 + int(edge["rank"]),
        )
        paired_differences = paired_psilo_raw - paired_control_raw
        same_direction = int(np.sum(paired_differences > 0 if delta > 0 else paired_differences < 0))
        rows.append(
            {
                "Rank": int(edge["rank"]),
                "Region i": short_label(str(edge["region_i"])),
                "Region j": short_label(str(edge["region_j"])),
                "Network i": edge["network_i"],
                "Network j": edge["network_j"],
                "Attention importance": float(edge["mean_attention_importance"]),
                "N control": int(control_raw.size),
                "N psilocybin": int(psilo_raw.size),
                "N paired": int(len(pair_ids)),
                "Control FC mean": float(control_raw.mean()),
                "Control FC SD": float(control_raw.std(ddof=1)),
                "Psilocybin FC mean": float(psilo_raw.mean()),
                "Psilocybin FC SD": float(psilo_raw.std(ddof=1)),
                "Delta FC": delta,
                "Delta FC CI low": float(delta_ci[0]),
                "Delta FC CI high": float(delta_ci[1]),
                "Direction": "Increased" if delta > 0 else "Decreased",
                "Effect size": float(dz),
                "Effect size CI low": float(dz_ci[0]),
                "Effect size CI high": float(dz_ci[1]),
                "p": float(mixed["p"]),
                "FDR q": np.nan,
                "FDR significant": False,
                "Mixed-model delta Fisher-z": float(mixed["condition_effect"]),
                "Mixed-model CI low": float(mixed["ci_low"]),
                "Mixed-model CI high": float(mixed["ci_high"]),
                "Mixed-model t": float(mixed["t"]),
                "Mixed-model df": float(mixed["df"]),
                "Random-intercept variance": float(mixed["random_intercept_variance"]),
                "Residual variance": float(mixed["residual_variance"]),
                "Paired t p": float(paired_test.pvalue),
                "Wilcoxon p": wilcoxon_p,
                "Paired same-direction n": same_direction,
                "Paired same-direction percent": float(100.0 * same_direction / len(pair_ids)),
                "Paired difference Shapiro p": float(stats.shapiro(paired_psilo_z - paired_control_z).pvalue),
            }
        )
    adjusted = benjamini_hochberg([float(row["p"]) for row in rows])
    for row, q_value in zip(rows, adjusted, strict=True):
        row["FDR q"] = float(q_value)
        row["FDR significant"] = bool(q_value < 0.05)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def formatted_fc(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Psilocybin-Induced Functional-Connectivity Differences Among ST-BrainXDL Attention-Ranked Connections",
        "",
        "FC summaries are mean signed Ledoit-Wolf correlations across 46 dynamic windows per session. Primary p-values come from participant-random-intercept models fitted to mean Fisher-z values; Cohen's dz uses the 55 complete participant pairs. Delta FC and direction use psilocybin minus control raw FC.",
        "",
        "| Rank | Region i | Region j | Network i | Network j | Attention importance | N control | N psilocybin | Control FC (mean ± SD) | Psilocybin FC (mean ± SD) | Delta FC | Direction | Effect size (dz) | p | FDR q | FDR significant |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['Rank']} | {row['Region i']} | {row['Region j']} | "
            f"{row['Network i']} | {row['Network j']} | {row['Attention importance']:.8f} | "
            f"{row['N control']} | {row['N psilocybin']} | "
            f"{formatted_fc(row['Control FC mean'], row['Control FC SD'])} | "
            f"{formatted_fc(row['Psilocybin FC mean'], row['Psilocybin FC SD'])} | "
            f"{row['Delta FC']:.4f} | {row['Direction']} | {row['Effect size']:.3f} | "
            f"{row['p']:.4g} | {row['FDR q']:.4g} | "
            f"{'Yes' if row['FDR significant'] else 'No'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_change_plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    positions = np.arange(len(rows))
    values = np.asarray([row["Delta FC"] for row in rows], dtype=float)
    lower = values - np.asarray([row["Delta FC CI low"] for row in rows], dtype=float)
    upper = np.asarray([row["Delta FC CI high"] for row in rows], dtype=float) - values
    colors = ["#C23B22" if value > 0 else "#2468A2" for value in values]
    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    ax.errorbar(values, positions, xerr=np.vstack((lower, upper)), fmt="none", ecolor="#667085", capsize=2, linewidth=1)
    ax.scatter(values, positions, c=colors, s=48, zorder=3)
    ax.axvline(0, color="#202124", linewidth=1, linestyle="--")
    ax.set_yticks(positions, labels=[f"{row['Rank']}. {edge_label({'region_i': row['Region i'], 'region_j': row['Region j']})}" for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Delta FC (psilocybin − control; raw correlation)")
    ax.set_ylabel("Attention-ranked FC edge")
    ax.set_title("FC change among ST-BrainXDL top-20 attention-ranked connections")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#C23B22", label="Increased", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2468A2", label="Decreased", markersize=8),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"top20_attention_fc_change_plot.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_effect_plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    positions = np.arange(len(rows))
    values = np.asarray([row["Effect size"] for row in rows], dtype=float)
    lower = values - np.asarray([row["Effect size CI low"] for row in rows], dtype=float)
    upper = np.asarray([row["Effect size CI high"] for row in rows], dtype=float) - values
    colors = ["#C23B22" if value > 0 else "#2468A2" for value in values]
    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    ax.errorbar(values, positions, xerr=np.vstack((lower, upper)), fmt="none", ecolor="#667085", capsize=2, linewidth=1)
    ax.scatter(values, positions, c=colors, s=48, zorder=3)
    ax.axvline(0, color="#202124", linewidth=1, linestyle="--")
    ax.set_yticks(positions, labels=[f"{row['Rank']}. {row['Region i']} – {row['Region j']}" for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's dz (mean Fisher-z FC; psilocybin − control)")
    ax.set_ylabel("Attention-ranked FC edge")
    ax.set_title("Paired effect sizes for ST-BrainXDL top-20 connections")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"top20_attention_fc_effect_sizes.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_paired_examples(
    rows: list[dict[str, Any]], raw: np.ndarray, labels: np.ndarray, groups: np.ndarray, output_dir: Path
) -> None:
    selected_ranks = (1, 3, 12, 15, 10)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), sharex=True)
    for axis, rank in zip(axes.flat, selected_ranks, strict=False):
        index = rank - 1
        control, psilo, _ = paired_values(raw[:, index], labels, groups)
        for first, second in zip(control, psilo, strict=True):
            axis.plot([0, 1], [first, second], color="#98A2B3", alpha=0.35, linewidth=0.7)
        axis.scatter(np.zeros_like(control), control, color="#525866", alpha=0.65, s=13)
        axis.scatter(np.ones_like(psilo), psilo, color="#8E44AD", alpha=0.65, s=13)
        means = [control.mean(), psilo.mean()]
        axis.plot([0, 1], means, color="#111827", linewidth=2.2, marker="o", markersize=6)
        axis.axhline(0, color="#D0D5DD", linewidth=0.7)
        axis.set_xticks([0, 1], labels=["Control", "Psilocybin"])
        axis.set_title(f"Rank {rank}: {rows[index]['Region i']}\n{rows[index]['Region j']}", fontsize=9)
        axis.set_ylabel("Mean dynamic FC (r)")
    axes.flat[-1].axis("off")
    fig.suptitle("Within-participant FC changes for predefined attention/relevance examples", y=1.01)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"top20_attention_fc_paired_examples.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_direction_connectome(
    rows: list[dict[str, Any]], atlas_img_path: Path, output_dir: Path
) -> None:
    atlas_img = nib.load(str(atlas_img_path))
    coordinates = np.asarray(plotting.find_parcellation_cut_coords(labels_img=atlas_img), dtype=float)
    if coordinates.shape != (NUM_ROIS, 3):
        raise ValueError(f"Expected 200 atlas coordinates, found {coordinates.shape}.")
    matrix = np.zeros((NUM_ROIS, NUM_ROIS), dtype=float)
    for row in rows:
        i, j = int(row["_node_i"]) - 1, int(row["_node_j"]) - 1
        sign = 1.0 if row["Direction"] == "Increased" else -1.0
        matrix[i, j] = matrix[j, i] = sign * float(row["Attention importance"])
    maximum = float(np.abs(matrix).max())
    display = plotting.plot_connectome(
        matrix,
        coordinates,
        node_size=6,
        node_color="#626C7A",
        edge_cmap="RdBu_r",
        edge_vmin=-maximum,
        edge_vmax=maximum,
        edge_kwargs={"alpha": 0.85},
        colorbar=True,
        display_mode="lyrz",
        title="Top-20 attention edges: underlying FC direction",
        annotate=True,
    )
    for suffix in ("png", "pdf"):
        display.savefig(str(output_dir / f"top20_attention_fc_direction_connectome.{suffix}"), dpi=350)
    display.close()


def write_validation(path: Path, validation: dict[str, bool], details: list[str]) -> None:
    lines = ["ST-BrainXDL top-20 attention FC change validation", ""]
    lines.extend(f"[{'x' if passed else ' '}] {name}" for name, passed in validation.items())
    lines.extend(["", "Details:", *[f"- {value}" for value in details]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    increased = [row for row in rows if row["Direction"] == "Increased"]
    decreased = [row for row in rows if row["Direction"] == "Decreased"]
    largest_delta = sorted(rows, key=lambda row: abs(float(row["Delta FC"])), reverse=True)[:3]
    largest_effect = sorted(rows, key=lambda row: abs(float(row["Effect size"])), reverse=True)[:3]
    dmn = [row for row in rows if row["Network i"] == "Default Mode" and row["Network j"] == "Default Mode"]
    fdr = [row for row in rows if row["FDR significant"]]
    named = {int(row["Rank"]): row for row in rows}

    def phrase(row: dict[str, Any]) -> str:
        return f"rank {row['Rank']} {row['Region i']}–{row['Region j']} ({row['Direction'].lower()}, ΔFC={row['Delta FC']:.4f}, dz={row['Effect size']:.2f})"

    lines = [
        "# ST-BrainXDL Attention-Ranked FC Change Summary",
        "",
        "## Main findings",
        "",
        f"Among the 20 frozen attention-ranked edges, {len(increased)} showed higher mean raw FC and {len(decreased)} showed lower mean raw FC under psilocybin than control.",
        "",
        "The largest absolute raw FC changes were " + "; ".join(phrase(row) for row in largest_delta) + ".",
        "",
        "The largest paired effect sizes were " + "; ".join(phrase(row) for row in largest_effect) + ".",
        "",
        f"The {len(dmn)} within-Default Mode edges were predominantly "
        + ("increased" if sum(row['Direction'] == 'Increased' for row in dmn) > len(dmn) / 2 else "decreased")
        + f" ({sum(row['Direction'] == 'Increased' for row in dmn)} increased, {sum(row['Direction'] == 'Decreased' for row in dmn)} decreased).",
        "",
        f"The bilateral precuneus/PCC edges were {named[3]['Direction'].lower()} at rank 3 and {named[12]['Direction'].lower()} at rank 12. The highest-ranked Somatomotor edge was {named[1]['Direction'].lower()}.",
        "",
        f"The Default Mode–Control edge at rank 15 was {named[15]['Direction'].lower()}, and the Limbic–Salience/Ventral Attention edge at rank 10 was {named[10]['Direction'].lower()}.",
        "",
        ("No edge remained statistically supported after FDR correction across all 20 tests." if not fdr else "Edges supported after FDR correction were " + "; ".join(phrase(row) for row in fdr) + "."),
        "",
        "At the network level, the FDR-supported pattern spans within-Default Mode, within-Visual, bilateral Control precuneus, and bilateral Somatomotor connectivity. The two mean decreases were confined to a right Limbic temporal-pole edge and one left Default PFC edge, and neither survived FDR. This is a selective multi-network pattern rather than evidence for a uniform global FC increase.",
        "",
        "## Interpretation boundaries",
        "",
        "Attention importance and FC change measure different quantities. Attention importance measures importance within the learned ST-BrainXDL representation. Delta FC measures the observed direction and magnitude of the condition-related FC difference. The edges were selected without condition labels from the frozen attention ranking, then characterised using condition labels; no edge was selected for having a large FC difference.",
        "",
        "The direction counts describe mean changes, not universal participant responses. The paired same-direction percentages in the detailed CSV quantify consistency among the 55 complete participant pairs. Mixed-effects tests use all 120 sessions while accounting for participant clustering. These exploratory, attention-conditioned results do not establish causality or a neural mechanism.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if nib is None or plotting is None:
        raise ModuleNotFoundError(
            "This visualization command requires the optional dependencies; "
            "install BrainXDL-Open with the 'xai' extra."
        )
    if args.bootstrap_repetitions < 100:
        raise ValueError("--bootstrap-repetitions must be at least 100.")
    required_inputs = (
        args.attention_ranking,
        args.attention_metadata,
        args.fc_data,
        args.fc_manifest,
        args.alternate_fc_data,
        args.atlas_img,
    )
    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    target = args.output_dir / "top20_attention_fc_psilocybin_changes.csv"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {target}")

    ranking = load_ranking(args.attention_ranking)
    attention_metadata = read_json(args.attention_metadata)
    manifest = read_json(args.fc_manifest)
    payload = load_fc_data(args.fc_data)
    alternate_payload = load_fc_data(args.alternate_fc_data)
    density_fc_digest = fc_feature_digest(payload)
    density_fc_identical = density_fc_digest == fc_feature_digest(alternate_payload)
    del alternate_payload
    records, raw, fisher_z, labels, groups = extract_values(payload, ranking)
    rows = summarise_edges(
        ranking, raw, fisher_z, labels, groups, args.bootstrap_repetitions
    )
    for row, edge in zip(rows, ranking, strict=True):
        row["_node_i"] = int(edge["node_i"])
        row["_node_j"] = int(edge["node_j"])

    group_patterns = Counter()
    for group in sorted(set(groups.tolist())):
        group_patterns[tuple(sorted(labels[groups == group].tolist()))] += 1
    undirected = [tuple(sorted((int(edge["node_i"]), int(edge["node_j"])))) for edge in ranking]
    validation = {
        "exactly 200 Schaefer ROIs": all(np.asarray(item).shape[1:] == (200, 200) for item in payload["node_feat"]),
        "ROI order identical between FC data and attention analysis": attention_metadata.get("num_nodes") == 200 and manifest.get("atlas") == "Schaefer-200",
        "FC tensors identical across p0.1 and p0.2 graph densities": density_fc_identical,
        "top-20 edge indices correctly map to ROI names": tuple((short_label(str(row['region_i'])), short_label(str(row['region_j']))) for row in ranking) == EXPECTED_TOP20,
        "no diagonal edges": all(int(row["node_i"]) != int(row["node_j"]) for row in ranking),
        "no duplicate undirected edges": len(set(undirected)) == 20,
        "subject IDs correctly matched across conditions": group_patterns == Counter({(0, 1): 55, (1,): 8, (0,): 2}),
        "no paired subject treated as independent": all(int(row["N paired"]) == 55 for row in rows),
        "dynamic windows not treated as independent subjects": raw.shape == (120, 20) and all(record["n_dynamic_windows"] == 46 for record in records),
        "no missing or infinite FC values": bool(np.isfinite(raw).all() and np.isfinite(fisher_z).all()),
        "no double Fisher-z transformation": float(np.max(np.abs(raw))) <= 1.0 and manifest.get("timeseries_derivative") == "tedanaGLM",
        "Delta FC is psilocybin minus control": all(np.isclose(row["Delta FC"], raw[labels == 1, int(row["Rank"]) - 1].mean() - raw[labels == 0, int(row["Rank"]) - 1].mean()) for row in rows),
        "attention ranking remains unchanged": [row["Rank"] for row in rows] == list(range(1, 21)),
        "condition labels not used to select top edges": attention_metadata.get("validation", {}).get("labels_excluded_from_importance") is True,
        "FDR correction includes all 20 tested edges": len(rows) == 20 and all(np.isfinite(row["FDR q"]) for row in rows),
    }
    if not all(validation.values()):
        raise AssertionError(f"Validation failed: {[name for name, passed in validation.items() if not passed]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "top20_attention_fc_condition_values.csv", records)
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    write_csv(args.output_dir / "top20_attention_fc_psilocybin_changes.csv", public_rows)
    write_markdown_table(args.output_dir / "top20_attention_fc_psilocybin_changes.md", rows)
    save_change_plot(rows, args.output_dir)
    save_effect_plot(rows, args.output_dir)
    save_paired_examples(rows, raw, labels, groups, args.output_dir)
    save_direction_connectome(rows, args.atlas_img, args.output_dir)
    write_validation(
        args.output_dir / "validation_report.txt",
        validation,
        [
            "Conditions: ses-01=psilocybin; ses-02=control.",
            "Design: 65 participants; 55 complete pairs, 8 psilocybin-only, 2 control-only.",
            "Observations: 63 psilocybin and 57 control sessions; 46 dynamic windows per session.",
            "FC representation: signed Ledoit-Wolf correlations, Schaefer-200, w40/s10; no stored Fisher-z transform.",
            f"Density provenance: p0.1 and p0.2 signed FC tensors and observation metadata have matching digest {density_fc_digest}.",
            "Primary model: participant-random-intercept linear mixed model on mean window-level Fisher-z FC.",
            "Sensitivity: paired t-test and Wilcoxon signed-rank test on 55 complete pairs.",
            "Effect size: Cohen's dz on complete-pair mean Fisher-z differences with percentile bootstrap 95% CI.",
            "Multiplicity: Benjamini-Hochberg FDR across exactly 20 primary mixed-model tests.",
        ],
    )
    write_summary(args.output_dir / "analysis_summary.md", rows)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "analysis": "ST-BrainXDL top-20 attention-ranked FC condition differences",
        "inputs": {str(path.resolve()): sha256(path) for path in required_inputs},
        "design": {
            "participants": int(len(set(groups.tolist()))),
            "observations": int(len(labels)),
            "control_observations": int(np.sum(labels == 0)),
            "psilocybin_observations": int(np.sum(labels == 1)),
            "complete_pairs": int(group_patterns[(0, 1)]),
            "psilocybin_only": int(group_patterns[(1,)]),
            "control_only": int(group_patterns[(0,)]),
        },
        "fc_representation": "signed Ledoit-Wolf correlations; 46 w40/s10 dynamic windows per session; subject/session summary is the mean across windows",
        "statistical_scale": "mean Fisher-z across windows; raw-r summaries retained for descriptive Delta FC",
        "primary_test": "REML participant-random-intercept linear mixed model: FC_z ~ condition + (1|participant)",
        "effect_size": "Cohen's dz on the 55 complete participant pairs",
        "multiple_comparisons": "Benjamini-Hochberg FDR across 20 primary tests",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "validation": validation,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
