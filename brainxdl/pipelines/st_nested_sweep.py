"""Strict nested-CV planning utilities for PsiConnect ST improvement sweeps."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from types import SimpleNamespace

from brainxdl.pipelines.st_workflow import run_st_training_workflow
from brainxdl.st.psiconnect import get_grouped_nested_fold_indices
from brainxdl.st.psiconnect import load_psiconnect_timeseries
from brainxdl.st.psiconnect import prepare_psiconnect_st


BASELINE_MODEL = {
    "model_name": "baseline",
    "hidden_dim": 32,
    "dropout": 0.5,
    "complexity_rank": 0,
    "model_params_override": {
        "y_hidden_dim": [32],
        "classifier_dropout": 0.5,
    },
}


@dataclass(frozen=True)
class SweepPlan:
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    graph_configs: list[dict[str, Any]]
    model_configs: list[dict[str, Any]]
    loss_weights: list[float]
    stage_a: list[dict[str, Any]]
    stage_b_planned: list[dict[str, Any]]
    selected_confirmation_runs: int


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Sweep config must be a mapping")
    return config


def make_run_id(prefix: str = "st_nested_sweep") -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def validate_sweep_config(config: dict[str, Any]) -> None:
    graph = config["graph_window_grid"]
    for length in graph["window_length"]:
        if int(length) <= 1:
            raise ValueError("window_length values must be > 1")
    for stride in graph["window_stride"]:
        if int(stride) <= 0:
            raise ValueError("window_stride values must be > 0")
    for length, stride in product(graph["window_length"], graph["window_stride"]):
        if int(stride) > int(length):
            raise ValueError("window_stride must be <= window_length")
    for density in graph["graph_density"]:
        if not 0 < float(density) < 1:
            raise ValueError("graph_density values must be in (0, 1)")
    if config["selection_metric"] != "val_log_loss":
        raise ValueError("Nested sweep selection_metric must be val_log_loss")
    if len(config["search_seeds"]) != 1:
        raise ValueError("Default search must use exactly one seed per candidate")
    if len(config["confirmation_seeds"]) < 1:
        raise ValueError("At least one confirmation seed is required")


def graph_window_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["graph_window_grid"]
    rows = []
    for window_length, window_stride, density in product(
        grid["window_length"],
        grid["window_stride"],
        grid["graph_density"],
    ):
        rows.append(
            {
                "window_length": int(window_length),
                "window_stride": int(window_stride),
                "graph_density": float(density),
                "graph_semantics": "proportional_density_threshold_p",
            }
        )
    return rows


def model_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = []
    baseline_hidden = int(config.get("baseline_hidden_dim", 32))
    baseline_dropout = float(config.get("baseline_dropout", 0.5))
    for item in config["model_grid"]:
        name = item["name"]
        if name == "baseline":
            hidden = baseline_hidden
            dropout = baseline_dropout
            rank = 0
        elif name == "wider":
            hidden = baseline_hidden * int(item.get("hidden_dim_multiplier", 2))
            dropout = baseline_dropout
            rank = 1
        elif name == "stronger_dropout":
            hidden = baseline_hidden
            dropout = float(item.get("dropout", 0.5))
            rank = 2
        else:
            raise ValueError(f"Unsupported model grid item: {name}")
        models.append(
            {
                "model_name": name,
                "hidden_dim": hidden,
                "dropout": dropout,
                "complexity_rank": rank,
                "model_params_override": {
                    "y_hidden_dim": [hidden],
                    "classifier_dropout": dropout,
                },
            }
        )
    return models


def _config_id(prefix: str, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def build_stage_a_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config["search_seeds"][0])
    baseline_loss = float(config.get("baseline_loss_weight", 1.0))
    rows = []
    for graph in graph_window_configs(config):
        row = {
            **graph,
            **BASELINE_MODEL,
            "stage": "A",
            "loss_weight": baseline_loss,
            "search_seed": seed,
        }
        row["config_id"] = _config_id("stageA", row)
        rows.append(row)
    return rows


def build_stage_b_candidates(
    config: dict[str, Any],
    retained_graphs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seed = int(config["search_seeds"][0])
    rows = []
    for graph, model, loss_weight in product(
        retained_graphs,
        model_configs(config),
        config["loss_weights"],
    ):
        row = {
            **graph,
            **model,
            "stage": "B",
            "loss_weight": float(loss_weight),
            "search_seed": seed,
        }
        row["config_id"] = _config_id("stageB", row)
        rows.append(row)
    return rows


def select_stage_b_candidates_from_stage_a(
    config: dict[str, Any],
    run_dir: Path,
    *,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Stage B candidates from completed Stage A validation summaries."""
    top_k = int(top_k if top_k is not None else config["stage_a_top_k"])
    stage_a_rows = load_candidate_table(run_dir, "A")
    outer_splits = int(config["outer_splits"])
    scored = []
    for row in stage_a_rows:
        config_id = str(row["config_id"])
        seed = int(float(row["search_seed"]))
        summaries = []
        for outer_fold in range(1, outer_splits + 1):
            path = candidate_inner_summary_path(
                run_dir,
                outer_fold=outer_fold,
                stage="A",
                config_id=config_id,
                seed=seed,
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing Stage A inner-validation summary: {path}"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "success":
                raise ValueError(f"Stage A summary is not successful: {path}")
            summaries.append(payload)
        losses = [float(payload["mean_val_log_loss"]) for payload in summaries]
        scored.append(
            {
                **row,
                "seed": seed,
                "completed_outer_folds": len(summaries),
                "mean_val_log_loss": float(np.mean(losses)),
                "std_val_log_loss": float(np.std(losses)),
                "outer_test_used_for_selection": False,
            }
        )
    ranked = sorted(
        scored,
        key=lambda row: (
            float(row["mean_val_log_loss"]),
            int(float(row.get("complexity_rank", 999))),
            float(row.get("std_val_log_loss", 0.0)),
            str(row["config_id"]),
        ),
    )
    retained_graphs = [
        {
            "window_length": int(float(row["window_length"])),
            "window_stride": int(float(row["window_stride"])),
            "graph_density": float(row["graph_density"]),
            "graph_semantics": row["graph_semantics"],
        }
        for row in ranked[:top_k]
    ]
    return ranked[:top_k], build_stage_b_candidates(config, retained_graphs)


def select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by validation log loss only, with deterministic non-test tie-breaks."""
    if not rows:
        raise ValueError("No candidates to select")
    return min(
        rows,
        key=lambda row: (
            float(row["mean_val_log_loss"]),
            int(row.get("complexity_rank", 999)),
            float(row.get("std_val_log_loss", 0.0)),
            str(row["config_id"]),
        ),
    )


def assert_group_disjoint(
    train_subjects: Iterable[str],
    val_subjects: Iterable[str],
    test_subjects: Iterable[str],
) -> None:
    train = set(map(str, train_subjects))
    val = set(map(str, val_subjects))
    test = set(map(str, test_subjects))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def build_split_manifest(
    labels: np.ndarray,
    groups: np.ndarray,
    sample_ids: list[str],
    *,
    outer_splits: int,
    inner_splits: int,
    seed: int,
) -> dict[str, Any]:
    folds = get_grouped_nested_fold_indices(
        labels,
        groups,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        seed=seed,
    )
    manifest = {"folds": {}}
    for fold_key, split_indices in folds.items():
        manifest["folds"][fold_key] = {}
        split_subjects = {}
        for split, indices in split_indices.items():
            subjects = [str(groups[index]) for index in indices]
            split_subjects[split] = subjects
            manifest["folds"][fold_key][split] = {
                "indices": [int(index) for index in indices],
                "subject_ids": subjects,
                "sample_ids": [sample_ids[index] for index in indices],
                "labels": [int(labels[index]) for index in indices],
            }
        assert_group_disjoint(
            split_subjects["train"],
            split_subjects["val"],
            split_subjects["test"],
        )
    return manifest


def reproducibility_metadata() -> dict[str, Any]:
    def command_text(command: list[str]) -> str:
        try:
            return subprocess.check_output(command, text=True).strip()
        except Exception:
            return "unavailable"

    return {
        "git_commit": command_text(["git", "rev-parse", "HEAD"]),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def create_plan(config: dict[str, Any], run_id: str | None = None) -> SweepPlan:
    validate_sweep_config(config)
    run_id = run_id or make_run_id()
    run_dir = Path(config.get("run_root", "./runs/st_nested_sweep")) / run_id
    stage_a = build_stage_a_candidates(config)
    retained = graph_window_configs(config)[: int(config["stage_a_top_k"])]
    stage_b = build_stage_b_candidates(config, retained)
    return SweepPlan(
        run_id=run_id,
        run_dir=run_dir,
        config=config,
        graph_configs=graph_window_configs(config),
        model_configs=model_configs(config),
        loss_weights=[float(value) for value in config["loss_weights"]],
        stage_a=stage_a,
        stage_b_planned=stage_b,
        selected_confirmation_runs=len(config["confirmation_seeds"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plan_artifacts(plan: SweepPlan, *, dry_run: bool = False) -> None:
    plan.run_dir.mkdir(parents=True, exist_ok=False)
    (plan.run_dir / "sweep_config.yaml").write_text(
        yaml.safe_dump(plan.config, sort_keys=False), encoding="utf-8"
    )
    (plan.run_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(plan.run_dir / "summary" / "stage_a_plan.csv", plan.stage_a)
    write_csv(plan.run_dir / "summary" / "stage_b_plan_top_k_placeholder.csv", plan.stage_b_planned)
    inner_splits = int(plan.config["inner_splits"])
    search_runs_per_outer = (len(plan.stage_a) + len(plan.stage_b_planned)) * inner_splits
    summary = {
        "run_id": plan.run_id,
        "dry_run": dry_run,
        "graph_window_configurations": len(plan.graph_configs),
        "stage_a_candidates_per_outer_fold": len(plan.stage_a),
        "stage_a_training_runs_per_outer_fold": len(plan.stage_a) * inner_splits,
        "stage_b_candidates_per_outer_fold": len(plan.stage_b_planned),
        "stage_b_training_runs_per_outer_fold": len(plan.stage_b_planned) * inner_splits,
        "inner_folds_per_candidate": inner_splits,
        "confirmation_runs_per_outer_fold": plan.selected_confirmation_runs,
        "expected_training_runs_per_outer_fold": (
            search_runs_per_outer + plan.selected_confirmation_runs
        ),
        "outer_folds": int(plan.config["outer_splits"]),
        "expected_total_training_runs": int(plan.config["outer_splits"])
        * (
            search_runs_per_outer + plan.selected_confirmation_runs
        ),
        "selection_metric": plan.config["selection_metric"],
        "outer_test_used_for_selection": False,
        "subject_grouping_verified": False,
    }
    timeseries_root = Path(plan.config.get("timeseries_root", ""))
    if timeseries_root.is_dir():
        data, labels, groups, sample_ids = load_psiconnect_timeseries(
            timeseries_root,
            expected_rois=int(plan.config.get("expected_rois", 200)),
            matched_eeg_root=plan.config.get("matched_eeg_root"),
        )
        split_manifest = build_split_manifest(
            labels,
            groups,
            sample_ids,
            outer_splits=int(plan.config["outer_splits"]),
            inner_splits=int(plan.config["inner_splits"]),
            seed=int(plan.config.get("split_seed", 42)),
        )
        split_manifest["n_samples"] = len(data)
        split_manifest["n_participants"] = int(len(np.unique(groups)))
        (plan.run_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["subject_grouping_verified"] = True
        summary["n_samples"] = len(data)
        summary["n_participants"] = int(len(np.unique(groups)))
    else:
        summary["subject_grouping_warning"] = (
            f"timeseries_root not available in this environment: {timeseries_root}"
        )
    (plan.run_dir / "summary" / "dry_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def print_plan(plan: SweepPlan) -> None:
    inner_splits = int(plan.config["inner_splits"])
    per_fold = (
        (len(plan.stage_a) + len(plan.stage_b_planned)) * inner_splits
        + plan.selected_confirmation_runs
    )
    total = per_fold * int(plan.config["outer_splits"])
    print(f"Run ID: {plan.run_id}")
    print(f"Run directory: {plan.run_dir}")
    print(f"Graph/window configurations: {len(plan.graph_configs)}")
    print(f"Stage A candidates per outer fold: {len(plan.stage_a)}")
    print(f"Stage A training runs per outer fold: {len(plan.stage_a) * inner_splits}")
    print(f"Stage B planned candidates per outer fold: {len(plan.stage_b_planned)}")
    print(f"Stage B training runs per outer fold: {len(plan.stage_b_planned) * inner_splits}")
    print(f"Inner folds per search candidate: {inner_splits}")
    print(f"Confirmation runs per outer fold: {plan.selected_confirmation_runs}")
    print(f"Expected training runs per outer fold: {per_fold}")
    print(f"Expected total training runs: {total}")
    print("Selection metric: validation log loss only")
    print("Outer-test loader is not passed to search planning.")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def slug_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def graph_data_dir(config: dict[str, Any], candidate: dict[str, Any]) -> Path:
    name = (
        f"w{int(candidate['window_length'])}_"
        f"s{int(candidate['window_stride'])}_"
        f"p{slug_float(float(candidate['graph_density']))}"
    )
    return Path(config.get("data_root", "./data/folds_psiconnect_nested")) / name


def remove_graph_data_dir(config: dict[str, Any], data_dir: str | Path) -> Path:
    """Remove one resolved graph configuration beneath the configured data root."""
    data_root = Path(config.get("data_root", "./data/folds_psiconnect_nested")).resolve()
    resolved = Path(data_dir).resolve()
    if resolved == data_root or data_root not in resolved.parents:
        raise ValueError(
            f"Refusing graph cleanup outside configured data root: {resolved} "
            f"(root: {data_root})"
        )
    if resolved.exists():
        shutil.rmtree(resolved)
    return resolved


def candidate_output_dir(
    run_dir: Path,
    *,
    outer_fold: int,
    inner_fold: int,
    stage: str,
    config_id: str,
    seed: int,
) -> Path:
    return (
        run_dir
        / f"outer_fold_{outer_fold:02d}"
        / f"inner_fold_{inner_fold:02d}"
        / "candidates"
        / f"stage_{stage.upper()}"
        / config_id
        / f"seed_{seed}"
    )


def candidate_metrics_path(
    run_dir: Path,
    *,
    outer_fold: int,
    inner_fold: int,
    stage: str,
    config_id: str,
    seed: int,
) -> Path:
    return candidate_output_dir(
        run_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        stage=stage,
        config_id=config_id,
        seed=seed,
    ) / "metrics.json"


def candidate_checkpoint_dir(
    run_dir: Path,
    *,
    outer_fold: int,
    inner_fold: int,
    stage: str,
    config_id: str,
    seed: int,
) -> Path:
    return candidate_output_dir(
        run_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        stage=stage,
        config_id=config_id,
        seed=seed,
    ) / "checkpoints"


def candidate_inner_summary_path(
    run_dir: Path,
    *,
    outer_fold: int,
    stage: str,
    config_id: str,
    seed: int,
) -> Path:
    return (
        run_dir
        / f"outer_fold_{outer_fold:02d}"
        / "candidate_summaries"
        / f"stage_{stage.upper()}"
        / config_id
        / f"seed_{seed}"
        / "inner_validation_summary.json"
    )


def aggregate_candidate_inner_metrics(
    config: dict[str, Any],
    run_dir: Path,
    *,
    outer_fold: int,
    stage: str,
    config_id: str,
    seed: int,
) -> dict[str, Any] | None:
    """Aggregate a candidate only when every configured inner fold succeeded."""
    rows = []
    for inner_fold in range(1, int(config["inner_splits"]) + 1):
        path = candidate_metrics_path(
            run_dir,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            stage=stage,
            config_id=config_id,
            seed=seed,
        )
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "success":
            return None
        rows.append(row)
    losses = [float(row["val_log_loss"]) for row in rows]
    payload = {
        "status": "success",
        "run_id": rows[0]["run_id"],
        "outer_fold": outer_fold,
        "stage": stage.upper(),
        "config_id": config_id,
        "seed": seed,
        "inner_folds": list(range(1, int(config["inner_splits"]) + 1)),
        "n_inner_folds": len(rows),
        "mean_val_log_loss": float(np.mean(losses)),
        "std_val_log_loss": float(np.std(losses)),
        "outer_test_used_for_selection": False,
        "inner_metrics": [
            {"inner_fold": row["inner_fold"], "metrics_path": row["metrics_path"]}
            for row in rows
        ],
    }
    atomic_write_json(
        candidate_inner_summary_path(
            run_dir,
            outer_fold=outer_fold,
            stage=stage,
            config_id=config_id,
            seed=seed,
        ),
        payload,
    )
    return payload


def graph_stage_consumers_remaining(
    config: dict[str, Any],
    run_dir: Path,
    *,
    stage: str,
    candidate: dict[str, Any],
) -> list[str]:
    """Return planned stage consumers without successful inner-fold metrics."""
    graph_keys = ("window_length", "window_stride", "graph_density")
    target = tuple(float(candidate[key]) for key in graph_keys)
    matching = [
        row
        for row in load_candidate_table(run_dir, stage)
        if tuple(float(row[key]) for key in graph_keys) == target
    ]
    remaining = []
    for row in matching:
        seed = int(float(row["search_seed"]))
        for outer_fold in range(1, int(config["outer_splits"]) + 1):
            for inner_fold in range(1, int(config["inner_splits"]) + 1):
                path = candidate_metrics_path(
                    run_dir,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    stage=stage,
                    config_id=str(row["config_id"]),
                    seed=seed,
                )
                if not path.exists():
                    remaining.append(
                        f"outer={outer_fold},inner={inner_fold},"
                        f"config={row['config_id']},seed={seed}"
                    )
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "success":
                    remaining.append(
                        f"outer={outer_fold},inner={inner_fold},"
                        f"config={row['config_id']},seed={seed}"
                    )
    return remaining


def load_candidate_table(run_dir: Path, stage: str) -> list[dict[str, Any]]:
    stage = stage.upper()
    if stage == "A":
        path = run_dir / "summary" / "stage_a_plan.csv"
    elif stage == "B":
        selected_path = run_dir / "summary" / "stage_b_plan.csv"
        placeholder_path = run_dir / "summary" / "stage_b_plan_top_k_placeholder.csv"
        path = selected_path if selected_path.exists() else placeholder_path
    else:
        raise ValueError("stage must be A or B")
    if not path.exists():
        raise FileNotFoundError(f"Candidate plan not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_candidate(
    run_dir: Path,
    *,
    stage: str,
    config_id: str | None = None,
    candidate_index: int | None = None,
) -> dict[str, Any]:
    rows = load_candidate_table(run_dir, stage)
    if config_id is not None:
        matches = [row for row in rows if row["config_id"] == config_id]
        if not matches:
            raise ValueError(f"No {stage} candidate with config_id={config_id}")
        return matches[0]
    if candidate_index is None:
        raise ValueError("Provide config_id or candidate_index")
    if candidate_index < 0 or candidate_index >= len(rows):
        raise IndexError(
            f"candidate_index {candidate_index} out of range for {len(rows)} candidates"
        )
    return rows[candidate_index]


def best_validation_log_loss(checkpoint_path: Path) -> tuple[float, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    values = checkpoint.get("val_losses", {}).get("y_BCE", [])
    if not values:
        raise ValueError(f"Checkpoint has no validation BCE history: {checkpoint_path}")
    losses = [float(value) for value in values]
    best_index = min(range(len(losses)), key=losses.__getitem__)
    return losses[best_index], best_index + 1


def _candidate_float(candidate: dict[str, Any], key: str) -> float:
    return float(candidate[key])


def _candidate_int(candidate: dict[str, Any], key: str) -> int:
    return int(float(candidate[key]))


def train_candidate(
    *,
    config: dict[str, Any],
    run_id: str,
    outer_fold: int,
    inner_fold: int,
    stage: str,
    candidate: dict[str, Any],
    seed: int,
    prepare_graphs: bool = False,
    force: bool = False,
    dry_run: bool = False,
    cleanup_graph_after_success: bool = False,
) -> dict[str, Any]:
    """Run or describe one strict nested-sweep candidate.

    The function never constructs a test loader and writes one atomic metrics
    JSON when training completes successfully.
    """
    plan = create_plan(config, run_id=run_id)
    run_dir = plan.run_dir
    config_id = str(candidate["config_id"])
    metrics_path = candidate_metrics_path(
        run_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        stage=stage,
        config_id=config_id,
        seed=seed,
    )
    if metrics_path.exists() and not force:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["skipped_existing"] = True
        return payload

    data_dir = graph_data_dir(config, candidate)
    checkpoint_dir = candidate_checkpoint_dir(
        run_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        stage=stage,
        config_id=config_id,
        seed=seed,
    )
    version = f"{run_id}_{stage.upper()}_{config_id}_seed{seed}"
    command = {
        "data_dir": str(data_dir),
        "save_dir": str(checkpoint_dir),
        "outer": outer_fold,
        "inner": inner_fold,
        "version": version,
        "seed": seed,
    }
    if dry_run:
        return {
            "status": "dry_run",
            "run_id": run_id,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "stage": stage.upper(),
            "config_id": config_id,
            "seed": seed,
            "data_dir": str(data_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "version": version,
            "command": command,
            "graph_dir_exists": data_dir.exists(),
        }

    if not data_dir.exists():
        if not prepare_graphs:
            raise FileNotFoundError(
                f"Graph fold directory not found: {data_dir}. "
                "Precompute it or rerun with --prepare-graphs."
            )
        prepare_psiconnect_st(
            config["timeseries_root"],
            data_dir,
            expected_rois=int(config.get("expected_rois", 200)),
            matched_eeg_root=config.get("matched_eeg_root"),
            window_size=_candidate_int(candidate, "window_length"),
            shift=_candidate_int(candidate, "window_stride"),
            threshold_p=_candidate_float(candidate, "graph_density"),
            n_splits=int(config["outer_splits"]),
            inner_splits=int(config["inner_splits"]),
            seed=int(config.get("split_seed", 42)),
            device=str(config.get("preprocessing_device", "cuda")),
        )

    training = config.get("training", {})
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now().astimezone()
    args = SimpleNamespace(
        outer=outer_fold,
        inner=inner_fold,
        version=version,
        seed=seed,
        data_dir=str(data_dir),
        save_dir=str(checkpoint_dir),
        prefer_mps=False,
        no_sparse_on_cpu=bool(training.get("no_sparse_on_cpu", True)),
        batch_size=int(training.get("batch_size", 4)),
        num_epochs=int(training.get("num_epochs", 100)),
        learn_rate=float(training.get("learn_rate", 1e-5)),
        l2factor=float(training.get("l2factor", 5e-3)),
        kl_weight=1.0,
        kl_start_weight=None,
        kl_warmup_epochs=0,
        bce_weight=_candidate_float(candidate, "loss_weight"),
        mce_weight=1.0,
        patience=int(training.get("patience", 30)),
        val_freq=int(training.get("val_freq", 1)),
        grad_clip=float(training.get("grad_clip", 1.0)),
        no_early_stop=bool(training.get("no_early_stop", False)),
        no_verbose=True,
        strict_validation=True,
        no_test_during_training=True,
        omit_test_loader=True,
        selection_metric="val_bce",
        no_variational=False,
        no_recurrent=False,
        no_graphRNN=False,
        dat=False,
        model_params_override={
            "y_hidden_dim": [_candidate_int(candidate, "hidden_dim")],
            "classifier_dropout": _candidate_float(candidate, "dropout"),
        },
    )
    result = run_st_training_workflow(args)
    best_loss, best_epoch = best_validation_log_loss(Path(result.save_path))
    runtime = (datetime.now().astimezone() - start).total_seconds()
    payload = {
        "status": "success",
        "run_id": run_id,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "stage": stage.upper(),
        "config_id": config_id,
        "window_length": _candidate_int(candidate, "window_length"),
        "window_stride": _candidate_int(candidate, "window_stride"),
        "graph_density": _candidate_float(candidate, "graph_density"),
        "model_name": candidate["model_name"],
        "hidden_dim": _candidate_int(candidate, "hidden_dim"),
        "dropout": _candidate_float(candidate, "dropout"),
        "loss_weight": _candidate_float(candidate, "loss_weight"),
        "search_seed": seed,
        "best_epoch": best_epoch,
        "val_log_loss": best_loss,
        "val_auc": float("nan"),
        "val_balanced_accuracy": float("nan"),
        "training_status": "success",
        "runtime": runtime,
        "checkpoint": result.save_path,
        "metrics_path": str(metrics_path),
        "outer_test_used_for_selection": False,
        "test_loader_constructed": False,
        "graph_cleanup_requested": cleanup_graph_after_success,
    }
    atomic_write_json(metrics_path, payload)
    inner_summary = aggregate_candidate_inner_metrics(
        config,
        run_dir,
        outer_fold=outer_fold,
        stage=stage,
        config_id=config_id,
        seed=seed,
    )
    if inner_summary is not None:
        payload["inner_validation_summary"] = str(
            candidate_inner_summary_path(
                run_dir,
                outer_fold=outer_fold,
                stage=stage,
                config_id=config_id,
                seed=seed,
            )
        )
    if cleanup_graph_after_success:
        remaining = graph_stage_consumers_remaining(
            config,
            run_dir,
            stage=stage,
            candidate=candidate,
        )
        payload["graph_cleanup_pending_consumers"] = len(remaining)
        if not remaining:
            removed_path = remove_graph_data_dir(config, data_dir)
            payload["graph_cleanup_completed"] = True
            payload["removed_graph_dir"] = str(removed_path)
        else:
            payload["graph_cleanup_completed"] = False
    atomic_write_json(metrics_path, payload)
    return payload
