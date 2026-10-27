#!/usr/bin/env python3
"""Run seed-2026 held-out attention extraction for all five frozen ST-D1 folds."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from brainxdl.st.artifacts import FOLD_DENSITY, resolve_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--expected-samples", type=int, default=120)
    parser.add_argument("--prefer-mps", action="store_true")
    parser.add_argument(
        "--sparse-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def run_or_print(command: list[str], plan_only: bool) -> None:
    print(shlex.join(command), flush=True)
    if not plan_only:
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    if args.seed != 2026:
        raise ValueError("The primary ST attention study is frozen to seed 2026.")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")

    frozen = read_json(args.d1_root / "frozen_selection.json")
    if frozen.get("outer_test_used") is not False:
        raise ValueError("ST-D1 frozen selection must explicitly exclude outer-test data.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_outputs: list[Path] = []
    for fold in range(1, 6):
        metrics_path = (
            args.d1_root / "final" / f"seed_{args.seed}" / f"outer_{fold:02d}" / "metrics.json"
        )
        metrics = read_json(metrics_path)
        if int(metrics.get("seed", -1)) != args.seed or int(metrics.get("outer_fold", -1)) != fold:
            raise ValueError(f"Fold {fold}: final metrics seed/fold metadata mismatch.")
        checkpoint = resolve_checkpoint(metrics_path, metrics)
        density = FOLD_DENSITY[fold]
        fold_file = args.data_root / f"w40_s10_{density}" / f"graphs_outer{fold}_inner1.pkl"
        if not fold_file.is_file():
            raise FileNotFoundError(fold_file)

        fold_output = args.output_dir / f"fold_{fold:02d}"
        fold_npz = fold_output / f"st_attention_fold{fold}_test.npz"
        fold_json = fold_output / f"st_attention_fold{fold}_test.json"
        fold_outputs.append(fold_npz)
        complete = False
        if fold_npz.is_file() and fold_json.is_file() and not args.overwrite:
            complete = read_json(fold_json).get("status") == "complete"
        if complete:
            print(f"Fold {fold}: complete output already exists; skipping.", flush=True)
            continue

        command = [
            sys.executable,
            "scripts/st/extract_st_attention.py",
            "--checkpoint",
            str(checkpoint),
            "--fold-file",
            str(fold_file),
            "--outer-fold",
            str(fold),
            "--seed",
            str(args.seed),
            "--split",
            "test",
            "--output-dir",
            str(fold_output),
            "--top-k",
            str(args.top_k),
        ]
        if args.prefer_mps:
            command.append("--prefer-mps")
        if args.sparse_on_cpu is not None:
            command.append("--sparse-on-cpu" if args.sparse_on_cpu else "--no-sparse-on-cpu")
        if args.overwrite:
            command.append("--overwrite")
        run_or_print(command, args.plan_only)

    summary_command = [
        sys.executable,
        "scripts/st/summarize_st_attention.py",
        "--fold-files",
        *map(str, fold_outputs),
        "--output-dir",
        str(args.output_dir / "summary"),
        "--seed",
        str(args.seed),
        "--top-k",
        str(args.top_k),
        "--expected-samples",
        str(args.expected_samples),
    ]
    if args.overwrite:
        summary_command.append("--overwrite")
    summary_json = args.output_dir / "summary" / "st_attention_summary.json"
    if summary_json.is_file() and not args.overwrite:
        if read_json(summary_json).get("status") == "complete":
            print("Five-fold summary already exists and is complete; skipping.", flush=True)
            return 0
    run_or_print(summary_command, args.plan_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
