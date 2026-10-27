#!/usr/bin/env python
"""Run one strict PsiConnect ST nested-sweep candidate."""

from __future__ import annotations

import argparse
import json

from brainxdl.pipelines.st_nested_sweep import (
    find_candidate,
    load_sweep_config,
    train_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/psiconnect/st_nested_sweep.yaml")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    parser.add_argument("--config_id")
    parser.add_argument("--candidate_index", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prepare-graphs", action="store_true")
    parser.add_argument(
        "--cleanup-graph-after-success",
        action="store_true",
        help=(
            "Delete this graph configuration only after every planned outer/inner "
            "consumer in the stage has successful atomic metrics."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sweep_config(args.config)
    run_dir = (
        __import__("pathlib").Path(config.get("run_root", "./runs/st_nested_sweep"))
        / args.run_id
    )
    candidate = find_candidate(
        run_dir,
        stage=args.stage,
        config_id=args.config_id,
        candidate_index=args.candidate_index,
    )
    seed = args.seed if args.seed is not None else int(candidate["search_seed"])
    payload = train_candidate(
        config=config,
        run_id=args.run_id,
        outer_fold=args.outer,
        inner_fold=args.inner,
        stage=args.stage,
        candidate=candidate,
        seed=seed,
        prepare_graphs=args.prepare_graphs,
        force=args.force,
        dry_run=args.dry_run,
        cleanup_graph_after_success=args.cleanup_graph_after_success,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
