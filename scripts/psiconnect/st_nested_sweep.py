#!/usr/bin/env python
"""Plan a strict nested-CV PsiConnect ST improvement sweep."""

from __future__ import annotations

import argparse

from brainxdl.pipelines.st_nested_sweep import (
    create_plan,
    load_sweep_config,
    print_plan,
    write_plan_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/psiconnect/st_nested_sweep.yaml")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Write config, reproducibility metadata, and planned candidate CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sweep_config(args.config)
    plan = create_plan(config, run_id=args.run_id)
    print_plan(plan)
    if args.dry_run or args.write_plan:
        write_plan_artifacts(plan, dry_run=args.dry_run)
        print(f"Wrote plan artifacts to {plan.run_dir}")


if __name__ == "__main__":
    main()
