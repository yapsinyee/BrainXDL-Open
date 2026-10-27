#!/usr/bin/env python
"""Select Stage B candidates from completed PsiConnect ST Stage A summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from brainxdl.pipelines.st_nested_sweep import (
    load_sweep_config,
    select_stage_b_candidates_from_stage_a,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/psiconnect/st_nested_sweep.yaml")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Write summary/stage_b_plan.csv for the selected candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sweep_config(args.config)
    run_dir = Path(config.get("run_root", "./runs/st_nested_sweep")) / args.run_id
    selected_stage_a, stage_b = select_stage_b_candidates_from_stage_a(
        config,
        run_dir,
        top_k=args.top_k,
    )
    print("Top Stage A candidates:")
    for row in selected_stage_a:
        print(
            f"{row['config_id']} "
            f"mean_val_log_loss={float(row['mean_val_log_loss']):.6f} "
            f"window={int(float(row['window_length']))} "
            f"stride={int(float(row['window_stride']))} "
            f"density={float(row['graph_density']):g}"
        )

    out = run_dir / "summary" / "stage_b_plan.csv"
    if args.write_plan:
        write_csv(out, stage_b)
    print(f"{'Wrote' if args.write_plan else 'Planned'} {len(stage_b)} Stage B rows to {out}")


if __name__ == "__main__":
    main()
