#!/usr/bin/env python3
"""Run controlled participant-grouped refinements of static MM-BrainXDL."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from brainxdl.mm.participant_grouped_benchmark import atomic_write_json, load_benchmark_cache
from brainxdl.mm.refined_benchmark import VARIANTS, aggregate_refined_variant, run_refined_fold


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "outputs/psiconnect/mm/participant_grouped_static_mm"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/psiconnect/mm/refined_mm_brainxdl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold", type=int, choices=range(1, 6), action="append")
    parser.add_argument("--manifest", type=Path, default=SOURCE_ROOT / "manifests/participant_grouped_folds.json")
    parser.add_argument("--cache-path", type=Path, default=SOURCE_ROOT / "manifests/mm_static_cache.pkl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--connectivity-skip-dim", type=int, choices=(16, 32, 64))
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data = load_benchmark_cache(args.cache_path.resolve(), args.manifest.resolve())
    output_root = args.output_dir.resolve() / ("smoke" if args.smoke_test else "")
    folds = sorted(set(args.fold or range(1, 6)))
    for fold in folds:
        destination = output_root / args.variant / f"seed{args.seed}" / f"fold{fold:02d}"
        if args.resume and (destination / "complete.json").is_file():
            print(f"[resume] complete: {destination}")
            continue
        if destination.exists() and any(destination.iterdir()) and not args.resume:
            raise RuntimeError(f"Refusing to overwrite non-empty fold directory: {destination}")
        result = run_refined_fold(
            variant_name=args.variant, seed=args.seed, fold=fold, data=data,
            manifest_path=args.manifest.resolve(), output_root=output_root, device=device,
            epochs=2 if args.smoke_test else args.epochs,
            patience=2 if args.smoke_test else args.patience,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, diagnostics=args.diagnostics,
            connectivity_skip_dim=args.connectivity_skip_dim,
        )
        atomic_write_json(
            result / "complete.json",
            {"complete": True, "variant": args.variant, "seed": args.seed, "fold": fold},
        )
        print(f"completed: {result}")
    complete = all(
        (output_root / args.variant / f"seed{args.seed}" / f"fold{fold:02d}" / "complete.json").is_file()
        for fold in range(1, 6)
    )
    if complete:
        print(f"aggregated: {aggregate_refined_variant(output_root, args.variant, args.seed)}")


if __name__ == "__main__":
    main()
