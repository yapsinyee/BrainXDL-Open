#!/usr/bin/env python3
"""Prepare or run the participant-grouped static PsiConnect MM benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brainxdl.mm.participant_grouped_benchmark import (
    AVAILABLE_MODELS,
    UNAVAILABLE_MODELS,
    benchmark_shape_summary,
    load_benchmark_cache,
    prepare_benchmark_cache,
    read_manifest,
    run_fold,
    validate_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data/folds_psiconnect_nested/w40_s10_p0p1/psiconnect_st_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/psiconnect/mm/participant_grouped_static_mm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--fmri-path", type=Path)
    parser.add_argument("--eeg-path", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", choices=AVAILABLE_MODELS + tuple(UNAVAILABLE_MODELS))
    parser.add_argument(
        "--modality", choices=("fmri", "eeg"),
        help="For gcn; defaults to the established fMRI route.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold", type=int, choices=range(1, 6), action="append")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.prepare_only:
        if args.fmri_path is None or args.eeg_path is None:
            parser.error("--prepare-only requires --fmri-path and --eeg-path")
    elif args.model is None:
        parser.error("--model is required unless --prepare-only is used")
    if args.modality is not None and args.model != "gcn":
        parser.error("--modality is supported only for gcn")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    cache_path = (args.cache_path or args.output_dir / "manifests/mm_static_cache.pkl").resolve()
    if args.prepare_only:
        audit = prepare_benchmark_cache(
            fmri_path=args.fmri_path.resolve(),
            eeg_path=args.eeg_path.resolve(),
            manifest_path=args.manifest.resolve(),
            cache_path=cache_path,
            output_root=args.output_dir,
        )
        print(json.dumps(audit, indent=2))
        return
    if args.model in UNAVAILABLE_MODELS:
        raise SystemExit(f"{args.model}: NOT AVAILABLE — {UNAVAILABLE_MODELS[args.model]}")
    manifest = read_manifest(args.manifest.resolve())
    validate_manifest(manifest)
    data = load_benchmark_cache(cache_path, args.manifest.resolve())
    model_name = f"{args.model}_eeg" if args.modality == "eeg" else args.model
    print(json.dumps({
        "model": args.model,
        "modality": args.modality or "fmri/default",
        "n_observations": len(data),
        "tensor_shapes": benchmark_shape_summary(data),
    }, indent=2))
    for fold in sorted(set(args.fold or range(1, 6))):
        destination = run_fold(
            model_name=model_name,
            seed=args.seed,
            fold=fold,
            data=data,
            manifest=manifest,
            output_root=args.output_dir,
            device_name=args.device,
            resume=args.resume,
            overwrite=args.overwrite,
            smoke_test=args.smoke_test,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        print(f"completed: {destination}")


if __name__ == "__main__":
    main()
