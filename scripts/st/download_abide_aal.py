#!/usr/bin/env python3
"""Download ABIDE PCP AAL ROI time-series derivatives with Nilearn.

This fetches the ``rois_aal`` derivative from the preprocessed ABIDE PCP
dataset and writes a small manifest next to the downloaded data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


def _parse_subject_ids(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_path_manifest(paths: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for path in paths:
            f.write(f"{path}\n")


def _find_downloaded_rois_aal(data_dir: str) -> list[str]:
    data_root = Path(data_dir).expanduser()
    return [str(path) for path in sorted(data_root.rglob("*_rois_aal.1D"))]


def _write_phenotypic_csv(phenotypic: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(phenotypic, "to_csv"):
        phenotypic.to_csv(output_path, index=False)
        return

    names = getattr(getattr(phenotypic, "dtype", None), "names", None)
    if not names:
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for row in phenotypic:
            writer.writerow([row[name] for name in names])


def download_abide_aal(args: argparse.Namespace) -> int:
    try:
        from nilearn.datasets import fetch_abide_pcp
    except ImportError:
        print(
            "Nilearn is required to download ABIDE PCP data. "
            "Install it with: python3 -m pip install nilearn",
            file=sys.stderr,
        )
        return 1

    filters: dict[str, Any] = {}
    subject_ids = _parse_subject_ids(args.subject_ids)
    if subject_ids is not None:
        filters["SUB_ID"] = subject_ids
    if args.dx_group is not None:
        filters["DX_GROUP"] = args.dx_group
    if args.sex is not None:
        filters["SEX"] = args.sex

    dataset = fetch_abide_pcp(
        data_dir=args.data_dir,
        n_subjects=args.n_subjects,
        pipeline=args.pipeline,
        band_pass_filtering=args.band_pass_filtering,
        global_signal_regression=args.global_signal_regression,
        derivatives=["rois_aal"],
        quality_checked=not args.include_failed_qc,
        verbose=args.verbose,
        **filters,
    )

    rois_aal = [
        str(path) for path in dataset.rois_aal
        if isinstance(path, (str, Path)) and Path(path).exists()
    ]
    if not rois_aal:
        rois_aal = _find_downloaded_rois_aal(args.data_dir)
    output_dir = Path(args.manifest_dir)
    paths_file = output_dir / "abide_rois_aal_paths.txt"
    pheno_file = output_dir / "abide_rois_aal_phenotypic.csv"

    _write_path_manifest(rois_aal, paths_file)
    _write_phenotypic_csv(dataset.phenotypic, pheno_file)

    print(f"Downloaded/found {len(rois_aal)} AAL ROI time-series files.")
    print(f"Data directory: {Path(args.data_dir).expanduser()}")
    print(f"File manifest: {paths_file}")
    if pheno_file.exists():
        print(f"Phenotypic CSV: {pheno_file}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ABIDE PCP AAL ROI time-series derivatives via Nilearn."
    )
    parser.add_argument(
        "--data_dir",
        default="./data/nilearn",
        help="Directory where Nilearn will store downloaded ABIDE data.",
    )
    parser.add_argument(
        "--manifest_dir",
        default="./data/manifests",
        help="Directory for generated path and phenotypic manifests.",
    )
    parser.add_argument(
        "--n_subjects",
        type=int,
        default=None,
        help="Number of subjects to download. Omit for all matching subjects.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("ccs", "cpac", "dparsf", "niak"),
        default="cpac",
        help="ABIDE PCP preprocessing pipeline.",
    )
    parser.add_argument(
        "--band_pass_filtering",
        action="store_true",
        help="Fetch derivatives from the band-pass filtered preprocessing variant.",
    )
    parser.add_argument(
        "--global_signal_regression",
        action="store_true",
        help="Fetch derivatives from the global signal regression preprocessing variant.",
    )
    parser.add_argument(
        "--include_failed_qc",
        action="store_true",
        help="Include subjects that did not pass all quality-control raters.",
    )
    parser.add_argument(
        "--subject_ids",
        default=None,
        help="Comma-separated ABIDE SUB_ID values to fetch, for example 50003,50004.",
    )
    parser.add_argument(
        "--dx_group",
        type=int,
        choices=(1, 2),
        default=None,
        help="Optional diagnosis filter: 1=autism, 2=control.",
    )
    parser.add_argument(
        "--sex",
        type=int,
        choices=(1, 2),
        default=None,
        help="Optional sex filter: 1=male, 2=female.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Nilearn verbosity level.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return download_abide_aal(args)


if __name__ == "__main__":
    raise SystemExit(main())
