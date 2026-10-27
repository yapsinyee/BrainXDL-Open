#!/usr/bin/env python
"""Apply ComBat harmonization to ABIDE ST-BrainXDL LDW connectivity output."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import dill

from brainxdl.st.harmonization import (
    align_abide_phenotype_to_roi_order,
    harmonize_ldw_connectivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonize ABIDE LDW dynamic connectivity matrices with neuroCombat. "
            "Run after brainxdl.st.preprocessing and before brainxdl.st.data_prep."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing LDW_abide_data.pkl and win_info.pkl.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where harmonized LDW_abide_data.pkl will be written.",
    )
    parser.add_argument(
        "--rois_dir",
        required=True,
        help="ABIDE AAL ROI directory used for LDW preprocessing.",
    )
    parser.add_argument(
        "--phenotypic_csv",
        required=True,
        help="ABIDE phenotype CSV used for LDW preprocessing.",
    )
    parser.add_argument("--roi_pattern", default="*_rois_aal.1D")
    parser.add_argument("--batch_col", default="SITE_ID")
    parser.add_argument(
        "--categorical_cols",
        default="auto",
        help="Comma-separated categorical covariates to preserve, or auto for DX_GROUP,SEX when available.",
    )
    parser.add_argument(
        "--continuous_cols",
        default="auto",
        help="Comma-separated continuous covariates to preserve, or auto for AGE_AT_SCAN when available.",
    )
    parser.add_argument("--threshold_p", type=float, default=0.40)
    parser.add_argument("--no_eb", action="store_true")
    parser.add_argument("--nonparametric", action="store_true")
    parser.add_argument("--mean_only", action="store_true")
    parser.add_argument("--ref_batch", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ldw_path = input_dir / "LDW_abide_data.pkl"
    win_info_path = input_dir / "win_info.pkl"

    if not ldw_path.exists():
        raise FileNotFoundError(f"Missing LDW input: {ldw_path}")
    if not win_info_path.exists():
        raise FileNotFoundError(f"Missing window info input: {win_info_path}")

    with ldw_path.open("rb") as handle:
        ldw_data = dill.load(handle)
    with win_info_path.open("rb") as handle:
        win_info = pickle.load(handle)

    covars = align_abide_phenotype_to_roi_order(
        args.rois_dir,
        args.phenotypic_csv,
        pattern=args.roi_pattern,
    )
    if "DX_GROUP" in covars.columns:
        expected_labels = covars["DX_GROUP"].map(lambda value: 1 if int(value) == 1 else 0)
        if not (expected_labels.to_numpy() == ldw_data["labels"]).all():
            raise ValueError(
                "Phenotype DX_GROUP labels do not match LDW labels after ROI-order alignment."
            )

    node_feats, adj_mats, reports = harmonize_ldw_connectivity(
        ldw_data["node_feat"],
        covars,
        batch_col=args.batch_col,
        categorical_cols=args.categorical_cols,
        continuous_cols=args.continuous_cols,
        threshold_p=args.threshold_p,
        eb=not args.no_eb,
        parametric=not args.nonparametric,
        mean_only=args.mean_only,
        ref_batch=args.ref_batch,
        verbose=args.verbose,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ldw = {
        "adj_mat": adj_mats,
        "node_feat": node_feats,
        "labels": ldw_data["labels"],
    }
    with (output_dir / "LDW_abide_data.pkl").open("wb") as handle:
        pickle.dump(out_ldw, handle, protocol=4)
    with (output_dir / "win_info.pkl").open("wb") as handle:
        pickle.dump(win_info, handle, protocol=4)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rois_dir": args.rois_dir,
        "phenotypic_csv": args.phenotypic_csv,
        "batch_col": args.batch_col,
        "categorical_cols": args.categorical_cols,
        "continuous_cols": args.continuous_cols,
        "threshold_p": args.threshold_p,
        "eb": not args.no_eb,
        "parametric": not args.nonparametric,
        "mean_only": args.mean_only,
        "ref_batch": args.ref_batch,
        "windows": [report.__dict__ for report in reports],
    }
    with (output_dir / "combat_harmonization_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Harmonized LDW data saved to {output_dir}")
    print(f"Windows harmonized: {len(reports)}")


if __name__ == "__main__":
    main()
