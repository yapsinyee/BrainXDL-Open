#!/usr/bin/env python
"""Build participant-grouped ST-BrainXDL folds from PsiConnect fMRI signals."""

import argparse

from brainxdl.st.psiconnect import DEFAULT_PATTERN, prepare_psiconnect_st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries_root", required=True)
    parser.add_argument("--output_dir", default="./data/folds_psiconnect_schaefer200_rest")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--expected_rois", type=int, default=200)
    parser.add_argument(
        "--matched_eeg_root",
        required=True,
        help=(
            "Restrict ST samples to rest-task subject/session pairs available to "
            "the MM EEG input, keeping the whole-framework cohort aligned."
        ),
    )
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--shift", type=int, default=10)
    parser.add_argument("--threshold_p", type=float, default=0.40)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Preprocessing device. Defaults to cuda and fails if no GPU is allocated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_psiconnect_st(
        args.timeseries_root,
        args.output_dir,
        pattern=args.pattern,
        expected_rois=args.expected_rois,
        matched_eeg_root=args.matched_eeg_root,
        window_size=args.window_size,
        shift=args.shift,
        threshold_p=args.threshold_p,
        n_splits=args.n_splits,
        inner_splits=args.inner_splits,
        seed=args.seed,
        device=args.device,
    )
    print(
        f"Prepared {manifest['n_samples']} samples from "
        f"{manifest['n_participants']} participants in {args.output_dir}."
    )


if __name__ == "__main__":
    main()
