"""Entry point for evaluating a saved MM checkpoint on held-out data.

Loads a checkpoint written by train_mm.py, runs evaluate_metrics on the
full dataset, and prints results. Pass --config to replay the original CLI
flags, or supply individual --* overrides.
"""

import argparse
import shlex
import sys

from brainxdl.pipelines.mm_evaluation import evaluate_mm_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MM checkpoint produced by train_mm.py."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--fmri_path", type=str, default="data/fmri_clean", help="Directory containing fMRI .mat files.")
    parser.add_argument("--eeg_path", type=str, default="data/eeg_clean", help="Directory containing EEG .mat files.")
    parser.add_argument(
        "--model",
        type=str,
        default="mmodal_gcn",
        choices=[
            "mmodal_gcn",
            "mmodal_gcn_nogate",
            "fusion_mlp",
            "mm_brainxdl_dynamic",
            "brainmgf_dynamic",
            "gcn",
        ],
    )
    parser.add_argument("--modality", type=str, choices=["fusion", "fmri", "eeg"], default="fusion")
    parser.add_argument("--loss_type", type=str, choices=["ce", "bce"], default="ce")
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--embid_dim", type=int, default=128)
    parser.add_argument("--k_fmri", type=int, default=40)
    parser.add_argument("--k_eeg", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="REST")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--graph_mode", type=str, choices=["static", "dynamic"], default="static")
    parser.add_argument("--window_len", type=int, default=50)
    parser.add_argument("--window_stride", type=int, default=25)
    parser.add_argument("--temporal_pool", type=str, choices=["mean", "max"], default="mean")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--gcn_hidden_dim", type=int, default=256)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    args = parser.parse_args()
    args.command = " ".join(shlex.quote(a) for a in sys.argv)
    evaluate_mm_checkpoint(args)


if __name__ == "__main__":
    main()
