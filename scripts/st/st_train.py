#!/usr/bin/env python
"""ST-BrainXDL training entry point.

Equivalent to legacy_code/1-ST/main.py but uses clean package imports
and exposes all configuration via argparse so no file editing is needed.

Usage (run from repo root with brainxdl installed, e.g. pip install -e .):

    python scripts/st/st_train.py

    # Different fold:
    python scripts/st/st_train.py --outer 2 --inner 3

    # Override checkpoint paths:
    python scripts/st/st_train.py --save_dir ./my_runs --version v1

All model hyperparameters and training settings are identical to the
published IJCAI 2024 defaults unless overridden.
"""

import argparse

from brainxdl.pipelines.st_workflow import run_st_training_workflow

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ST-BrainXDL on the ABIDE fMRI dataset.")

    # Fold selection
    p.add_argument("--outer",   type=int, default=1, help="Outer CV fold (1-5).")
    p.add_argument("--inner",   type=int, default=1, help="Inner CV fold (1-5).")
    p.add_argument("--version", type=str, default="v3", help="Checkpoint version tag.")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional explicit training seed. Omit to preserve legacy RNG behavior.",
    )

    # Paths
    p.add_argument(
        "--data_dir",  default="./data/st_folds",
        help="Directory containing graphs_outerX_innerY.pkl files.",
    )
    p.add_argument(
        "--save_dir",  default="./runs/st",
        help="Directory to write model checkpoints.",
    )

    # Device
    p.add_argument(
        "--prefer_mps", action="store_true", default=True,
        help="Prefer MPS (Apple Silicon) over CPU.",
    )
    p.add_argument(
        "--no_sparse_on_cpu",
        action="store_true",
        help=(
            "Keep ST graph/PyG modules on the selected device instead of forcing "
            "them to CPU. Intended for CUDA HPC runs; default preserves legacy "
            "CPU sparse placement."
        ),
    )

    # Training hyperparameters (defaults match IJCAI 2024 published run)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--num_epochs",   type=int,   default=100)
    p.add_argument("--learn_rate",   type=float, default=1e-5,
                   help="Learning rate (applied to both optimisers).")
    p.add_argument("--l2factor",     type=float, default=5e-3,
                   help="L2 weight decay (applied to both optimisers).")
    p.add_argument(
        "--kl_weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier for the variational z_KLD term. Default 1.0 preserves "
            "legacy ST-BrainXDL behavior."
        ),
    )
    p.add_argument(
        "--kl_start_weight",
        type=float,
        default=None,
        help=(
            "Initial z_KLD multiplier for linear KL warm-up. If omitted, uses "
            "--kl_weight and preserves fixed-KL behavior."
        ),
    )
    p.add_argument(
        "--kl_warmup_epochs",
        type=int,
        default=0,
        help=(
            "Number of epochs for linear KL warm-up from --kl_start_weight to "
            "--kl_weight. Default 0 preserves fixed-KL behavior."
        ),
    )
    p.add_argument(
        "--bce_weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier for the classifier BCE/CE term. Default 1.0 preserves "
            "legacy ST-BrainXDL behavior."
        ),
    )
    p.add_argument(
        "--mce_weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier for the adjacency matrix cross-entropy a_NLL term. "
            "Default 1.0 preserves legacy ST-BrainXDL behavior."
        ),
    )
    p.add_argument(
        "--encoder_cls_weight",
        type=float,
        default=0.0,
        help=(
            "Classification-loss multiplier for supervised encoder fine-tuning. "
            "Default 0.0 preserves legacy detached-encoder behavior."
        ),
    )
    p.add_argument("--patience",     type=int,   default=30,
                   help="Early-stopping patience (epochs).")
    p.add_argument("--val_freq",     type=int,   default=1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--no_verbose",    action="store_true")
    p.add_argument(
        "--strict_validation",
        action="store_true",
        help=(
            "Keep validation graphs out of optimizer training. Required for "
            "hyperparameter tuning; the default preserves legacy merging."
        ),
    )
    p.add_argument(
        "--no_test_during_training",
        action="store_true",
        help=(
            "Do not evaluate the outer test split during training. Required "
            "while selecting hyperparameters from validation data."
        ),
    )
    p.add_argument(
        "--omit_test_loader",
        action="store_true",
        help=(
            "Do not construct or pass a test DataLoader during training. Intended "
            "for nested-CV hyperparameter search jobs."
        ),
    )
    p.add_argument(
        "--selection_metric",
        choices=["legacy_train_bce_acc", "val_bce", "val_total", "val_acc"],
        default="legacy_train_bce_acc",
        help=(
            "Metric used for checkpoint selection and early stopping. The default "
            "preserves legacy behavior; val_bce selects the lowest validation "
            "classification BCE."
        ),
    )

    # Model flags
    p.add_argument("--no_variational",  action="store_true")
    p.add_argument("--no_recurrent",    action="store_true")
    p.add_argument("--no_graphRNN",     action="store_true")
    p.add_argument("--dat",             action="store_true",
                   help="Enable Domain Adversarial Training.")
    p.add_argument(
        "--readout_mode",
        choices=["legacy", "compact_temporal"],
        default="legacy",
        help="Classifier readout. Default preserves the published flattened final-window readout.",
    )
    p.add_argument("--readout_projection_dim", type=int, default=32)
    p.add_argument(
        "--stable_fc_skip",
        action="store_true",
        help="Fuse a projected temporal-mean FC vector into the classifier readout.",
    )
    p.add_argument("--stable_fc_dim", type=int, default=32)
    p.add_argument("--classifier_hidden_dim", type=int, default=None)
    p.add_argument(
        "--window_subsample",
        type=int,
        default=1,
        help="Uniform screening-only window thinning factor; 1 preserves all windows.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_st_training_workflow(args)


if __name__ == "__main__":
    main()
