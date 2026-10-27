"""Entry point for training MM (multimodal fMRI+EEG) models.

Source: parse_args() extracted from legacy_code/2-MM/main.py. Reusable
post-parse workflow orchestration lives in brainxdl.pipelines.mm_workflow.
"""

import argparse
import shlex
import sys

from brainxdl.common.logging import METRIC_DIRECTIONS
from brainxdl.pipelines.mm_workflow import print_args, run_mm_training_workflow

main = run_mm_training_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PsiConnect models, including the multi-modal GCN and fMRI-only baselines."
    )
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
            "st_embedding_mlp",
            "st_embedding_fusion_mlp",
            "st_embedding_mmodal_gcn",
            "all",
        ],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--graph_mode",
        type=str,
        choices=["static", "dynamic"],
        default="static",
        help="Use full-series graphs (static) or sliding-window dynamic graphs.",
    )
    parser.add_argument("--window_len", type=int, default=50, help="Window length (timepoints) for dynamic graph construction.")
    parser.add_argument("--window_stride", type=int, default=25, help="Stride (timepoints) for dynamic graph construction.")
    parser.add_argument(
        "--temporal_pool",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Temporal pooling strategy after per-window graph encoders (dynamic graphs).",
    )
    parser.add_argument("--dataset", type=str, default="PSICONNECT", help="Dataset identifier (must be PSICONNECT).")
    parser.add_argument("--fmri_path", type=str, default="data/fmri_clean", help="Directory containing fMRI .mat files.")
    parser.add_argument("--eeg_path", type=str, default="data/eeg_clean", help="Directory containing EEG .mat files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Optional random seed for outer CV splits. Defaults to --seed.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability in the classifier head.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for cross entropy loss.")
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["ce", "bce"],
        default="ce",
        help="Classification loss: cross-entropy (ce) or BCE-with-logits (bce) with pos_weight.",
    )
    parser.add_argument("--pos_weight", type=float, default=None, help="Optional override for positive-class weight (BCE).")
    parser.add_argument("--class_weighted", action="store_true", help="Use inverse-frequency class weighting.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio from the training set.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--min_epochs", type=int, default=5, help="Minimum epochs before early stopping.")
    parser.add_argument("--no_early_stop", action="store_true", help="Disable early stopping.")
    parser.add_argument("--es_metric", type=str, choices=["loss", "acc"], default="loss", help="Metric for early stopping.")
    parser.add_argument("--es_min_delta", type=float, default=0.0, help="Minimum change to count as improvement.")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping max norm (0 to disable).")
    parser.add_argument("--balance_batches", action="store_true", help="Balance class frequency via sampling.")
    parser.add_argument(
        "--balance_threshold",
        type=float,
        default=0.8,
        help="Auto-enable WeightedRandomSampler when minority/majority ratio < threshold. Set <=0 to disable.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=1,
        help=(
            "Number of folds. Repeated-participant PsiConnect data require grouped "
            "cross-validation with at least two folds; thesis runs use five."
        ),
    )
    parser.add_argument("--embid_dim", type=int, default=128, help="Embedding dimension for ROI encoders.")
    parser.add_argument("--gcn_hidden_dim", type=int, default=256, help="Hidden size for the fMRI GCN classifier head.")
    parser.add_argument("--out_channels", type=int, default=2, help="Number of output classes.")
    parser.add_argument("--k_fmri", type=int, default=40, help="k for kNN thresholding on fMRI graphs.")
    parser.add_argument("--k_eeg", type=int, default=40, help="k for kNN thresholding on EEG graphs.")
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g., 'cuda:0'). Default auto-select.")
    parser.add_argument(
        "--modality",
        type=str,
        choices=["fusion", "fmri", "eeg"],
        default="fusion",
        help="Select which modality to use during training/evaluation.",
    )
    parser.add_argument("--task", type=str, default="REST", help="Single BIDS task filter (comma-separated allowed).")
    parser.add_argument("--tasks", nargs="+", default=None, help="Multiple BIDS task filters (overrides --task).")
    parser.add_argument("--use_cache", action="store_true", help="Load preprocessed data from cache if available.")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory for cached data files.")
    parser.add_argument("--cache_path", type=str, default=None, help="Explicit path to the cache file.")
    parser.add_argument(
        "--st_embedding_dir",
        type=str,
        default=None,
        help="Directory containing st_embedding_features_fold*.npz for ST-derived fMRI embeddings.",
    )
    parser.add_argument(
        "--st_embedding_feature_set",
        type=str,
        choices=["readout", "logits"],
        default="readout",
        help="ST feature array to use from each st_embedding_features_fold*.npz file.",
    )
    parser.add_argument(
        "--st_fold_manifest",
        type=str,
        default=None,
        help="PsiConnect ST manifest with fold split_indices. Required for leakage-safe ST embedding alignment.",
    )
    parser.add_argument(
        "--st_embedding_seed",
        type=int,
        default=None,
        help="Require the ST embedding manifest and exported rows to use this source seed.",
    )
    parser.add_argument("--plot_debug", action="store_true", help="Plot example correlation matrices.")
    parser.add_argument("--plot_curves", action="store_true", help="Save training/validation curves for each fold.")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to store training curve figures.")
    parser.add_argument("--plot_show", action="store_true", help="Display training curves interactively.")
    parser.add_argument("--no_plot_show", action="store_true", help="Force-disable interactive display of training curves.")
    parser.add_argument("--metrics_dir", type=str, default="result", help="Directory to store aggregated evaluation summaries.")
    parser.add_argument("--save_predictions", action="store_true", help="Save per-fold validation and test probabilities for diagnostics.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to store model checkpoints.")
    parser.add_argument("--save_best_model", action="store_true", help="Persist the best-performing model checkpoint.")
    parser.add_argument(
        "--save_metric",
        type=str,
        default="acc",
        choices=sorted(METRIC_DIRECTIONS.keys()),
        help="Evaluation metric key to monitor when saving checkpoints.",
    )
    parser.add_argument("--plot_latents", action="store_true", help="Generate t-SNE and UMAP projections of latent embeddings.")
    parser.add_argument("--latent_dir", type=str, default="plots/latents", help="Directory to store latent manifold figures.")
    parser.add_argument("--latent_methods", nargs="+", default=("tsne", "umap"), help="Latent manifold methods to run.")
    parser.add_argument(
        "--latent_split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all", "train_test"],
        help="Dataset split to visualise when plotting latent embeddings.",
    )
    parser.add_argument("--latent_batch_size", type=int, default=None, help="Batch size override for latent extraction.")
    parser.add_argument("--latent_components", type=int, default=2, help="Number of output dimensions for manifold projections.")
    parser.add_argument("--latent_seed", type=int, default=None, help="Random seed for t-SNE/UMAP.")
    parser.add_argument("--latent_tsne_perplexity", type=float, default=30.0, help="Perplexity for t-SNE.")
    parser.add_argument("--latent_tsne_lr", type=float, default=200.0, help="Learning rate used by t-SNE.")
    parser.add_argument("--latent_tsne_iter", type=int, default=1000, help="Number of optimisation iterations for t-SNE.")
    parser.add_argument("--latent_tsne_metric", type=str, default="euclidean", help="Distance metric for t-SNE.")
    parser.add_argument("--latent_tsne_verbose", type=int, default=0, help="Verbosity level for t-SNE.")
    parser.add_argument("--latent_tsne_init", type=str, default="random", choices=["random", "pca"], help="Init strategy for t-SNE.")
    parser.add_argument("--latent_umap_neighbors", type=int, default=15, help="Number of neighbours for UMAP.")
    parser.add_argument("--latent_umap_min_dist", type=float, default=0.1, help="Minimum distance hyper-parameter for UMAP.")
    parser.add_argument("--latent_umap_metric", type=str, default="euclidean", help="Distance metric for UMAP.")
    parser.add_argument(
        "--gnn_explain",
        action="store_true",
        help=(
            "Run the project-specific optimized dense edge-mask explainer after "
            "evaluation (not the standard PyG GNNExplainer implementation)."
        ),
    )
    parser.add_argument(
        "--explain_split",
        type=str,
        nargs="?",
        const="test",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used when generating explanations.",
    )
    parser.add_argument("--explain_indices", nargs="+", type=int, default=None, help="Explicit indices to explain.")
    parser.add_argument("--explain_per_class", type=int, default=1, help="Samples per class to explain.")
    parser.add_argument("--explain_max_samples", type=int, default=2, help="Maximum samples to explain per run.")
    parser.add_argument("--explain_topk_edges", type=int, default=20, help="Number of highest-saliency edges to include.")
    parser.add_argument("--explain_dir", type=str, default="plots/explanations", help="Directory for edge-mask explanation outputs.")
    parser.add_argument(
        "--roi_coords_path",
        type=str,
        default="data/schaefer_200_mni_centroids.tsv",
        help="Path to ROI centroid coordinates in MNI space.",
    )
    parser.add_argument("--eeg_coords_path", type=str, default=None, help="Optional EEG channel coordinates in MNI space.")
    parser.add_argument("--glass_edge_percentile", type=float, default=85.0, help="Percentile threshold for glass-brain edges.")
    parser.add_argument("--explain_glass_brain", action="store_true", help="Force-enable glass-brain connectome plots.")
    parser.add_argument("--no_explain_glass_brain", action="store_true", help="Disable glass-brain connectome plots.")
    parser.add_argument("--explain_steps", type=int, default=200, help="Optimisation iterations for the edge-mask explainer.")
    parser.add_argument("--explain_lr", type=float, default=0.1, help="Learning rate for the explainer mask optimiser.")
    parser.add_argument("--explain_size_coeff", type=float, default=0.01, help="Regularisation coefficient for sparse subgraphs.")
    parser.add_argument("--explain_entropy_coeff", type=float, default=5e-4, help="Entropy regulariser weight for explainer masks.")

    parsed_args = parser.parse_args()
    parsed_args.require_grouped_cv = parsed_args.dataset.upper() == "PSICONNECT"
    parsed_args.command = " ".join(shlex.quote(arg) for arg in sys.argv)
    parsed_args.explain_glass_brain = getattr(parsed_args, "explain_glass_brain", False) or not getattr(
        parsed_args, "no_explain_glass_brain", False
    )
    return parsed_args



if __name__ == "__main__":
    ARGS = parse_args()
    ARGS.dataset = ARGS.dataset.upper()
    print_args(ARGS)
    main(ARGS)
