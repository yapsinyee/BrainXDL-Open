"""Smoke tests: verify all MM submodules can be imported without error."""


def test_mm_package_import():
    import brainxdl.mm  # noqa: F401


def test_mm_models_import():
    from brainxdl.mm.models import (  # noqa: F401
        DynamicMMBrainXDL,
        FmriGCN,
        GraphConvModel,
        LateFusionMLP,
        MMBrainXDL,
        MoEFusion,
        build_model,
        forward_model,
    )


def test_mm_data_import():
    from brainxdl.mm.data import (  # noqa: F401
        build_weighted_sampler,
        compute_min_max_ratio,
        infer_pos_weight,
        prepare_data,
        resolve_cache_path,
        summarize_split_counts,
    )


def test_mm_training_import():
    from brainxdl.mm.training import (  # noqa: F401
        analyze_fusion_gate_bias,
        compute_probs_and_preds,
        evaluate,
        evaluate_metrics,
        plot_training_history,
        run_kfold_cv,
        train_one_epoch,
    )


def test_mm_eval_import():
    from brainxdl.mm.eval import (  # noqa: F401
        DenseGNNExplainer,
        maybe_run_gnn_explanations,
    )


def test_mm_utils_import():
    from brainxdl.mm.utils import (  # noqa: F401
        collect_latent_embeddings,
        compute_run_slug,
        compute_tsne,
        compute_umap,
        ensure_shared_manifold_hparams,
        ensure_unique_path,
        generate_latent_manifolds,
        get_task_list,
        plot_scatter,
        resolve_checkpoint_dir,
        slugify,
    )


def test_xai_import():
    from brainxdl.xai import (  # noqa: F401
        DenseGNNExplainer,
        collect_latent_embeddings,
        compute_tsne,
        compute_umap,
        ensure_shared_manifold_hparams,
        generate_latent_manifolds,
        maybe_run_gnn_explanations,
        plot_scatter,
    )
