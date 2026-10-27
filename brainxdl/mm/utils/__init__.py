"""MM utilities package."""
from brainxdl.mm.utils.io import (
    compute_run_slug,
    ensure_unique_path,
    get_task_list,
    resolve_checkpoint_dir,
    slugify,
)
from brainxdl.xai.latent import (
    collect_latent_embeddings,
    compute_tsne,
    compute_umap,
    ensure_shared_manifold_hparams,
    generate_latent_manifolds,
    plot_scatter,
)

__all__ = [
    "compute_run_slug",
    "ensure_unique_path",
    "get_task_list",
    "resolve_checkpoint_dir",
    "slugify",
    "collect_latent_embeddings",
    "compute_tsne",
    "compute_umap",
    "ensure_shared_manifold_hparams",
    "generate_latent_manifolds",
    "plot_scatter",
]
