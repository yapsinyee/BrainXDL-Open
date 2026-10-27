"""Backward-compatible MM import path for latent-space XAI utilities."""

from brainxdl.xai.latent import (
    collect_latent_embeddings,
    compute_tsne,
    compute_umap,
    ensure_shared_manifold_hparams,
    generate_latent_manifolds,
    plot_scatter,
)

__all__ = [
    "collect_latent_embeddings",
    "compute_tsne",
    "compute_umap",
    "ensure_shared_manifold_hparams",
    "generate_latent_manifolds",
    "plot_scatter",
]
