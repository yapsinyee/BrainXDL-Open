"""Shared explainability utilities for BrainXDL modules."""

from brainxdl.xai.explainer import DenseGNNExplainer, maybe_run_gnn_explanations
from brainxdl.xai.latent import (
    collect_latent_embeddings,
    compute_tsne,
    compute_umap,
    ensure_shared_manifold_hparams,
    generate_latent_manifolds,
    plot_scatter,
)
from brainxdl.xai.st_attention import (
    STAttentionRecord,
    TransformerAttentionCapture,
    records_to_component_series,
    summarize_attention_series,
    symmetrize_attention,
    top_undirected_edges,
)

__all__ = [
    "DenseGNNExplainer",
    "maybe_run_gnn_explanations",
    "collect_latent_embeddings",
    "compute_tsne",
    "compute_umap",
    "ensure_shared_manifold_hparams",
    "generate_latent_manifolds",
    "plot_scatter",
    "STAttentionRecord",
    "TransformerAttentionCapture",
    "records_to_component_series",
    "summarize_attention_series",
    "symmetrize_attention",
    "top_undirected_edges",
]
