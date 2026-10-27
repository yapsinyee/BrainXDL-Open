"""Backward-compatible MM import path for XAI explainers."""

from brainxdl.xai.explainer import DenseGNNExplainer, maybe_run_gnn_explanations

__all__ = ["DenseGNNExplainer", "maybe_run_gnn_explanations"]
