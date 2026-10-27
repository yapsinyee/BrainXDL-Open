"""MM-BrainXDL multimodal GCN model and factory utilities.

Contains the primary MM-BrainXDL architecture, supporting building blocks
(MoEFusion, GraphConvModel), and the model factory / dispatch helpers
(build_model, forward_model) used by the training and evaluation loops.

Source: extracted from legacy_code/2-MM/main.py.
"""

import argparse
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from brainxdl.common.connectivity import knn_thresholding


def _normalize_model_kind(model_kind: str) -> str:
    model_kind = model_kind.lower()
    if model_kind == "brainmgf_dynamic":
        return "mm_brainxdl_dynamic"
    return model_kind


class MoEFusion(nn.Module):
    """Dynamic gated mixture-of-experts fusion module."""

    def __init__(self, embed_dim: int, num_experts: int = 4) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, num_experts),
            nn.GELU(),
            nn.Linear(num_experts, 2),
        )
        self._last_weights: Optional[torch.Tensor] = None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.dim() == 3:
            x1 = x1.mean(dim=1)
        if x2.dim() == 3:
            x2 = x2.mean(dim=1)
        gate_input = torch.cat([x1, x2], dim=-1)
        weights = F.softmax(self.gate(gate_input), dim=-1)
        self._last_weights = weights
        return weights[:, 0:1] * x1 + weights[:, 1:2] * x2


class FeaturewiseFusion(nn.Module):
    """Bounded per-feature modality gates for concatenative fusion."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )
        self._last_weights: Optional[torch.Tensor] = None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.dim() == 3:
            x1 = x1.mean(dim=1)
        if x2.dim() == 3:
            x2 = x2.mean(dim=1)
        weights = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        weights = weights.reshape(weights.shape[0], 2, x1.shape[-1])
        self._last_weights = weights
        return torch.cat([weights[:, 0] * x1, weights[:, 1] * x2], dim=-1)


class ConnectivitySkip(nn.Module):
    """Compact projection of a connectivity matrix's strict upper triangle."""

    def __init__(self, num_rois: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        rows, columns = torch.triu_indices(num_rois, num_rois, offset=1)
        self.register_buffer("triangle_rows", rows, persistent=False)
        self.register_buffer("triangle_columns", columns, persistent=False)
        self.network = nn.Sequential(
            nn.Linear(len(rows), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, connectivity: torch.Tensor) -> torch.Tensor:
        return self.network(connectivity[:, self.triangle_rows, self.triangle_columns])


class ResidualConnectivityProjection(nn.Module):
    """Project raw connectivity into the graph embedding space for residual use."""

    def __init__(
        self, num_rois: int, bottleneck_dim: int, embed_dim: int, dropout: float
    ) -> None:
        super().__init__()
        rows, columns = torch.triu_indices(num_rois, num_rois, offset=1)
        self.register_buffer("triangle_rows", rows, persistent=False)
        self.register_buffer("triangle_columns", columns, persistent=False)
        self.network = nn.Sequential(
            nn.Linear(len(rows), bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, embed_dim),
        )

    def forward(self, connectivity: torch.Tensor) -> torch.Tensor:
        upper = connectivity[:, self.triangle_rows, self.triangle_columns]
        return self.network(upper)


class GraphReconstructionHead(nn.Module):
    """ST-BrainXDL-inspired inner-product decoder for static graph support.

    MM's row-wise top-k operation can produce an asymmetric signed graph, while
    an inner product is necessarily symmetric.  The target is therefore the
    symmetric support of the absolute thresholded adjacency.  Scaling by the
    latent width keeps logits finite without changing the parameter-free ST
    decoder principle.
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout = float(dropout)

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        latent = F.dropout(node_embedding, self.dropout, training=self.training)
        scale = node_embedding.shape[-1] ** -0.5
        return torch.matmul(latent, latent.transpose(-1, -2)) * scale

    @staticmethod
    def target(adjacency: torch.Tensor) -> torch.Tensor:
        support = adjacency.abs().gt(0)
        return (support | support.transpose(-1, -2)).to(adjacency.dtype)

    def loss(
        self, node_embedding: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        logits = self(node_embedding)
        target = self.target(adjacency)
        positive = target.sum()
        negative = target.numel() - positive
        pos_weight = (negative / positive.clamp_min(1)).clamp_min(1)
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        )


class GraphConvModel(nn.Module):
    """Three-layer graph convolutional encoder for ROI features."""

    def __init__(self, node_input_dim: int, roi_num: int, embed_dim: int) -> None:
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.block0 = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(inner_dim, inner_dim),
        )
        self.bn1 = nn.BatchNorm1d(inner_dim)
        self.block1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = nn.BatchNorm1d(inner_dim)
        self.block2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = nn.BatchNorm1d(embed_dim)

    def forward_components(
        self, node_feature: torch.Tensor, data_graph: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return every graph block output for deterministic encoder probes."""
        batch_size = data_graph.shape[0]
        x = torch.einsum("bij,bjk->bik", data_graph, node_feature)

        x = self.block0(x)
        x = x.reshape(batch_size * self.roi_num, -1)
        x = self.bn1(x)
        x = x.reshape(batch_size, self.roi_num, -1)
        block0 = x

        x = torch.einsum("bij,bjk->bik", data_graph, x)
        x = self.block1(x)
        x = x.reshape(batch_size * self.roi_num, -1)
        x = self.bn2(x)
        x = x.reshape(batch_size, self.roi_num, -1)
        block1 = x

        x = torch.einsum("bij,bjk->bik", data_graph, x)
        x = self.block2(x)
        x = x.reshape(batch_size * self.roi_num, -1)
        x = self.bn3(x)
        x = x.reshape(batch_size, self.roi_num, -1)
        return {"block0": block0, "block1": block1, "block2": x}

    def forward(self, node_feature: torch.Tensor, data_graph: torch.Tensor) -> torch.Tensor:
        return self.forward_components(node_feature, data_graph)["block2"]


class SelfPreservingGraphConvModel(nn.Module):
    """Graph encoder whose neighbourhood updates retain a learned self pathway.

    This is an opt-in refinement.  The published :class:`GraphConvModel` remains
    the default and is not changed.
    """

    def __init__(self, node_input_dim: int, roi_num: int, embed_dim: int) -> None:
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.neigh0 = nn.Linear(node_input_dim, inner_dim)
        self.self0 = nn.Linear(node_input_dim, inner_dim, bias=False)
        self.block0_tail = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(inner_dim, inner_dim),
        )
        self.bn1 = nn.BatchNorm1d(inner_dim)

        self.neigh1 = nn.Linear(inner_dim, inner_dim)
        self.self1 = nn.Linear(inner_dim, inner_dim, bias=False)
        self.block1_tail = nn.LeakyReLU(negative_slope=0.2)
        self.bn2 = nn.BatchNorm1d(inner_dim)

        self.neigh2 = nn.Linear(inner_dim, 64)
        self.self2 = nn.Linear(inner_dim, 64, bias=False)
        self.block2_tail = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = nn.BatchNorm1d(embed_dim)

    @staticmethod
    def _propagate(data_graph: torch.Tensor, node_feature: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bij,bjk->bik", data_graph, node_feature)

    @staticmethod
    def _batch_norm(
        values: torch.Tensor, normalizer: nn.BatchNorm1d, batch_size: int, roi_num: int
    ) -> torch.Tensor:
        values = normalizer(values.reshape(batch_size * roi_num, -1))
        return values.reshape(batch_size, roi_num, -1)

    def forward_components(
        self, node_feature: torch.Tensor, data_graph: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return all three self-preserving graph-block outputs."""
        batch_size = data_graph.shape[0]

        neighbours = self._propagate(data_graph, node_feature)
        x = self.block0_tail(self.neigh0(neighbours) + self.self0(node_feature))
        block0 = self._batch_norm(x, self.bn1, batch_size, self.roi_num)

        neighbours = self._propagate(data_graph, block0)
        x = self.block1_tail(self.neigh1(neighbours) + self.self1(block0))
        block1 = self._batch_norm(x, self.bn2, batch_size, self.roi_num)

        neighbours = self._propagate(data_graph, block1)
        x = self.block2_tail(self.neigh2(neighbours) + self.self2(block1))
        block2 = self._batch_norm(x, self.bn3, batch_size, self.roi_num)
        return {"block0": block0, "block1": block1, "block2": block2}

    def forward(self, node_feature: torch.Tensor, data_graph: torch.Tensor) -> torch.Tensor:
        return self.forward_components(node_feature, data_graph)["block2"]


class SignSeparatedFirstBlockGraphConvModel(nn.Module):
    """Published graph encoder with sign-separated normalised first propagation."""

    def __init__(self, node_input_dim: int, roi_num: int, embed_dim: int) -> None:
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.positive0 = nn.Linear(node_input_dim, inner_dim)
        self.negative0 = nn.Linear(node_input_dim, inner_dim, bias=False)
        self.block0_tail = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(inner_dim, inner_dim),
        )
        self.bn1 = nn.BatchNorm1d(inner_dim)
        self.block1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = nn.BatchNorm1d(inner_dim)
        self.block2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = nn.BatchNorm1d(embed_dim)

    @staticmethod
    def _normalize_nonnegative(adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True)
        return adjacency / degree.clamp_min(torch.finfo(adjacency.dtype).eps)

    @staticmethod
    def _propagate(data_graph: torch.Tensor, node_feature: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bij,bjk->bik", data_graph, node_feature)

    @staticmethod
    def _batch_norm(
        values: torch.Tensor, normalizer: nn.BatchNorm1d, batch_size: int, roi_num: int
    ) -> torch.Tensor:
        values = normalizer(values.reshape(batch_size * roi_num, -1))
        return values.reshape(batch_size, roi_num, -1)

    def forward_components(
        self, node_feature: torch.Tensor, data_graph: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return block outputs while changing only the first propagation operator."""
        batch_size = data_graph.shape[0]
        positive = self._normalize_nonnegative(data_graph.clamp_min(0))
        negative = self._normalize_nonnegative((-data_graph).clamp_min(0))
        positive_message = self._propagate(positive, node_feature)
        negative_message = self._propagate(negative, node_feature)
        x = self.positive0(positive_message) + self.negative0(negative_message)
        x = self.block0_tail(x)
        block0 = self._batch_norm(x, self.bn1, batch_size, self.roi_num)

        x = self._propagate(data_graph, block0)
        x = self.block1(x)
        block1 = self._batch_norm(x, self.bn2, batch_size, self.roi_num)

        x = self._propagate(data_graph, block1)
        x = self.block2(x)
        block2 = self._batch_norm(x, self.bn3, batch_size, self.roi_num)
        return {"block0": block0, "block1": block1, "block2": block2}

    def forward(self, node_feature: torch.Tensor, data_graph: torch.Tensor) -> torch.Tensor:
        return self.forward_components(node_feature, data_graph)["block2"]


def _summarize_nodes(node_embedding: torch.Tensor) -> torch.Tensor:
    """Deterministic mean/std/max aggregation without fitted preprocessing."""
    return torch.cat(
        [
            node_embedding.mean(dim=1),
            node_embedding.std(dim=1, unbiased=False),
            node_embedding.amax(dim=1),
        ],
        dim=-1,
    )


class MMBrainXDL(nn.Module):
    """MM-BrainXDL multimodal GCN for PsiConnect fMRI + EEG classification."""

    def __init__(
        self,
        in_channels_fmri: int,
        in_channels_eeg: int,
        embed_dim: int,
        fc_out: int,
        nrois_fmri: int,
        nrois_eeg: int,
        k_fmri: int = 40,
        k_eeg: int = 40,
        dropout: float = 0.3,
        use_fmri: bool = True,
        use_eeg: bool = True,
        fusion_mode: str = "gated",
        gate_type: str = "scalar",
        connectivity_skip_dim: int = 0,
        residual_connectivity_dim: int = 0,
        residual_lambda: float = 1.0,
        classifier_type: str = "mlp",
        graph_encoder_type: str = "published",
        graph_reconstruction: bool = False,
        graph_decoder_dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if not (use_fmri or use_eeg):
            raise ValueError("At least one modality must be enabled.")

        self.k_fmri = k_fmri
        self.k_eeg = k_eeg
        self.use_fmri = use_fmri
        self.use_eeg = use_eeg
        self.embed_dim = embed_dim
        self.connectivity_skip_dim = int(connectivity_skip_dim)
        self.residual_connectivity_dim = int(residual_connectivity_dim)
        self.residual_lambda = float(residual_lambda)
        self._last_gate_weights: Optional[torch.Tensor] = None
        self._last_components: Optional[Dict[str, torch.Tensor]] = None
        self._last_reconstruction_losses: Optional[Dict[str, torch.Tensor]] = None

        fusion_mode_normalized = fusion_mode.lower()
        if fusion_mode_normalized in {"avg", "mean"}:
            fusion_mode_normalized = "average"
        if fusion_mode_normalized == "adaptive_sum":
            fusion_mode_normalized = "gated"
        if fusion_mode_normalized not in {"gated", "average", "gated_concat", "concat"}:
            raise ValueError(
                f"Unsupported fusion_mode '{fusion_mode}'. Expected gated/adaptive_sum, "
                "average, concat, or gated_concat."
            )
        self.fusion_mode = fusion_mode_normalized
        self.gate_type = gate_type.lower()
        if self.gate_type not in {"scalar", "featurewise"}:
            raise ValueError("gate_type must be 'scalar' or 'featurewise'")
        self.classifier_type = classifier_type.lower()
        if self.classifier_type not in {"mlp", "linear"}:
            raise ValueError("classifier_type must be 'mlp' or 'linear'")
        self.graph_encoder_type = graph_encoder_type.lower()
        if self.graph_encoder_type not in {
            "published", "self_preserving", "sign_separated_first"
        }:
            raise ValueError(
                "graph_encoder_type must be 'published', 'self_preserving', "
                "or 'sign_separated_first'"
            )
        if self.connectivity_skip_dim < 0:
            raise ValueError("connectivity_skip_dim cannot be negative")
        if self.residual_connectivity_dim < 0:
            raise ValueError("residual_connectivity_dim cannot be negative")
        self.graph_reconstruction = bool(graph_reconstruction)

        if self.use_fmri:
            encoder_class = {
                "published": GraphConvModel,
                "self_preserving": SelfPreservingGraphConvModel,
                "sign_separated_first": SignSeparatedFirstBlockGraphConvModel,
            }[self.graph_encoder_type]
            self.gnn_fmri = encoder_class(in_channels_fmri, nrois_fmri, embed_dim)
            self.fc_p_fmri = nn.Sequential(
                nn.Linear(nrois_fmri, nrois_fmri),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            self.gnn_fmri = None
            self.fc_p_fmri = None

        if self.use_eeg:
            encoder_class = {
                "published": GraphConvModel,
                "self_preserving": SelfPreservingGraphConvModel,
                "sign_separated_first": SignSeparatedFirstBlockGraphConvModel,
            }[self.graph_encoder_type]
            self.gnn_eeg = encoder_class(in_channels_eeg, nrois_eeg, embed_dim)
            self.fc_p_eeg = nn.Sequential(
                nn.Linear(nrois_eeg, nrois_eeg),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            self.gnn_eeg = None
            self.fc_p_eeg = None

        self.graph_decoder_fmri = (
            GraphReconstructionHead(graph_decoder_dropout)
            if self.graph_reconstruction and self.use_fmri else None
        )
        self.graph_decoder_eeg = (
            GraphReconstructionHead(graph_decoder_dropout)
            if self.graph_reconstruction and self.use_eeg else None
        )

        if self.use_fmri and self.use_eeg and self.fusion_mode in {"gated", "gated_concat"}:
            self.fusion = FeaturewiseFusion(embed_dim) if self.gate_type == "featurewise" else MoEFusion(embed_dim)
        else:
            self.fusion = None

        if self.connectivity_skip_dim and not (self.use_fmri and self.use_eeg):
            raise ValueError("Connectivity skips currently require both modalities")
        self.skip_fmri = ConnectivitySkip(nrois_fmri, self.connectivity_skip_dim, dropout) if self.connectivity_skip_dim else None
        self.skip_eeg = ConnectivitySkip(nrois_eeg, self.connectivity_skip_dim, dropout) if self.connectivity_skip_dim else None
        if self.residual_connectivity_dim and not (self.use_fmri and self.use_eeg):
            raise ValueError("Residual connectivity projections currently require both modalities")
        self.residual_fmri = (
            ResidualConnectivityProjection(
                nrois_fmri, self.residual_connectivity_dim, embed_dim, dropout
            )
            if self.residual_connectivity_dim else None
        )
        self.residual_eeg = (
            ResidualConnectivityProjection(
                nrois_eeg, self.residual_connectivity_dim, embed_dim, dropout
            )
            if self.residual_connectivity_dim else None
        )

        fused_dim = embed_dim if self.fusion_mode in {"gated", "average"} else 2 * embed_dim
        fused_dim += 2 * self.connectivity_skip_dim
        self.norm = nn.LayerNorm(fused_dim, elementwise_affine=True)
        if self.classifier_type == "linear":
            self.classifier = nn.Linear(fused_dim, fc_out)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=dropout),
                nn.Linear(256, 32),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=dropout),
                nn.Linear(32, fc_out),
            )

    def encode_components(
        self,
        pearson_fmri: torch.Tensor,
        pearson_eeg: torch.Tensor,
        pcorr_fmri: torch.Tensor,
        pcorr_eeg: torch.Tensor,
        pseudo_fmri: torch.Tensor,
        pseudo_eeg: torch.Tensor,
        edge_mask_fmri: Optional[torch.Tensor] = None,
        edge_mask_eeg: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return named intermediate tensors used by refined MM diagnostics."""
        x_fmri: Optional[torch.Tensor] = None
        x_eeg: Optional[torch.Tensor] = None
        graph_layers_fmri: Optional[Dict[str, torch.Tensor]] = None
        graph_layers_eeg: Optional[Dict[str, torch.Tensor]] = None
        self._last_gate_weights = None
        self._last_components = None
        self._last_reconstruction_losses = None

        if self.use_fmri:
            if pearson_fmri is None or pcorr_fmri is None or pseudo_fmri is None:
                raise ValueError("fMRI tensors must be provided when use_fmri=True")
            expected = self.gnn_fmri.roi_num
            if pearson_fmri.ndim != 3 or pearson_fmri.shape[1:] != (expected, expected):
                raise ValueError(f"Expected fMRI connectivity [B,{expected},{expected}], got {tuple(pearson_fmri.shape)}")
            if not all(torch.isfinite(value).all() for value in (pearson_fmri, pcorr_fmri, pseudo_fmri)):
                raise ValueError("fMRI inputs contain NaN or Inf")
            pcorr_fmri = knn_thresholding(pcorr_fmri, k=self.k_fmri)
            if edge_mask_fmri is not None:
                if edge_mask_fmri.shape != pcorr_fmri.shape:
                    raise ValueError("edge_mask_fmri must match the shape of the fMRI adjacency tensor.")
                pcorr_fmri = pcorr_fmri * edge_mask_fmri
            pseudo_fmri = self.fc_p_fmri(pseudo_fmri)
            node_feat_fmri = pearson_fmri + pseudo_fmri
            graph_layers_fmri = self.gnn_fmri.forward_components(node_feat_fmri, pcorr_fmri)
            x_fmri = graph_layers_fmri["block2"]
            expected_embedding = (pearson_fmri.shape[0], expected, self.embed_dim)
            if x_fmri.shape != expected_embedding:
                raise RuntimeError(f"Invalid fMRI graph embedding shape: {tuple(x_fmri.shape)}")

        if self.use_eeg:
            if pearson_eeg is None or pcorr_eeg is None or pseudo_eeg is None:
                raise ValueError("EEG tensors must be provided when use_eeg=True")
            expected = self.gnn_eeg.roi_num
            if pearson_eeg.ndim != 3 or pearson_eeg.shape[1:] != (expected, expected):
                raise ValueError(f"Expected EEG connectivity [B,{expected},{expected}], got {tuple(pearson_eeg.shape)}")
            if not all(torch.isfinite(value).all() for value in (pearson_eeg, pcorr_eeg, pseudo_eeg)):
                raise ValueError("EEG inputs contain NaN or Inf")
            pcorr_eeg = knn_thresholding(pcorr_eeg, k=self.k_eeg)
            if edge_mask_eeg is not None:
                if edge_mask_eeg.shape != pcorr_eeg.shape:
                    raise ValueError("edge_mask_eeg must match the shape of the EEG adjacency tensor.")
                pcorr_eeg = pcorr_eeg * edge_mask_eeg
            pseudo_eeg = self.fc_p_eeg(pseudo_eeg)
            node_feat_eeg = pearson_eeg + pseudo_eeg
            graph_layers_eeg = self.gnn_eeg.forward_components(node_feat_eeg, pcorr_eeg)
            x_eeg = graph_layers_eeg["block2"]
            expected_embedding = (pearson_eeg.shape[0], expected, self.embed_dim)
            if x_eeg.shape != expected_embedding:
                raise RuntimeError(f"Invalid EEG graph embedding shape: {tuple(x_eeg.shape)}")

        if self.graph_reconstruction:
            reconstruction_losses: Dict[str, torch.Tensor] = {}
            if x_fmri is not None:
                if self.graph_decoder_fmri is None:
                    raise RuntimeError("fMRI graph decoder was not initialised")
                reconstruction_losses["fmri"] = self.graph_decoder_fmri.loss(
                    x_fmri, pcorr_fmri
                )
            if x_eeg is not None:
                if self.graph_decoder_eeg is None:
                    raise RuntimeError("EEG graph decoder was not initialised")
                reconstruction_losses["eeg"] = self.graph_decoder_eeg.loss(
                    x_eeg, pcorr_eeg
                )
            if not reconstruction_losses or not all(
                torch.isfinite(value) for value in reconstruction_losses.values()
            ):
                raise RuntimeError("Invalid graph reconstruction loss")
            self._last_reconstruction_losses = reconstruction_losses

        if self.use_fmri and self.use_eeg:
            if x_fmri is None or x_eeg is None:
                raise RuntimeError("Fusion requires both modality embeddings.")
            graph_fmri = x_fmri.mean(dim=1)
            graph_eeg = x_eeg.mean(dim=1)
            if self.residual_connectivity_dim:
                if self.residual_fmri is None or self.residual_eeg is None:
                    raise RuntimeError("Residual projections were not initialised")
                residual_fmri = self.residual_fmri(pearson_fmri)
                residual_eeg = self.residual_eeg(pearson_eeg)
                pooled_fmri = graph_fmri + self.residual_lambda * residual_fmri
                pooled_eeg = graph_eeg + self.residual_lambda * residual_eeg
            else:
                residual_fmri = None
                residual_eeg = None
                pooled_fmri = graph_fmri
                pooled_eeg = graph_eeg
            z_concat = torch.cat([pooled_fmri, pooled_eeg], dim=-1)
            adaptive_sum = 0.5 * (pooled_fmri + pooled_eeg)
            gate_logits: Optional[torch.Tensor] = None
            if self.fusion_mode in {"gated", "gated_concat"}:
                if self.fusion is None:
                    raise RuntimeError("Fusion module was not initialised for gated fusion.")
                fused_gate_output = self.fusion(pooled_fmri, pooled_eeg)
                self._last_gate_weights = getattr(self.fusion, "_last_weights", None)
                gate_logits = self.fusion.gate(z_concat)
                if self.gate_type == "scalar":
                    if self._last_gate_weights.shape != (z_concat.shape[0], 2):
                        raise RuntimeError(f"Invalid scalar gate shape: {tuple(self._last_gate_weights.shape)}")
                    if not torch.allclose(
                        self._last_gate_weights.sum(dim=-1),
                        torch.ones(z_concat.shape[0], device=z_concat.device),
                        atol=1e-6,
                    ):
                        raise RuntimeError("Scalar modality gate weights do not sum to one")
                    adaptive_sum = fused_gate_output
                    x = adaptive_sum if self.fusion_mode == "gated" else torch.cat(
                        [self._last_gate_weights[:, 0:1] * pooled_fmri,
                         self._last_gate_weights[:, 1:2] * pooled_eeg], dim=-1
                    )
                else:
                    if self.fusion_mode == "gated":
                        raise ValueError("featurewise gates require fusion_mode='gated_concat'")
                    x = fused_gate_output
                    adaptive_sum = x[:, : self.embed_dim] + x[:, self.embed_dim :]
            elif self.fusion_mode == "concat":
                x = z_concat
            else:
                x = adaptive_sum
        elif self.use_fmri:
            if x_fmri is None:
                raise RuntimeError("fMRI branch produced no output.")
            x = x_fmri
        else:
            if x_eeg is None:
                raise RuntimeError("EEG branch produced no output.")
            x = x_eeg

        if x.dim() == 3:
            # When only one modality is enabled, pool node embeddings across ROIs.
            x = x.mean(dim=1)
        components: Dict[str, torch.Tensor] = {"fused_pre_norm": x}
        if self.use_fmri and self.use_eeg:
            components.update(
                z_graph_fmri=graph_fmri,
                z_graph_eeg=graph_eeg,
                z_fmri=pooled_fmri,
                z_eeg=pooled_eeg,
                z_concat=z_concat,
                adaptive_sum=adaptive_sum,
            )
            if graph_layers_fmri is None or graph_layers_eeg is None:
                raise RuntimeError("Graph layer outputs are unavailable")
            for block_name, values in graph_layers_fmri.items():
                components[f"fmri_{block_name}_summary"] = _summarize_nodes(values)
            for block_name, values in graph_layers_eeg.items():
                components[f"eeg_{block_name}_summary"] = _summarize_nodes(values)
            if residual_fmri is not None and residual_eeg is not None:
                components["z_residual_fmri"] = residual_fmri
                components["z_residual_eeg"] = residual_eeg
                components["z_refined_fmri"] = pooled_fmri
                components["z_refined_eeg"] = pooled_eeg
            if gate_logits is not None:
                components["gate_logits"] = gate_logits
            if self._last_gate_weights is not None:
                components["gate_weights"] = self._last_gate_weights
        if self.connectivity_skip_dim:
            skip_fmri = self.skip_fmri(pearson_fmri)
            skip_eeg = self.skip_eeg(pearson_eeg)
            components["skip_fmri"] = skip_fmri
            components["skip_eeg"] = skip_eeg
            x = torch.cat([x, skip_fmri, skip_eeg], dim=-1)
        components["fused"] = self.norm(x)
        for name, value in components.items():
            if value.shape[0] != x.shape[0] or not torch.isfinite(value).all():
                raise RuntimeError(f"Invalid refined MM component: {name} {tuple(value.shape)}")
        self._last_components = components
        return components

    def encode(
        self,
        pearson_fmri: torch.Tensor,
        pearson_eeg: torch.Tensor,
        pcorr_fmri: torch.Tensor,
        pcorr_eeg: torch.Tensor,
        pseudo_fmri: torch.Tensor,
        pseudo_eeg: torch.Tensor,
        edge_mask_fmri: Optional[torch.Tensor] = None,
        edge_mask_eeg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the normalized latent embedding prior to the classifier."""
        return self.encode_components(
            pearson_fmri, pearson_eeg, pcorr_fmri, pcorr_eeg, pseudo_fmri, pseudo_eeg,
            edge_mask_fmri=edge_mask_fmri, edge_mask_eeg=edge_mask_eeg,
        )["fused"]

    def forward(
        self,
        pearson_fmri: torch.Tensor,
        pearson_eeg: torch.Tensor,
        pcorr_fmri: torch.Tensor,
        pcorr_eeg: torch.Tensor,
        pseudo_fmri: torch.Tensor,
        pseudo_eeg: torch.Tensor,
        edge_mask_fmri: Optional[torch.Tensor] = None,
        edge_mask_eeg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.classifier(
            self.encode(
                pearson_fmri,
                pearson_eeg,
                pcorr_fmri,
                pcorr_eeg,
                pseudo_fmri,
                pseudo_eeg,
                edge_mask_fmri=edge_mask_fmri,
                edge_mask_eeg=edge_mask_eeg,
            )
        )

    def get_latest_gate_weights(self) -> Optional[torch.Tensor]:
        """Return the gate weights from the most recent forward pass."""
        return self._last_gate_weights

    def get_latest_components(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return intermediate tensors from the most recent encode/forward pass."""
        return self._last_components

    def get_latest_reconstruction_losses(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return modality reconstruction losses from the most recent forward pass."""
        return self._last_reconstruction_losses


def forward_model(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    args: argparse.Namespace,
    return_latent: bool = False,
    return_gates: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Dispatch a batch through the model, handling per-architecture input routing.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    if len(batch) == 8:
        pearson_fmri, pearson_eeg, pcorr_fmri, pcorr_eeg, labels, pseudo_fmri, pseudo_eeg, st_fmri_embedding = batch
    else:
        pearson_fmri, pearson_eeg, pcorr_fmri, pcorr_eeg, labels, pseudo_fmri, pseudo_eeg = batch
        st_fmri_embedding = None
    labels = labels.to(device)
    latent: Optional[torch.Tensor] = None
    gate_weights: Optional[torch.Tensor] = None

    model_kind = _normalize_model_kind(getattr(args, "model", "mmodal_gcn"))

    if model_kind in {"st_embedding_mlp", "st_embedding_fusion_mlp", "st_embedding_mmodal_gcn"}:
        if st_fmri_embedding is None:
            raise ValueError(f"{model_kind} requires an ST embedding tensor in the dataset.")
        st_fmri_embedding = st_fmri_embedding.to(device)
        if getattr(args, "use_eeg", False):
            pearson_eeg = pearson_eeg.to(device)
            pcorr_eeg = pcorr_eeg.to(device)
            pseudo_eeg = pseudo_eeg.to(device)
        else:
            pearson_eeg = None
            pcorr_eeg = None
            pseudo_eeg = None
        logits = model(st_fmri_embedding, pearson_eeg, pcorr_eeg, pseudo_eeg)
        if return_latent:
            latent = model.encode(st_fmri_embedding, pearson_eeg, pcorr_eeg, pseudo_eeg)
        if return_gates and hasattr(model, "get_latest_gate_weights"):
            latest = model.get_latest_gate_weights()
            if latest is not None:
                gate_weights = latest.detach().cpu()

    elif model_kind in {"mmodal_gcn", "mmodal_gcn_nogate", "fusion_mlp", "mm_brainxdl_dynamic"}:
        if args.use_fmri:
            pearson_fmri = pearson_fmri.to(device)
            pcorr_fmri = pcorr_fmri.to(device)
            pseudo_fmri = pseudo_fmri.to(device)
        else:
            pearson_fmri = None
            pcorr_fmri = None
            pseudo_fmri = None

        if args.use_eeg:
            pearson_eeg = pearson_eeg.to(device)
            pcorr_eeg = pcorr_eeg.to(device)
            pseudo_eeg = pseudo_eeg.to(device)
        else:
            pearson_eeg = None
            pcorr_eeg = None
            pseudo_eeg = None

        logits = model(pearson_fmri, pearson_eeg, pcorr_fmri, pcorr_eeg, pseudo_fmri, pseudo_eeg)
        if return_latent:
            if not hasattr(model, "encode"):
                raise AttributeError("Model does not expose an encode method for latent extraction.")
            latent = model.encode(pearson_fmri, pearson_eeg, pcorr_fmri, pcorr_eeg, pseudo_fmri, pseudo_eeg)
        if return_gates and hasattr(model, "get_latest_gate_weights"):
            latest = model.get_latest_gate_weights()
            if latest is not None:
                gate_weights = latest.detach().cpu()

    elif model_kind == "gcn":
        pearson_fmri = pearson_fmri.to(device)
        pcorr_fmri = pcorr_fmri.to(device)
        pseudo_fmri = pseudo_fmri.to(device)
        logits = model(pearson_fmri, pcorr_fmri, pseudo_fmri)
        if return_latent:
            latent = model.encode(pearson_fmri, pcorr_fmri, pseudo_fmri)

    else:
        raise ValueError(f"Unsupported model type: {model_kind}")

    return logits, labels, latent, gate_weights


def build_model(
    args: argparse.Namespace,
    n_fmri: int,
    n_eeg: int,
    model_out_dim: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate and return the model specified by args.model.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    # Imported here to avoid circular imports at module load time.
    from brainxdl.mm.models.baselines import FmriGCN, LateFusionMLP, STEmbeddingFusion
    from brainxdl.mm.models.dynamic_brainmgf import DynamicMMBrainXDL

    model_kind = _normalize_model_kind(getattr(args, "model", "mmodal_gcn"))

    if model_kind in {"mmodal_gcn", "mmodal_gcn_nogate"}:
        fusion_mode = "gated" if model_kind == "mmodal_gcn" else "average"
        model = MMBrainXDL(
            in_channels_fmri=n_fmri,
            in_channels_eeg=n_eeg,
            embed_dim=args.embid_dim,
            fc_out=model_out_dim,
            nrois_fmri=n_fmri,
            nrois_eeg=n_eeg,
            dropout=args.dropout,
            k_fmri=getattr(args, "k_fmri", 40),
            k_eeg=getattr(args, "k_eeg", 40),
            use_fmri=args.use_fmri,
            use_eeg=args.use_eeg,
            fusion_mode=fusion_mode,
        )
    elif model_kind == "mm_brainxdl_dynamic":
        fusion_mode = "gated"
        model = DynamicMMBrainXDL(
            in_channels_fmri=n_fmri,
            in_channels_eeg=n_eeg,
            embed_dim=args.embid_dim,
            fc_out=model_out_dim,
            nrois_fmri=n_fmri,
            nrois_eeg=n_eeg,
            dropout=args.dropout,
            k_fmri=getattr(args, "k_fmri", 40),
            k_eeg=getattr(args, "k_eeg", 40),
            use_fmri=args.use_fmri,
            use_eeg=args.use_eeg,
            fusion_mode=fusion_mode,
            temporal_pool=getattr(args, "temporal_pool", "mean"),
        )
        print(f"[build_model] mm_brainxdl_dynamic using n_fmri={n_fmri}, n_eeg={n_eeg}, embed_dim={args.embid_dim}")
    elif model_kind in {"st_embedding_mlp", "st_embedding_fusion_mlp", "st_embedding_mmodal_gcn"}:
        st_embedding_dim = getattr(args, "st_embedding_dim", None)
        if st_embedding_dim is None:
            raise ValueError(f"{model_kind} requires args.st_embedding_dim.")
        if model_kind == "st_embedding_mlp":
            fusion_mode = "st_only"
            use_eeg = False
        elif model_kind == "st_embedding_fusion_mlp":
            fusion_mode = "late"
            use_eeg = True
        else:
            fusion_mode = "gated"
            use_eeg = True
        model = STEmbeddingFusion(
            st_embedding_dim=int(st_embedding_dim),
            in_channels_eeg=n_eeg,
            embed_dim=args.embid_dim,
            fc_out=model_out_dim,
            nrois_eeg=n_eeg,
            dropout=args.dropout,
            k_eeg=getattr(args, "k_eeg", 40),
            use_eeg=use_eeg,
            fusion_mode=fusion_mode,
        )
    elif model_kind == "fusion_mlp":
        if not (args.use_fmri and args.use_eeg):
            raise ValueError("fusion_mlp requires both fMRI and EEG modalities. Use --modality fusion.")
        model = LateFusionMLP(
            in_channels_fmri=n_fmri,
            in_channels_eeg=n_eeg,
            embed_dim=args.embid_dim,
            fc_out=model_out_dim,
            nrois_fmri=n_fmri,
            nrois_eeg=n_eeg,
            dropout=args.dropout,
            k_fmri=getattr(args, "k_fmri", 40),
            k_eeg=getattr(args, "k_eeg", 40),
            use_fmri=args.use_fmri,
            use_eeg=args.use_eeg,
        )
    elif model_kind == "gcn":
        hidden_dim = getattr(args, "gcn_hidden_dim", 256)
        model = FmriGCN(
            num_nodes=n_fmri,
            embed_dim=args.embid_dim,
            hidden_dim=hidden_dim,
            out_dim=model_out_dim,
            k=getattr(args, "k_fmri", 40),
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_kind}")

    return model.to(device)


BrainMGF = MMBrainXDL
