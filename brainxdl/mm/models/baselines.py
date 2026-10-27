"""Baseline model architectures for the MM module.

Contains project-owned late-fusion, ST-embedding-fusion, and fMRI GCN models.

Source: extracted from legacy_code/2-MM/main.py.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from brainxdl.common.connectivity import knn_thresholding
from brainxdl.mm.models.brainmgf import GraphConvModel, MoEFusion


class LateFusionMLP(nn.Module):
    """Late-fusion baseline that concatenates modality embeddings before classification."""

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
    ) -> None:
        super().__init__()

        if not (use_fmri and use_eeg):
            raise ValueError("LateFusionMLP requires both fMRI and EEG modalities.")

        self.k_fmri = k_fmri
        self.k_eeg = k_eeg
        self.use_fmri = use_fmri
        self.use_eeg = use_eeg

        self.gnn_fmri = GraphConvModel(in_channels_fmri, nrois_fmri, embed_dim)
        self.fc_p_fmri = nn.Sequential(
            nn.Linear(nrois_fmri, nrois_fmri),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.gnn_eeg = GraphConvModel(in_channels_eeg, nrois_eeg, embed_dim)
        self.fc_p_eeg = nn.Sequential(
            nn.Linear(nrois_eeg, nrois_eeg),
            nn.LeakyReLU(negative_slope=0.2),
        )

        fused_dim = 2 * embed_dim
        self.norm = nn.LayerNorm(fused_dim, elementwise_affine=True)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(32, fc_out),
        )

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
        if pearson_fmri is None or pcorr_fmri is None or pseudo_fmri is None:
            raise ValueError("fMRI tensors must be provided for LateFusionMLP.")
        if pearson_eeg is None or pcorr_eeg is None or pseudo_eeg is None:
            raise ValueError("EEG tensors must be provided for LateFusionMLP.")

        pcorr_fmri = knn_thresholding(pcorr_fmri, k=self.k_fmri)
        if edge_mask_fmri is not None:
            if edge_mask_fmri.shape != pcorr_fmri.shape:
                raise ValueError("edge_mask_fmri must match the shape of the fMRI adjacency tensor.")
            pcorr_fmri = pcorr_fmri * edge_mask_fmri
        pseudo_fmri = self.fc_p_fmri(pseudo_fmri)
        node_feat_fmri = pearson_fmri + pseudo_fmri
        x_fmri = self.gnn_fmri(node_feat_fmri, pcorr_fmri).mean(dim=1)

        pcorr_eeg = knn_thresholding(pcorr_eeg, k=self.k_eeg)
        if edge_mask_eeg is not None:
            if edge_mask_eeg.shape != pcorr_eeg.shape:
                raise ValueError("edge_mask_eeg must match the shape of the EEG adjacency tensor.")
            pcorr_eeg = pcorr_eeg * edge_mask_eeg
        pseudo_eeg = self.fc_p_eeg(pseudo_eeg)
        node_feat_eeg = pearson_eeg + pseudo_eeg
        x_eeg = self.gnn_eeg(node_feat_eeg, pcorr_eeg).mean(dim=1)

        fused = torch.cat([x_fmri, x_eeg], dim=-1)
        return self.norm(fused)

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
        fused_latent = self.encode(
            pearson_fmri,
            pearson_eeg,
            pcorr_fmri,
            pcorr_eeg,
            pseudo_fmri,
            pseudo_eeg,
            edge_mask_fmri=edge_mask_fmri,
            edge_mask_eeg=edge_mask_eeg,
        )
        return self.classifier(fused_latent)


class STEmbeddingFusion(nn.Module):
    """Fuse frozen ST fMRI readout embeddings with the existing EEG graph branch."""

    def __init__(
        self,
        st_embedding_dim: int,
        in_channels_eeg: int,
        embed_dim: int,
        fc_out: int,
        nrois_eeg: int,
        k_eeg: int = 40,
        dropout: float = 0.3,
        use_eeg: bool = True,
        fusion_mode: str = "late",
    ) -> None:
        super().__init__()
        fusion_mode = fusion_mode.lower()
        if fusion_mode not in {"st_only", "late", "gated"}:
            raise ValueError("fusion_mode must be one of: st_only, late, gated")
        if fusion_mode in {"late", "gated"} and not use_eeg:
            raise ValueError("late/gated ST embedding fusion requires use_eeg=True")

        self.use_fmri = True
        self.use_eeg = use_eeg
        self.fusion_mode = fusion_mode
        self.k_eeg = k_eeg
        self._last_gate_weights: Optional[torch.Tensor] = None

        self.st_projector = nn.Sequential(
            nn.Linear(st_embedding_dim, embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(embed_dim),
        )

        if self.use_eeg:
            self.gnn_eeg = GraphConvModel(in_channels_eeg, nrois_eeg, embed_dim)
            self.fc_p_eeg = nn.Sequential(
                nn.Linear(nrois_eeg, nrois_eeg),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            self.gnn_eeg = None
            self.fc_p_eeg = None

        if self.fusion_mode == "gated":
            self.fusion = MoEFusion(embed_dim)
            latent_dim = embed_dim
        elif self.fusion_mode == "late":
            self.fusion = None
            latent_dim = 2 * embed_dim
        else:
            self.fusion = None
            latent_dim = embed_dim

        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=True)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(32, fc_out),
        )

    def encode(
        self,
        st_fmri_embedding: torch.Tensor,
        pearson_eeg: Optional[torch.Tensor] = None,
        pcorr_eeg: Optional[torch.Tensor] = None,
        pseudo_eeg: Optional[torch.Tensor] = None,
        edge_mask_eeg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if st_fmri_embedding is None:
            raise ValueError("st_fmri_embedding must be provided.")
        self._last_gate_weights = None
        x_st = self.st_projector(st_fmri_embedding)

        if self.fusion_mode == "st_only":
            return self.norm(x_st)

        if pearson_eeg is None or pcorr_eeg is None or pseudo_eeg is None:
            raise ValueError("EEG tensors must be provided for ST embedding fusion.")
        pcorr_eeg = knn_thresholding(pcorr_eeg, k=self.k_eeg)
        if edge_mask_eeg is not None:
            if edge_mask_eeg.shape != pcorr_eeg.shape:
                raise ValueError("edge_mask_eeg must match the shape of the EEG adjacency tensor.")
            pcorr_eeg = pcorr_eeg * edge_mask_eeg
        pseudo_eeg = self.fc_p_eeg(pseudo_eeg)
        node_feat_eeg = pearson_eeg + pseudo_eeg
        x_eeg = self.gnn_eeg(node_feat_eeg, pcorr_eeg).mean(dim=1)

        if self.fusion_mode == "gated":
            fused = self.fusion(x_st, x_eeg)
            self._last_gate_weights = getattr(self.fusion, "_last_weights", None)
        else:
            fused = torch.cat([x_st, x_eeg], dim=-1)
        return self.norm(fused)

    def forward(
        self,
        st_fmri_embedding: torch.Tensor,
        pearson_eeg: Optional[torch.Tensor] = None,
        pcorr_eeg: Optional[torch.Tensor] = None,
        pseudo_eeg: Optional[torch.Tensor] = None,
        edge_mask_eeg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.classifier(
            self.encode(
                st_fmri_embedding,
                pearson_eeg=pearson_eeg,
                pcorr_eeg=pcorr_eeg,
                pseudo_eeg=pseudo_eeg,
                edge_mask_eeg=edge_mask_eeg,
            )
        )

    def get_latest_gate_weights(self) -> Optional[torch.Tensor]:
        return self._last_gate_weights


class FmriGCN(nn.Module):
    """Single-modality GCN baseline operating on fMRI connectivity graphs."""

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int,
        hidden_dim: int,
        out_dim: int,
        k: int = 40,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.k = max(1, k)
        self.encoder = GraphConvModel(num_nodes, num_nodes, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(
        self,
        pearson_fmri: torch.Tensor,
        pcorr_fmri: torch.Tensor,
        pseudo_fmri: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        adjacency = knn_thresholding(pcorr_fmri, k=self.k)
        node_features = pearson_fmri if pseudo_fmri is None else pearson_fmri + pseudo_fmri
        x = self.encoder(node_features, adjacency)
        x = x.mean(dim=1)
        return self.norm(x)

    def forward(
        self,
        pearson_fmri: torch.Tensor,
        pcorr_fmri: torch.Tensor,
        pseudo_fmri: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.classifier(self.encode(pearson_fmri, pcorr_fmri, pseudo_fmri))


