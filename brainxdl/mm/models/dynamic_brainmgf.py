"""Dynamic MM-BrainXDL scaffold with simple temporal pooling.

This is an experimental variant that allows per-time-step adjacency matrices.
See legacy_code/2-MM/models/dynamic_brainmgf.py for the original.

Duplicated MoEFusion, GraphConvModel, and knn_thresholding have been removed;
they are imported from brainxdl.mm.models.brainmgf and brainxdl.common.connectivity.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from brainxdl.common.connectivity import knn_thresholding
from brainxdl.mm.models.brainmgf import GraphConvModel, MoEFusion


class DynamicMMBrainXDL(nn.Module):
    """Minimal dynamic-graph variant of MM-BrainXDL with temporal pooling."""

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
        temporal_pool: str = "mean",
    ) -> None:
        super().__init__()

        if not (use_fmri or use_eeg):
            raise ValueError("At least one modality must be enabled.")

        fusion_mode_normalized = fusion_mode.lower()
        if fusion_mode_normalized in {"avg", "mean"}:
            fusion_mode_normalized = "average"
        if fusion_mode_normalized not in {"gated", "average"}:
            raise ValueError(f"Unsupported fusion_mode '{fusion_mode}'. Expected one of: gated, average.")

        self.k_fmri = k_fmri
        self.k_eeg = k_eeg
        self.use_fmri = use_fmri
        self.use_eeg = use_eeg
        self.fusion_mode = fusion_mode_normalized
        self.temporal_pool = temporal_pool
        self._last_gate_weights: Optional[torch.Tensor] = None

        if self.use_fmri:
            self.gnn_fmri = GraphConvModel(in_channels_fmri, nrois_fmri, embed_dim)
            self.fc_p_fmri = nn.Sequential(
                nn.Linear(nrois_fmri, nrois_fmri),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            self.gnn_fmri = None
            self.fc_p_fmri = None

        if self.use_eeg:
            self.gnn_eeg = GraphConvModel(in_channels_eeg, nrois_eeg, embed_dim)
            self.fc_p_eeg = nn.Sequential(
                nn.Linear(nrois_eeg, nrois_eeg),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            self.gnn_eeg = None
            self.fc_p_eeg = None

        if self.use_fmri and self.use_eeg and self.fusion_mode == "gated":
            self.fusion = MoEFusion(embed_dim)
        else:
            self.fusion = None

        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(32, fc_out),
        )

    @staticmethod
    def _flatten_time(x: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], int]:
        """Merge batch and time dims; returns (flattened, T)."""
        if x is None:
            return None, 1
        if x.dim() == 3:
            # (B, N, C) static
            return x, 1
        if x.dim() == 4:
            # Accept (B, T, N, C) or (B, N, C, T) where N==C (adjacency)
            if x.shape[1] == x.shape[2] and x.shape[-1] != x.shape[-2]:
                # assume layout (B, N, C, T) with N==C; move T to dim=1
                x = x.permute(0, 3, 1, 2)
            b, t, n, c = x.shape
            return x.reshape(b * t, n, c), t
        raise ValueError("Expected tensor shape (B, N, C) or (B, T, N, C)")

    @staticmethod
    def _flatten_adj(a: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if a is None:
            return None
        if a.dim() == 3:
            return a
        if a.dim() == 4:
            # Accept (B, T, N, N) or (B, N, N, T)
            if a.shape[1] == a.shape[2] and a.shape[-1] != a.shape[-2]:
                a = a.permute(0, 3, 1, 2)
            b, t, n, _ = a.shape
            return a.reshape(b * t, n, n)
        raise ValueError("Expected adjacency shape (B, N, N) or (B, T, N, N)")

    def _temporal_reduce(self, x: torch.Tensor, T: int) -> torch.Tensor:
        if T == 1:
            return x
        if x.dim() != 3:
            return x
        bT, n, c = x.shape
        b = bT // T
        x = x.view(b, T, n, c)
        if self.temporal_pool == "mean":
            return x.mean(dim=1)
        if self.temporal_pool == "max":
            return x.max(dim=1).values
        raise ValueError(f"Unsupported temporal_pool '{self.temporal_pool}'")

    def encode(
        self,
        pearson_fmri: Optional[torch.Tensor],
        pearson_eeg: Optional[torch.Tensor],
        pcorr_fmri: Optional[torch.Tensor],
        pcorr_eeg: Optional[torch.Tensor],
        pseudo_fmri: Optional[torch.Tensor],
        pseudo_eeg: Optional[torch.Tensor],
        edge_mask_fmri: Optional[torch.Tensor] = None,
        edge_mask_eeg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the normalized latent embedding prior to the classifier."""
        x_fmri: Optional[torch.Tensor] = None
        x_eeg: Optional[torch.Tensor] = None
        self._last_gate_weights = None

        if self.use_fmri:
            if pearson_fmri is None or pcorr_fmri is None or pseudo_fmri is None:
                raise ValueError("fMRI tensors must be provided when use_fmri=True")
            flat_pcorr_fmri = self._flatten_adj(pcorr_fmri)
            flat_pseudo_fmri, T_fmri = self._flatten_time(pseudo_fmri)
            flat_pearson_fmri, _ = self._flatten_time(pearson_fmri)

            flat_pcorr_fmri = knn_thresholding(flat_pcorr_fmri, k=self.k_fmri)
            if edge_mask_fmri is not None:
                if edge_mask_fmri.shape != pcorr_fmri.shape:
                    raise ValueError("edge_mask_fmri must match the shape of the fMRI adjacency tensor.")
                flat_mask = self._flatten_adj(edge_mask_fmri)
                flat_pcorr_fmri = flat_pcorr_fmri * flat_mask

            flat_pseudo_fmri = self.fc_p_fmri(flat_pseudo_fmri)
            node_feat_fmri = flat_pearson_fmri + flat_pseudo_fmri
            flat_x_fmri = self.gnn_fmri(node_feat_fmri, flat_pcorr_fmri)
            x_fmri = self._temporal_reduce(flat_x_fmri, T_fmri)

        if self.use_eeg:
            if pearson_eeg is None or pcorr_eeg is None or pseudo_eeg is None:
                raise ValueError("EEG tensors must be provided when use_eeg=True")
            flat_pcorr_eeg = self._flatten_adj(pcorr_eeg)
            flat_pseudo_eeg, T_eeg = self._flatten_time(pseudo_eeg)
            flat_pearson_eeg, _ = self._flatten_time(pearson_eeg)

            flat_pcorr_eeg = knn_thresholding(flat_pcorr_eeg, k=self.k_eeg)
            if edge_mask_eeg is not None:
                if edge_mask_eeg.shape != pcorr_eeg.shape:
                    raise ValueError("edge_mask_eeg must match the shape of the EEG adjacency tensor.")
                flat_mask = self._flatten_adj(edge_mask_eeg)
                flat_pcorr_eeg = flat_pcorr_eeg * flat_mask

            flat_pseudo_eeg = self.fc_p_eeg(flat_pseudo_eeg)
            node_feat_eeg = flat_pearson_eeg + flat_pseudo_eeg
            flat_x_eeg = self.gnn_eeg(node_feat_eeg, flat_pcorr_eeg)
            x_eeg = self._temporal_reduce(flat_x_eeg, T_eeg)

        if self.use_fmri and self.use_eeg:
            if x_fmri is None or x_eeg is None:
                raise RuntimeError("Fusion requires both modality embeddings.")
            if self.fusion_mode == "gated":
                if self.fusion is None:
                    raise RuntimeError("Fusion module was not initialised despite fusion_mode='gated'.")
                x = self.fusion(x_fmri, x_eeg)
                self._last_gate_weights = getattr(self.fusion, "_last_weights", None)
            else:
                pooled_fmri = x_fmri.mean(dim=1) if x_fmri.dim() == 3 else x_fmri
                pooled_eeg = x_eeg.mean(dim=1) if x_eeg.dim() == 3 else x_eeg
                x = 0.5 * (pooled_fmri + pooled_eeg)
        elif self.use_fmri:
            if x_fmri is None:
                raise RuntimeError("fMRI branch produced no output.")
            x = x_fmri.mean(dim=1) if x_fmri.dim() == 3 else x_fmri
        else:
            if x_eeg is None:
                raise RuntimeError("EEG branch produced no output.")
            x = x_eeg.mean(dim=1) if x_eeg.dim() == 3 else x_eeg

        return self.norm(x)

    def forward(
        self,
        pearson_fmri: Optional[torch.Tensor],
        pearson_eeg: Optional[torch.Tensor],
        pcorr_fmri: Optional[torch.Tensor],
        pcorr_eeg: Optional[torch.Tensor],
        pseudo_fmri: Optional[torch.Tensor],
        pseudo_eeg: Optional[torch.Tensor],
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


DynamicBrainMGF = DynamicMMBrainXDL
