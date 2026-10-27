"""Project-specific dense edge-mask explanations for BrainXDL modules.

The compatibility name ``DenseGNNExplainer`` is retained for existing scripts.
The implementation freezes the predictive model, optimizes symmetric
modality-specific edge masks, and reports signed saliency as ``mask *
adjacency``. It is not the standard PyG GNNExplainer algorithm and does not
implement gradient-times-activation attribution.
"""

import argparse
import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from brainxdl.common.connectivity import knn_thresholding
from brainxdl.mm.utils.io import compute_run_slug, slugify


class DenseGNNExplainer:
    """Optimize dense symmetric edge masks for MM-BrainXDL graphs.

    The historical name is preserved as part of the existing API. Model
    parameters are frozen while mask logits are optimized for prediction
    fidelity with size and entropy regularization.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        binary_mode: bool,
        use_fmri: bool,
        use_eeg: bool,
        lr: float = 0.1,
        epochs: int = 200,
        size_coeff: float = 0.01,
        entropy_coeff: float = 5e-4,
        mask_init: float = 0.5,
    ) -> None:
        self.model = model
        self.device = device
        self.binary_mode = binary_mode
        self.use_fmri = use_fmri
        self.use_eeg = use_eeg
        self.lr = lr
        self.epochs = epochs
        self.size_coeff = size_coeff
        self.entropy_coeff = entropy_coeff
        mask_init = float(np.clip(mask_init, 1e-3, 1 - 1e-3))
        self._mask_logit = float(np.log(mask_init / (1.0 - mask_init)))

    def _init_mask_param(self, num_nodes: int) -> nn.Parameter:
        return nn.Parameter(
            torch.full((num_nodes, num_nodes), self._mask_logit, dtype=torch.float32, device=self.device)
        )

    def _build_masks(
        self, mask_param: Optional[nn.Parameter]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        reg_term = torch.tensor(0.0, device=self.device)
        if mask_param is None:
            return None, reg_term
        mask = torch.sigmoid(mask_param)
        mask = 0.5 * (mask + mask.transpose(0, 1))
        eye = torch.eye(mask.size(0), device=self.device)
        off_diag = mask * (1.0 - eye)
        entropy = -off_diag * torch.log(off_diag + 1e-8) - (1.0 - off_diag) * torch.log(1.0 - off_diag + 1e-8)
        reg_term = reg_term + self.size_coeff * off_diag.mean() + self.entropy_coeff * entropy.mean()
        mask = off_diag + eye
        return mask.unsqueeze(0), reg_term

    def explain(
        self,
        pearson_fmri: Optional[torch.Tensor],
        pearson_eeg: Optional[torch.Tensor],
        pcorr_fmri: Optional[torch.Tensor],
        pcorr_eeg: Optional[torch.Tensor],
        pseudo_fmri: Optional[torch.Tensor],
        pseudo_eeg: Optional[torch.Tensor],
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        from brainxdl.mm.models.baselines import LateFusionMLP
        from brainxdl.mm.models.brainmgf import MMBrainXDL

        if not isinstance(self.model, (MMBrainXDL, LateFusionMLP)):
            raise TypeError("DenseGNNExplainer currently supports MM-BrainXDL and LateFusionMLP backbones.")

        fmri_nodes = pearson_fmri.shape[-1] if pearson_fmri is not None else None
        eeg_nodes = pearson_eeg.shape[-1] if pearson_eeg is not None else None

        fmri_mask_param = self._init_mask_param(fmri_nodes) if self.use_fmri and fmri_nodes else None
        eeg_mask_param = self._init_mask_param(eeg_nodes) if self.use_eeg and eeg_nodes else None
        params = [p for p in [fmri_mask_param, eeg_mask_param] if p is not None]
        optimizer = optim.Adam(params, lr=self.lr) if params else None

        self.model.eval()
        grad_state = [param.requires_grad for param in self.model.parameters()]
        for param in self.model.parameters():
            param.requires_grad_(False)

        try:
            with torch.no_grad():
                base_logits = self.model(
                    pearson_fmri,
                    pearson_eeg,
                    pcorr_fmri,
                    pcorr_eeg,
                    pseudo_fmri,
                    pseudo_eeg,
                )
            if self.binary_mode:
                probs = torch.sigmoid(base_logits).squeeze(-1)
                pred_label = int((probs >= 0.5).long().item())
                pred_prob = float(probs.item()) if pred_label == 1 else float(1.0 - probs.item())
            else:
                prob_vec = torch.softmax(base_logits, dim=-1)
                pred_label = int(prob_vec.argmax(dim=-1).item())
                pred_prob = float(prob_vec[0, pred_label].item())

            target = pred_label if target_class is None else int(target_class)
            if optimizer is None:
                return {
                    "edge_masks": {},
                    "prediction": pred_label,
                    "prediction_prob": pred_prob,
                }

            for _ in range(self.epochs):
                optimizer.zero_grad()
                mask_kwargs: Dict[str, Optional[torch.Tensor]] = {}
                reg_term = torch.tensor(0.0, device=self.device)

                mask_fmri, reg_fmri = self._build_masks(fmri_mask_param)
                mask_eeg, reg_eeg = self._build_masks(eeg_mask_param)
                reg_term = reg_term + reg_fmri + reg_eeg
                if mask_fmri is not None:
                    mask_kwargs["edge_mask_fmri"] = mask_fmri
                if mask_eeg is not None:
                    mask_kwargs["edge_mask_eeg"] = mask_eeg

                logits = self.model(
                    pearson_fmri,
                    pearson_eeg,
                    pcorr_fmri,
                    pcorr_eeg,
                    pseudo_fmri,
                    pseudo_eeg,
                    edge_mask_fmri=mask_kwargs.get("edge_mask_fmri"),
                    edge_mask_eeg=mask_kwargs.get("edge_mask_eeg"),
                )

                if self.binary_mode:
                    target_tensor = torch.tensor([float(target)], device=self.device)
                    loss_pred = F.binary_cross_entropy_with_logits(logits.view(-1), target_tensor)
                else:
                    target_tensor = torch.tensor([target], dtype=torch.long, device=self.device)
                    loss_pred = F.nll_loss(F.log_softmax(logits, dim=-1), target_tensor)

                loss = loss_pred + reg_term
                loss.backward()
                optimizer.step()
        finally:
            for param, flag in zip(self.model.parameters(), grad_state):
                param.requires_grad_(flag)

        edge_masks: Dict[str, np.ndarray] = {}
        if fmri_mask_param is not None:
            mask = torch.sigmoid(fmri_mask_param).detach()
            mask = 0.5 * (mask + mask.transpose(0, 1))
            edge_masks["fmri"] = np.asarray(mask.cpu().tolist(), dtype=np.float32)
        if eeg_mask_param is not None:
            mask = torch.sigmoid(eeg_mask_param).detach()
            mask = 0.5 * (mask + mask.transpose(0, 1))
            edge_masks["eeg"] = np.asarray(mask.cpu().tolist(), dtype=np.float32)

        return {
            "edge_masks": edge_masks,
            "prediction": pred_label,
            "prediction_prob": pred_prob,
        }


def _dataset_to_indices(dataset: torch.utils.data.Dataset) -> np.ndarray:
    if isinstance(dataset, torch.utils.data.Subset):
        return np.asarray(dataset.indices)
    return np.arange(len(dataset))


def _default_explain_indices(labels: np.ndarray, per_class: int) -> List[int]:
    selected: List[int] = []
    per_class = max(1, int(per_class))
    unique_labels = sorted(set(int(x) for x in labels.tolist()))
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        if idxs.size == 0:
            continue
        selected.extend(list(idxs[:per_class]))
    if not selected and labels.size > 0:
        selected = [0]
    return selected


def _label_to_condition(label: int) -> str:
    if label == 1:
        return "psilocybin"
    if label == 0:
        return "control"
    return f"class-{label}"


def _prepare_sample_for_explainer(
    dataset: torch.utils.data.Dataset,
    local_idx: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, Optional[torch.Tensor]], int]:
    sample = dataset[local_idx]
    if len(sample) < 7:
        raise ValueError("Dataset sample does not contain all required tensors for explanation.")
    pearson_fmri, pearson_eeg, pcorr_fmri, pcorr_eeg, label_tensor, pseudo_fmri, pseudo_eeg = sample

    def maybe_enable(tensor: torch.Tensor, enabled: bool) -> Optional[torch.Tensor]:
        if not enabled:
            return None
        return tensor.unsqueeze(0).to(device)

    inputs = {
        "pearson_fmri": maybe_enable(pearson_fmri, args.use_fmri),
        "pearson_eeg": maybe_enable(pearson_eeg, args.use_eeg),
        "pcorr_fmri": maybe_enable(pcorr_fmri, args.use_fmri),
        "pcorr_eeg": maybe_enable(pcorr_eeg, args.use_eeg),
        "pseudo_fmri": maybe_enable(pseudo_fmri, args.use_fmri),
        "pseudo_eeg": maybe_enable(pseudo_eeg, args.use_eeg),
    }
    originals = {
        "pcorr_fmri": pcorr_fmri.unsqueeze(0) if args.use_fmri else None,
        "pcorr_eeg": pcorr_eeg.unsqueeze(0) if args.use_eeg else None,
    }
    label = int(label_tensor.item())
    return inputs, originals, label


def _rank_edges(
    signed_saliency: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    if signed_saliency.size == 0:
        return []
    upper = np.triu_indices_from(signed_saliency, k=1)
    magnitude = np.abs(signed_saliency[upper])
    order = np.argsort(magnitude)[::-1]
    top_edges: List[Dict[str, Any]] = []
    limit = min(top_k, order.size)
    for idx in range(limit):
        pos = order[idx]
        i = int(upper[0][pos])
        j = int(upper[1][pos])
        mag = float(magnitude[pos])
        if mag <= 0:
            break
        sign = float(np.sign(signed_saliency[i, j]))
        top_edges.append({"nodes": (i, j), "score": mag, "sign": sign})
    return top_edges


def _plot_top_edges(
    top_edges: List[Dict[str, Any]],
    modality: str,
    sample_tag: str,
    out_path: str,
) -> None:
    if not top_edges:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is not installed; skipping salient edge bar plot.")
        return

    labels = [f"{edge['nodes'][0] + 1}-{edge['nodes'][1] + 1}" for edge in top_edges]
    scores = [edge["score"] for edge in top_edges]
    plt.figure(figsize=(10, max(4, 0.3 * len(top_edges))))
    plt.barh(labels[::-1], scores[::-1], color="#1f77b4")
    plt.xlabel("Saliency (|mask × edge|)")
    plt.title(f"Top {len(top_edges)} salient edges · {modality.upper()} · {sample_tag}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Explain] Saved top-edge plot to {out_path}")


def _plot_edge_heatmap(
    saliency_matrix: np.ndarray,
    modality: str,
    sample_tag: str,
    out_path: str,
) -> None:
    if saliency_matrix.size == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is not installed; skipping saliency heatmap.")
        return

    plt.figure(figsize=(5, 4))
    vmax = max(float(np.max(saliency_matrix)), 1e-6)
    plt.imshow(saliency_matrix, cmap="magma", vmin=0.0, vmax=vmax)
    plt.colorbar(label="|mask × edge weight|")
    plt.title(f"Saliency heatmap · {modality.upper()} · {sample_tag}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Explain] Saved saliency heatmap to {out_path}")


def _fallback_spherical_coords(num_nodes: int, radius: float = 50.0) -> np.ndarray:
    """Generate approximately uniform spherical coordinates as a fallback layout."""
    if num_nodes <= 0:
        return np.empty((0, 3), dtype=np.float32)
    coords = []
    offset = 2.0 / num_nodes
    increment = math.pi * (3.0 - math.sqrt(5.0))
    for idx in range(num_nodes):
        y = ((idx * offset) - 1.0) + (offset / 2.0)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        phi = idx * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        coords.append((x * radius, y * radius, z * radius))
    return np.asarray(coords, dtype=np.float32)


def _load_node_coordinates(path: Optional[str], num_nodes: int, label: str) -> Tuple[np.ndarray, str]:
    """Load ROI/channel coordinates with a resilient fallback."""
    import scipy.io as sio

    if path:
        path = os.path.expanduser(path)
    if path and os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        coords: Optional[np.ndarray] = None
        try:
            if ext in {".npy", ".npz"}:
                data = np.load(path)
                if isinstance(data, np.lib.npyio.NpzFile):
                    for key in ("coords", "arr_0"):
                        if key in data:
                            coords = np.asarray(data[key])
                            break
                else:
                    coords = np.asarray(data)
            elif ext == ".mat":
                try:
                    mat = sio.loadmat(path)
                except Exception as exc:
                    warnings.warn(f"Failed to read MATLAB coords from {path}: {exc}")
                    mat = {}

                def _extract_struct_array(arr: Any) -> Optional[np.ndarray]:
                    if not hasattr(arr, "dtype") or not arr.dtype.names:
                        return None
                    for key in ("coords", "xyz", "coord", "pos", "positions"):
                        if key in arr.dtype.names:
                            candidate = arr[key]
                            try:
                                candidate = np.asarray(candidate).squeeze()
                                while isinstance(candidate, np.ndarray) and candidate.dtype == object:
                                    candidate = np.asarray(candidate.tolist())
                                return candidate
                            except Exception:
                                continue
                    return None

                for key, value in mat.items():
                    if key.startswith("__"):
                        continue
                    extracted = _extract_struct_array(value)
                    if extracted is not None:
                        coords = extracted
                        break
                    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                        if value.ndim >= 2 and value.shape[1] >= 2:
                            coords = value
                            break
            else:
                try:
                    import pandas as pd
                except ImportError:
                    pd = None
                if pd is not None:
                    df = pd.read_csv(path, sep=None, engine="python")
                    for cols in (("x", "y", "z"), ("X", "Y", "Z")):
                        if all(col in df.columns for col in cols):
                            coords = df[list(cols)].to_numpy()
                            break
                    if coords is None and df.shape[1] >= 3:
                        numeric_df = df.select_dtypes(include=[np.number])
                        if numeric_df.shape[1] >= 3:
                            coords = numeric_df.iloc[:, :3].to_numpy()
                        else:
                            coerced = df.iloc[:, -3:].apply(pd.to_numeric, errors="coerce")
                            if coerced.shape[1] >= 3:
                                coords = coerced.to_numpy()
                if coords is None:
                    raw = np.genfromtxt(path, delimiter=None)
                    if isinstance(raw, np.ndarray):
                        if raw.ndim == 1:
                            raw = np.expand_dims(raw, axis=0)
                        if raw.dtype.names:
                            first_three = raw.dtype.names[:3]
                            coords = np.column_stack([raw[name] for name in first_three])
                        elif raw.ndim >= 2 and raw.shape[1] >= 3:
                            coords = raw[:, :3]
        except Exception as exc:
            warnings.warn(f"Failed to load coordinates from {path}: {exc}")
            coords = None

        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            if coords.ndim == 3:
                coords = coords.reshape(coords.shape[0], coords.shape[-1])
            if coords.ndim == 1:
                coords = coords.reshape(-1, 1)
            if coords.shape[1] == 2:
                coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)], axis=1)
            if coords.shape[1] >= 3:
                coords = coords[:num_nodes, :3]
                if coords.shape[0] != num_nodes:
                    warnings.warn(
                        f"{label} coordinate file '{path}' contains {coords.shape[0]} rows; "
                        f"expected {num_nodes}. Using available rows."
                    )
                return coords, path

        shape = getattr(coords, "shape", None)
        warnings.warn(f"{label} coordinate file '{path}' is malformed (shape={shape}). Falling back.")
    warnings.warn(
        f"No valid {label} coordinates found at '{path}'. Using a synthetic spherical layout; "
        f"provide --{label.lower()}_coords_path to override."
    )
    return _fallback_spherical_coords(num_nodes), "synthetic_sphere"


def _scale_node_sizes(
    node_scores: Optional[List[float]], min_size: float = 20.0, max_size: float = 80.0
) -> Optional[np.ndarray]:
    if not node_scores:
        return None
    scores = np.asarray(node_scores, dtype=np.float32)
    scores = np.maximum(scores, 0.0)
    if scores.size == 0:
        return None
    if float(scores.max()) > 0.0:
        scores = scores / float(scores.max())
    sizes = min_size + (max_size - min_size) * scores
    return sizes


def _apply_percentile_threshold(saliency: np.ndarray, percentile: float) -> np.ndarray:
    """Zero-out edges below the given percentile of |saliency|."""
    if saliency.size == 0:
        return saliency
    saliency = np.asarray(saliency, dtype=np.float32)
    upper = np.abs(np.triu(saliency, k=1))
    non_zero = upper[upper > 0]
    if non_zero.size == 0:
        return np.zeros_like(saliency)
    percentile = float(np.clip(percentile, 0.0, 100.0))
    cutoff = float(np.percentile(non_zero, percentile))
    if not np.isfinite(cutoff):
        cutoff = 0.0
    mask = np.abs(saliency) >= max(cutoff, 1e-6)
    trimmed = saliency * mask
    trimmed = np.triu(trimmed, k=1)
    trimmed = trimmed + trimmed.T
    np.fill_diagonal(trimmed, 0.0)
    return trimmed


def _plot_glass_brain(
    saliency_matrix: np.ndarray,
    node_coords: np.ndarray,
    node_sizes: Optional[np.ndarray],
    modality: str,
    sample_tag: str,
    out_path: str,
    percentile: float,
    node_color: Union[Sequence[str], str] = "#1f77b4",
) -> None:
    if saliency_matrix.size == 0 or node_coords.size == 0:
        return
    if node_coords.shape[0] != saliency_matrix.shape[0]:
        warnings.warn(
            f"Coordinate count ({node_coords.shape[0]}) does not match saliency matrix size "
            f"({saliency_matrix.shape[0]}); skipping glass brain."
        )
        return
    try:
        import matplotlib.pyplot as plt
        from nilearn import plotting
    except ImportError:
        warnings.warn("nilearn/matplotlib not installed; skipping glass brain plots.")
        return

    filtered = _apply_percentile_threshold(saliency_matrix, percentile)
    vmax = float(np.max(np.abs(filtered)))
    if vmax <= 0:
        warnings.warn("Saliency matrix is empty after thresholding; skipping glass brain.")
        return
    if node_sizes is not None and len(node_sizes) != filtered.shape[0]:
        warnings.warn("Node-size vector length mismatch; ignoring node sizes for glass brain plot.")
        node_sizes = None

    fig = plt.figure(figsize=(8, 6))
    display = plotting.plot_connectome(
        adjacency_matrix=filtered,
        node_coords=node_coords,
        node_size=node_sizes,
        node_color=node_color,
        edge_cmap="coolwarm",
        edge_vmin=-vmax,
        edge_vmax=vmax,
        edge_kwargs={"lw": 1.2},
        display_mode="lyrz",
        colorbar=True,
        figure=fig,
        title=f"{modality.upper()} · {sample_tag}",
    )
    display.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [Explain] Saved glass brain to {out_path}")


def _build_fused_connectome(
    modality_payload: Dict[str, Dict[str, Any]],
    coords_map: Dict[str, np.ndarray],
    node_size_map: Dict[str, Optional[np.ndarray]],
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]]:
    """Combine modality-specific saliency maps into a block-diagonal fused graph."""
    order = ["fmri", "eeg"]
    saliency_blocks: List[np.ndarray] = []
    coord_blocks: List[np.ndarray] = []
    size_blocks: List[np.ndarray] = []
    node_colors: List[str] = []
    color_map = {"fmri": "#1f77b4", "eeg": "#e15759"}

    for modality in order:
        payload = modality_payload.get(modality)
        coords = coords_map.get(modality)
        if payload is None or coords is None:
            continue
        saliency = payload.get("saliency_signed")
        if saliency is None or saliency.size == 0:
            continue
        saliency_blocks.append(saliency)
        coord_blocks.append(coords)
        sizes = node_size_map.get(modality)
        if sizes is None:
            sizes = np.full(coords.shape[0], 40.0, dtype=np.float32)
        size_blocks.append(sizes)
        node_colors.extend([color_map.get(modality, "#4d4d4d")] * coords.shape[0])

    if not saliency_blocks:
        return None

    total_nodes = sum(block.shape[0] for block in saliency_blocks)
    fused = np.zeros((total_nodes, total_nodes), dtype=np.float32)
    cursor = 0
    fused_sizes: Optional[np.ndarray] = np.concatenate(size_blocks) if size_blocks else None
    for block in saliency_blocks:
        n = block.shape[0]
        fused[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n

    fused_coords = np.vstack(coord_blocks)
    return fused, fused_coords, fused_sizes, node_colors


def _plot_shared_vs_specific_edges(
    sample_summaries: List[Dict[str, Any]],
    modality: str,
    split_name: str,
    out_path: str,
    top_k: int,
) -> None:
    if len(sample_summaries) < 2:
        return
    class_edge_scores: Dict[str, Dict[Tuple[int, int], float]] = {}
    for summary in sample_summaries:
        class_name = summary["label_name"]
        raw_modality = summary.get("top_edges_raw", {}).get(modality, {})
        edges = raw_modality.get("top_edges")
        if edges is None:
            edges = summary.get("top_edges", {}).get(modality, [])
            edges = [
                {
                    "nodes": (edge["nodes"][0] - 1, edge["nodes"][1] - 1),
                    "score": edge["score"],
                }
                for edge in edges
            ]
        if not edges:
            continue
        edge_map = class_edge_scores.setdefault(class_name, {})
        for edge in edges[:top_k]:
            key = tuple(edge["nodes"])
            edge_map[key] = max(edge_map.get(key, 0.0), float(edge["score"]))
    if len(class_edge_scores) < 2:
        return
    union_edges = sorted(set().union(*[scores.keys() for scores in class_edge_scores.values()]))
    if not union_edges:
        return
    combined = []
    for edge in union_edges:
        total = sum(scores.get(edge, 0.0) for scores in class_edge_scores.values())
        combined.append((edge, total))
    combined.sort(key=lambda item: item[1], reverse=True)
    combined = combined[:top_k]
    if not combined:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is not installed; skipping shared-edge plot.")
        return

    edge_labels = [f"{edge[0] + 1}-{edge[1] + 1}" for edge, _ in combined]
    class_names = sorted(class_edge_scores.keys())
    fig, ax = plt.subplots(figsize=(min(14, 6 + 0.3 * len(edge_labels)), 5))
    bottoms = np.zeros(len(edge_labels))
    for cls in class_names:
        contributions = [class_edge_scores[cls].get(edge, 0.0) for edge, _ in combined]
        ax.bar(edge_labels, contributions, 0.8, bottom=bottoms, label=cls.title())
        bottoms += np.array(contributions)

    ax.set_ylabel("Saliency score")
    ax.set_title(f"Shared vs modality-specific important edges · {modality.upper()} · {split_name}")
    ax.legend()
    import matplotlib.pyplot as plt
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Explain] Saved shared-edge plot to {out_path}")


def _write_explanation_payload(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"  [Explain] Saved JSON summary to {path}")


def maybe_run_gnn_explanations(
    model: nn.Module,
    split_datasets: Dict[str, torch.utils.data.Dataset],
    args: argparse.Namespace,
    device: torch.device,
    binary_mode: bool,
    labels_np: np.ndarray,
    fold_tag: Optional[str] = None,
) -> None:
    """Orchestrate per-sample GNN explanation generation.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    from brainxdl.mm.models.baselines import LateFusionMLP
    from brainxdl.mm.models.brainmgf import MMBrainXDL

    if not getattr(args, "gnn_explain", False):
        return
    if not isinstance(model, (MMBrainXDL, LateFusionMLP)):
        warnings.warn("GNN explanations currently support MM-BrainXDL style fusion models; skipping.")
        return
    split_name = getattr(args, "explain_split", "test")
    dataset = split_datasets.get(split_name)
    if dataset is None:
        warnings.warn(f"Split '{split_name}' is unavailable; cannot generate explanations.")
        return

    subset_indices = _dataset_to_indices(dataset)
    if subset_indices.size == 0:
        warnings.warn(f"Split '{split_name}' is empty; skipping explanations.")
        return

    split_labels = labels_np[subset_indices]
    if split_labels.size == 0:
        warnings.warn(f"Labels missing for split '{split_name}'; skipping explanations.")
        return

    explicit_indices = getattr(args, "explain_indices", None)
    if explicit_indices:
        local_indices = []
        for idx in explicit_indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(dataset):
                local_indices.append(idx_int)
            else:
                warnings.warn(
                    f"Requested explain index {idx_int} is outside the bounds of split '{split_name}'. Skipping."
                )
    else:
        local_indices = _default_explain_indices(split_labels, getattr(args, "explain_per_class", 1))

    max_samples = getattr(args, "explain_max_samples", len(local_indices) or len(dataset))
    max_samples = max(1, int(max_samples))
    local_indices = local_indices[:max_samples]
    if not local_indices:
        warnings.warn("No valid indices selected for explanation; skipping.")
        return

    explain_dir = getattr(args, "explain_dir", "plots/explanations")
    if not os.path.isabs(explain_dir):
        explain_dir = os.path.join(os.getcwd(), explain_dir)
    run_slug = compute_run_slug(args)
    run_dir = os.path.join(explain_dir, run_slug)
    if fold_tag:
        run_dir = os.path.join(run_dir, fold_tag)
    os.makedirs(run_dir, exist_ok=True)

    explainer = DenseGNNExplainer(
        model=model,
        device=device,
        binary_mode=binary_mode,
        use_fmri=args.use_fmri,
        use_eeg=args.use_eeg,
        lr=getattr(args, "explain_lr", 0.1),
        epochs=getattr(args, "explain_steps", 200),
        size_coeff=getattr(args, "explain_size_coeff", 0.01),
        entropy_coeff=getattr(args, "explain_entropy_coeff", 5e-4),
    )

    subset_indices = np.asarray(subset_indices)
    sample_summaries: List[Dict[str, Any]] = []
    top_k_edges = getattr(args, "explain_topk_edges", 20)
    coord_cache: Dict[str, Tuple[np.ndarray, str]] = {}

    for local_idx in local_indices:
        global_idx = int(subset_indices[local_idx])
        inputs, originals, label = _prepare_sample_for_explainer(dataset, local_idx, device, args)
        label_name = _label_to_condition(label)
        explanation = explainer.explain(
            inputs["pearson_fmri"],
            inputs["pearson_eeg"],
            inputs["pcorr_fmri"],
            inputs["pcorr_eeg"],
            inputs["pseudo_fmri"],
            inputs["pseudo_eeg"],
        )
        adjacency_maps: Dict[str, np.ndarray] = {}
        if args.use_fmri and originals["pcorr_fmri"] is not None:
            adj = knn_thresholding(originals["pcorr_fmri"].to(device), k=getattr(args, "k_fmri", 40))
            adjacency_maps["fmri"] = np.asarray(adj.squeeze(0).detach().cpu().tolist(), dtype=np.float32)
        if args.use_eeg and originals["pcorr_eeg"] is not None:
            adj = knn_thresholding(originals["pcorr_eeg"].to(device), k=getattr(args, "k_eeg", 40))
            adjacency_maps["eeg"] = np.asarray(adj.squeeze(0).detach().cpu().tolist(), dtype=np.float32)

        modality_payload: Dict[str, Dict[str, Any]] = {}
        for modality, mask in explanation["edge_masks"].items():
            adjacency = adjacency_maps.get(modality)
            if adjacency is None:
                continue
            signed_saliency = mask * adjacency
            np.fill_diagonal(signed_saliency, 0.0)
            top_edges = _rank_edges(signed_saliency, top_k_edges)
            node_scores = (np.abs(signed_saliency).sum(axis=0) + np.abs(signed_saliency).sum(axis=1)).tolist()
            modality_payload[modality] = {
                "saliency_matrix": np.abs(signed_saliency),
                "saliency_signed": signed_saliency,
                "top_edges": top_edges,
                "node_scores": node_scores,
            }
            if modality not in coord_cache:
                coord_path = getattr(
                    args,
                    "roi_coords_path" if modality == "fmri" else "eeg_coords_path",
                    None,
                )
                coord_label = "ROI" if modality == "fmri" else "EEG"
                coords, source = _load_node_coordinates(coord_path, adjacency.shape[0], coord_label)
                coord_cache[modality] = (coords, source)

        pred_label = int(explanation["prediction"])
        pred_prob = float(explanation["prediction_prob"])
        pred_name = _label_to_condition(pred_label)
        split_slug = slugify(f"{split_name}")
        sample_tag = slugify(
            f"{split_slug}_idx-{global_idx}_label-{label_name}_pred-{pred_name}"
        )
        if fold_tag:
            sample_tag = f"{slugify(fold_tag)}_{sample_tag}"

        payload = {
            "split": split_name,
            "fold": fold_tag,
            "local_index": int(local_idx),
            "global_index": global_idx,
            "label": label,
            "label_name": label_name,
            "prediction": pred_label,
            "prediction_name": pred_name,
            "prediction_prob": pred_prob,
            "top_edges": {},
            "node_scores": {},
        }

        for modality, summary in modality_payload.items():
            top_edges = [
                {
                    "nodes": [int(edge["nodes"][0]) + 1, int(edge["nodes"][1]) + 1],
                    "score": float(edge["score"]),
                    "sign": float(edge["sign"]),
                }
                for edge in summary["top_edges"]
            ]
            payload["top_edges"][modality] = top_edges
            payload["node_scores"][modality] = summary["node_scores"]

            top_path = os.path.join(run_dir, f"{sample_tag}_{modality}_top{len(top_edges)}_edges.png")
            heatmap_path = os.path.join(run_dir, f"{sample_tag}_{modality}_saliency.png")
            _plot_top_edges(summary["top_edges"], modality, sample_tag, top_path)
            _plot_edge_heatmap(summary["saliency_matrix"], modality, sample_tag, heatmap_path)

            saliency_path = os.path.join(run_dir, f"{sample_tag}_{modality}_saliency.npz")
            np.savez(
                saliency_path,
                saliency=summary["saliency_matrix"].astype(np.float32),
                saliency_signed=summary["saliency_signed"].astype(np.float32),
            )

            if getattr(args, "explain_glass_brain", True):
                coords, _ = coord_cache.get(modality, (np.empty((0, 3)), ""))
                node_sizes = _scale_node_sizes(summary["node_scores"])
                glass_path = os.path.join(run_dir, f"{sample_tag}_{modality}_glass.png")
                _plot_glass_brain(
                    summary["saliency_signed"],
                    coords,
                    node_sizes,
                    modality,
                    sample_tag,
                    glass_path,
                    getattr(args, "glass_edge_percentile", 85.0),
                    node_color="#1f77b4" if modality == "fmri" else "#e15759",
                )

        if getattr(args, "explain_glass_brain", True) and modality_payload:
            coords_map = {k: v[0] for k, v in coord_cache.items()}
            node_size_map = {
                modality: _scale_node_sizes(summary.get("node_scores"))
                for modality, summary in modality_payload.items()
            }
            fused = _build_fused_connectome(modality_payload, coords_map, node_size_map)
            if fused is not None:
                fused_adj, fused_coords, fused_sizes, fused_colors = fused
                fused_path = os.path.join(run_dir, f"{sample_tag}_fused_glass.png")
                _plot_glass_brain(
                    fused_adj,
                    fused_coords,
                    fused_sizes,
                    "fused",
                    sample_tag,
                    fused_path,
                    getattr(args, "glass_edge_percentile", 85.0),
                    node_color=fused_colors,
                )

        payload["glass_brain"] = {
            "enabled": bool(getattr(args, "explain_glass_brain", True)),
            "edge_percentile": float(getattr(args, "glass_edge_percentile", 85.0)),
            "coord_sources": {modality: source for modality, (_, source) in coord_cache.items()},
        }

        json_path = os.path.join(run_dir, f"{sample_tag}_explanation.json")
        _write_explanation_payload(json_path, payload)
        payload["top_edges_raw"] = modality_payload
        sample_summaries.append(payload)

    for modality in ("fmri", "eeg"):
        out_path = os.path.join(run_dir, f"{slugify(split_name)}_{modality}_shared_edges.png")
        _plot_shared_vs_specific_edges(sample_summaries, modality, split_name, out_path, top_k_edges)
