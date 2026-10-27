"""Latent-space collection and manifold visualisation utilities.

Source: extracted from legacy_code/2-MM/main.py.
"""

import argparse
import inspect
import json
import os
import re
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from brainxdl.mm.utils.io import get_task_list


def collect_latent_embeddings(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate over a loader and gather latent encodings alongside labels.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    from brainxdl.mm.models.brainmgf import forward_model

    latents: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            _, lbls, latent_batch, _ = forward_model(model, batch, device, args, return_latent=True)
            if latent_batch is None:
                warnings.warn("Model did not return latent embeddings; aborting manifold generation.")
                return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
            latent_cpu = latent_batch.detach().cpu()
            label_cpu = lbls.detach().cpu()
            latents.append(np.asarray(latent_cpu.tolist(), dtype=np.float32))
            labels.append(np.asarray(label_cpu.tolist(), dtype=np.int64))

    if not latents:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)


def ensure_shared_manifold_hparams(latent_dir: str, config: Dict[str, Any]) -> None:
    """Persist and validate manifold hyper-parameters to keep runs comparable.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    os.makedirs(latent_dir, exist_ok=True)
    cfg_path = os.path.join(latent_dir, "manifold_config.json")

    def canonicalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
        base = json.loads(json.dumps(config))

        def merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    merge(dst[key], value)
                else:
                    dst[key] = value

        merge(base, cfg)
        return base

    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_norm = canonicalize(existing)
        config_norm = canonicalize(config)
        if existing_norm != config_norm:
            raise ValueError(
                "Existing manifold_config.json uses different hyper-parameters. "
                "Please align the CLI flags or remove the file to regenerate it."
            )
        if existing != existing_norm:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(existing_norm, f, indent=2, sort_keys=True)
    else:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)


def compute_tsne(latents: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Run t-SNE with safety checks on perplexity for small sample sizes.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    from sklearn.manifold import TSNE

    n_samples = latents.shape[0]
    if n_samples < 2:
        return np.zeros((n_samples, params["n_components"]), dtype=np.float32)

    max_perplexity = max(1.0, float(n_samples - 1))
    recommended_perplexity = max(1.0, min(max_perplexity - 1e-3, float(max(1, (n_samples - 1) // 3)) or 1.0))
    recommended_perplexity = max(recommended_perplexity, 5.0 if n_samples > 5 else 1.0)
    perplexity = float(params["perplexity"])
    if perplexity >= max_perplexity:
        warnings.warn(
            f"Requested perplexity={perplexity:.2f} exceeds n_samples-1={n_samples-1}. "
            f"Clamping to {max_perplexity - 1e-3:.2f} for stability."
        )
        perplexity = max_perplexity - 1e-3
    if perplexity > recommended_perplexity:
        warnings.warn(
            f"Perplexity {perplexity:.2f} is high for {n_samples} samples. "
            f"Using {recommended_perplexity:.2f} to accelerate convergence."
        )
        perplexity = recommended_perplexity
    min_perplexity = 5.0 if n_samples > 5 else 1.0
    perplexity = max(perplexity, min_perplexity)

    tsne_kwargs = {
        "n_components": params["n_components"],
        "perplexity": perplexity,
        "learning_rate": params["learning_rate"],
        "init": params["init"],
        "metric": params["metric"],
        "random_state": params["random_state"],
    }
    tsne_signature = inspect.signature(TSNE)
    if "max_iter" in tsne_signature.parameters:
        tsne_kwargs["max_iter"] = params["n_iter"]
    else:
        tsne_kwargs["n_iter"] = params["n_iter"]
    if "verbose" in tsne_signature.parameters and params.get("verbose", 0):
        tsne_kwargs["verbose"] = params["verbose"]
    if "square_distances" in tsne_signature.parameters:
        tsne_kwargs["square_distances"] = True
    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(latents)


def compute_umap(latents: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Project latent vectors with UMAP using shared hyper-parameters.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    import umap  # type: ignore

    n_samples = latents.shape[0]
    if n_samples < 2:
        return np.zeros((n_samples, params["n_components"]), dtype=np.float32)

    n_neighbors = int(min(max(2, params["n_neighbors"]), max(2, n_samples - 1)))
    if n_neighbors != params["n_neighbors"]:
        warnings.warn(
            f"Requested n_neighbors={params['n_neighbors']} exceeds dataset size. "
            f"Using n_neighbors={n_neighbors} instead."
        )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=params["min_dist"],
        n_components=params["n_components"],
        metric=params["metric"],
        random_state=params["random_state"],
    )
    return reducer.fit_transform(latents)


def plot_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: Dict[int, str],
    title: str,
    out_path: str,
) -> None:
    """Render and save a 2D scatter plot for manifold coordinates.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    unique_labels = sorted(int(lbl) for lbl in np.unique(labels.astype(int)))
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 2))

    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=30,
            alpha=0.85,
            color=cmap(idx),
            label=class_names.get(label, f"class {label}"),
            edgecolors="none",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_latent_manifolds(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
    device: torch.device,
    split_name: str,
) -> None:
    """Compute and plot t-SNE / UMAP projections of latent embeddings.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    batch_size = getattr(args, "latent_batch_size", None) or args.batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    latents, labels = collect_latent_embeddings(loader, model, device, args)
    if latents.size == 0:
        warnings.warn(f"No latent embeddings available for split '{split_name}'. Skipping manifold plots.")
        return

    scaler = StandardScaler()
    latents_std = scaler.fit_transform(latents) if latents.shape[0] > 1 else latents

    latent_seed = getattr(args, "latent_seed", None)
    if latent_seed is None:
        latent_seed = args.seed

    manifold_config = {
        "latent_seed": int(latent_seed),
        "n_components": int(getattr(args, "latent_components", 2)),
        "tsne": {
            "perplexity": float(getattr(args, "latent_tsne_perplexity", 30.0)),
            "learning_rate": float(getattr(args, "latent_tsne_lr", 200.0)),
            "n_iter": int(getattr(args, "latent_tsne_iter", 1000)),
            "metric": getattr(args, "latent_tsne_metric", "euclidean"),
            "init": getattr(args, "latent_tsne_init", "random"),
            "verbose": int(getattr(args, "latent_tsne_verbose", 0) or 0),
        },
        "umap": {
            "n_neighbors": int(getattr(args, "latent_umap_neighbors", 15)),
            "min_dist": float(getattr(args, "latent_umap_min_dist", 0.1)),
            "metric": getattr(args, "latent_umap_metric", "euclidean"),
        },
    }
    ensure_shared_manifold_hparams(args.latent_dir, manifold_config)

    tsne_params = {
        "n_components": manifold_config["n_components"],
        "perplexity": manifold_config["tsne"]["perplexity"],
        "learning_rate": manifold_config["tsne"]["learning_rate"],
        "n_iter": manifold_config["tsne"]["n_iter"],
        "metric": manifold_config["tsne"]["metric"],
        "random_state": latent_seed,
        "init": manifold_config["tsne"]["init"],
        "verbose": manifold_config["tsne"].get("verbose", 0),
    }
    umap_params = {
        "n_components": manifold_config["n_components"],
        "n_neighbors": manifold_config["umap"]["n_neighbors"],
        "min_dist": manifold_config["umap"]["min_dist"],
        "metric": manifold_config["umap"]["metric"],
        "random_state": latent_seed,
    }

    methods = [m.lower() for m in getattr(args, "latent_methods", ["tsne", "umap"])]
    class_name_defaults = {0: "Control", 1: "Psilocybin"}
    present_labels = sorted(int(lbl) for lbl in np.unique(labels.astype(int)))
    class_names = class_name_defaults.copy()
    for label in present_labels:
        class_names.setdefault(label, f"class {label}")

    tasks_list = get_task_list(args)
    if tasks_list:
        task_descriptor = "-".join(sorted(set(tasks_list)))
    else:
        task_descriptor = str(getattr(args, "task", "all"))
    task_slug = re.sub(r"[^A-Za-z0-9]+", "-", task_descriptor.strip().lower()).strip("-") or "all"

    base_title = f"{args.modality.upper()} · {task_slug.upper()} · {split_name.upper()}"
    base_filename = f"{args.modality.lower()}_{task_slug}_{split_name.lower()}"

    if "tsne" in methods:
        print(
            f"[Latents] Running t-SNE on split '{split_name}' "
            f"({latents_std.shape[0]} samples, perplexity={tsne_params['perplexity']:.2f}, "
            f"max_iter={tsne_params['n_iter']}, verbose={tsne_params.get('verbose', 0)})"
        )
        coords = compute_tsne(latents_std, tsne_params)
        plot_path = os.path.join(args.latent_dir, f"{base_filename}_tsne.png")
        plot_scatter(coords, labels, class_names, f"{base_title} · t-SNE", plot_path)
        np.savez_compressed(
            os.path.join(args.latent_dir, f"{base_filename}_tsne.npz"),
            coords=coords,
            labels=labels,
            method="tsne",
            modality=args.modality,
            split=split_name,
        )
        print(f"[Latents] Saved t-SNE outputs to {plot_path}")

    if "umap" in methods:
        print(
            f"[Latents] Running UMAP on split '{split_name}' "
            f"({latents_std.shape[0]} samples, n_neighbors={umap_params['n_neighbors']}, "
            f"min_dist={umap_params['min_dist']:.2f})"
        )
        coords = compute_umap(latents_std, umap_params)
        plot_path = os.path.join(args.latent_dir, f"{base_filename}_umap.png")
        plot_scatter(coords, labels, class_names, f"{base_title} · UMAP", plot_path)
        np.savez_compressed(
            os.path.join(args.latent_dir, f"{base_filename}_umap.npz"),
            coords=coords,
            labels=labels,
            method="umap",
            modality=args.modality,
            split=split_name,
        )
        print(f"[Latents] Saved UMAP outputs to {plot_path}")
