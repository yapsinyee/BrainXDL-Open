"""MM-specific I/O and path helpers.

Source: extracted from legacy_code/2-MM/main.py.
"""

import argparse
import os
import re
from typing import Any, List, Optional


def slugify(text: Any) -> str:
    """Convert arbitrary text into a filesystem-friendly slug."""
    if text is None:
        return "na"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(text).strip()).strip("-").lower()
    return slug or "na"


def ensure_unique_path(path: str) -> str:
    """Return a path that does not already exist by appending a numeric suffix."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def resolve_checkpoint_dir(
    args: argparse.Namespace, base_metrics_dir: Optional[str] = None
) -> str:
    """Resolve the directory where checkpoints should be stored.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if checkpoint_dir:
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.join(script_dir, checkpoint_dir)
    else:
        root_dir = base_metrics_dir
        if root_dir is None:
            root_dir = getattr(args, "metrics_dir", None) or getattr(args, "plot_dir", "plots")
            if not os.path.isabs(root_dir):
                root_dir = os.path.join(script_dir, root_dir)
        checkpoint_dir = os.path.join(root_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_task_list(args: argparse.Namespace) -> List[str]:
    """Return a flat list of task names from args.tasks or args.task."""
    if getattr(args, "tasks", None):
        return [t.strip().lower() for t in args.tasks if t.strip()]
    if getattr(args, "task", None):
        return [t.strip().lower() for t in str(args.task).split(",") if t.strip()]
    return []


def compute_run_slug(args: argparse.Namespace) -> str:
    """Create a slug identifying the current training configuration."""
    model_slug = slugify(getattr(args, "model", "model"))
    modality_slug = slugify(getattr(args, "modality", "modality"))
    loss_slug = slugify(getattr(args, "loss_type", "loss"))
    graph_slug = slugify(getattr(args, "graph_mode", "static"))
    tasks_list = get_task_list(args)
    if tasks_list:
        task_descriptor = "-".join(sorted(set(tasks_list)))
    else:
        task_descriptor = str(getattr(args, "task", "all"))
    task_slug = slugify(task_descriptor)
    return f"{model_slug}_{task_slug}_{modality_slug}_graph-{graph_slug}_loss-{loss_slug}"
