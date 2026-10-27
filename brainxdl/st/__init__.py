"""brainxdl.st — ST-BrainXDL: dynamic graph representation learning for brain disorders.

Publication: IJCAI 2024 (Brain DSVB).

Submodules
----------
model        : ST-BrainXDL model (GRU-GCN + VAE).
dataset      : STGraphDataset, padseq, load_fold_data.
trainer      : init_checkpoint, train, validate.
preprocessing: Step-1 pipeline (LDW covariance → adjacency).
data_prep    : Step-2 pipeline (cross-validation folds → PyG graphs).
visualize    : plot_loss_curves, analyze_checkpoint.
"""

import importlib

__all__ = [
    "STBrainXDL",
    "VGRNN",
    "STGraphDataset",
    "load_fold_data",
    "padseq",
    "init_checkpoint",
    "train",
    "validate",
]


def __getattr__(name):
    if name in {"STBrainXDL", "VGRNN"}:
        from brainxdl.st.model import STBrainXDL, VGRNN
        return STBrainXDL if name == "STBrainXDL" else VGRNN
    if name in {"STGraphDataset", "load_fold_data", "padseq"}:
        dataset = importlib.import_module("brainxdl.st.dataset")
        return getattr(dataset, name)
    if name in {"init_checkpoint", "train", "validate"}:
        trainer = importlib.import_module("brainxdl.st.trainer")
        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
