"""BrainXDL multimodal (MM) package: fMRI + EEG classification with MM-BrainXDL."""

from brainxdl.mm.models.brainmgf import BrainMGF, MMBrainXDL, build_model, forward_model
from brainxdl.mm.training.trainer import run_kfold_cv, evaluate_metrics

__all__ = [
    "MMBrainXDL",
    "BrainMGF",
    "build_model",
    "forward_model",
    "prepare_data",
    "run_kfold_cv",
    "evaluate_metrics",
]


def __getattr__(name):
    if name == "prepare_data":
        from brainxdl.mm.data.dataset import prepare_data

        return prepare_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
