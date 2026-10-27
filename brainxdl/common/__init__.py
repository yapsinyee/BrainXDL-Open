"""brainxdl.common — shared utilities for all BrainXDL modules.

Import from the submodules directly for fine-grained control, or use
the convenience re-exports below.

Submodules
----------
seed          : seed_everything
device        : get_device, pick_device
logging       : METRIC_DIRECTIONS, metric_is_better, format_metric_value
checkpointing : save_obj, load_obj, check_path, to_cpu_state_dict,
                save_checkpoint, load_checkpoint, load_rng_state
metrics       : compute_probs_and_preds, compute_classification_metrics
connectivity  : stabilise_timeseries, compute_correlation,
                compute_partial_correlation, generate_connectivity_matrix,
                cov2corr, knn_thresholding
layers        : get_activation, dense_vary, GCN_vary, recurrent_cell
training      : EarlyStopping, train_one_epoch, evaluate, evaluate_full
"""

import importlib

from brainxdl.common.seed import seed_everything
from brainxdl.common.device import get_device, pick_device
from brainxdl.common.logging import (
    METRIC_DIRECTIONS,
    metric_is_better,
    format_metric_value,
)
from brainxdl.common.checkpointing import (
    save_obj,
    load_obj,
    check_path,
    to_cpu_state_dict,
    save_checkpoint,
    load_checkpoint,
    load_rng_state,
)
from brainxdl.common.metrics import (
    compute_probs_and_preds,
    compute_classification_metrics,
)
from brainxdl.common.connectivity import (
    stabilise_timeseries,
    compute_correlation,
    compute_partial_correlation,
    generate_connectivity_matrix,
    cov2corr,
    knn_thresholding,
)

__all__ = [
    # seed
    "seed_everything",
    # device
    "get_device",
    "pick_device",
    # logging
    "METRIC_DIRECTIONS",
    "metric_is_better",
    "format_metric_value",
    # checkpointing
    "save_obj",
    "load_obj",
    "check_path",
    "to_cpu_state_dict",
    "save_checkpoint",
    "load_checkpoint",
    "load_rng_state",
    # metrics
    "compute_probs_and_preds",
    "compute_classification_metrics",
    # connectivity
    "stabilise_timeseries",
    "compute_correlation",
    "compute_partial_correlation",
    "generate_connectivity_matrix",
    "cov2corr",
    "knn_thresholding",
    # layers
    "get_activation",
    "dense_vary",
    "GCN_vary",
    "recurrent_cell",
    # training
    "EarlyStopping",
    "train_one_epoch",
    "evaluate",
    "evaluate_full",
]


def __getattr__(name):
    if name in {"get_activation", "dense_vary", "GCN_vary", "recurrent_cell"}:
        layers = importlib.import_module("brainxdl.common.layers")
        return getattr(layers, name)
    if name in {"EarlyStopping", "train_one_epoch", "evaluate", "evaluate_full"}:
        training = importlib.import_module("brainxdl.common.training")
        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
