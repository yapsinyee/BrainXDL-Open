"""Reusable workflow orchestration for BrainXDL.

The pipelines package is reserved for end-to-end workflows that compose ST, MM,
XAI, and common utilities without owning scientific model logic. Module-specific
training steps stay in ``brainxdl.st`` and ``brainxdl.mm``; shared explainer and
latent-space routines stay in ``brainxdl.xai``.

Planned pipeline modules:
- ``st_workflow``: ST data loading, fold selection, model setup, checkpoint
  setup, and training orchestration currently duplicated in entry-point scripts.
- ``mm_workflow``: MM cache/load, split construction, model setup, train/eval,
  checkpointing, and optional XAI orchestration.
- ``evaluation``: checkpoint-to-metrics workflows shared by CLI entry points.
- ``artifacts``: consistent run directory, checkpoint, metrics, and plot
  manifest handling across modules.

Keep this package free of CLI parsing and one-off experiment analysis.
"""

from brainxdl.pipelines.st_workflow import (
    STWorkflowResult,
    infer_model_dimensions,
    run_st_training_workflow,
)
from brainxdl.pipelines.mm_workflow import print_args, run_mm_training_workflow
from brainxdl.pipelines.mm_evaluation import evaluate_mm_checkpoint

__all__ = [
    "STWorkflowResult",
    "evaluate_mm_checkpoint",
    "infer_model_dimensions",
    "print_args",
    "run_mm_training_workflow",
    "run_st_training_workflow",
]
