"""MM models package."""
from brainxdl.mm.models.brainmgf import (
    BrainMGF,
    GraphConvModel,
    MMBrainXDL,
    MoEFusion,
    SelfPreservingGraphConvModel,
    SignSeparatedFirstBlockGraphConvModel,
    build_model,
    forward_model,
)
from brainxdl.mm.models.baselines import (
    FmriGCN,
    LateFusionMLP,
)
from brainxdl.mm.models.dynamic_brainmgf import DynamicBrainMGF, DynamicMMBrainXDL

__all__ = [
    "MMBrainXDL",
    "BrainMGF",
    "GraphConvModel",
    "SelfPreservingGraphConvModel",
    "SignSeparatedFirstBlockGraphConvModel",
    "MoEFusion",
    "build_model",
    "forward_model",
    "FmriGCN",
    "LateFusionMLP",
    "DynamicMMBrainXDL",
    "DynamicBrainMGF",
]
