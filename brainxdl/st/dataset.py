"""ST-BrainXDL: Dataset classes and fold-data loading for ST-BrainXDL training.

Contains:
- load_fold_data: load one cross-validation fold from preprocessed .pkl files.
- STGraphDataset: PyTorch Dataset wrapping subject-level graph sequences.
- padseq: DataLoader collate function that batches sequences across time steps.

Sources: extracted from legacy_code/1-ST/train.py
  (load_data → load_fold_data, myDataset → STGraphDataset, padseq).
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData


def load_fold_data(
    outer_loop: int,
    inner_loop: int,
    data_dir: str = "./data/folds_data",
):
    """Load train / val / test graph sequences for one cross-validation fold.

    The files are produced by brainxdl.st.data_prep (step 2 of preprocessing).

    Args:
        outer_loop: Outer fold index (1–5).
        inner_loop: Inner fold index (1–5).
        data_dir:   Directory containing graphs_outerX_innerY.pkl files.
                    Defaults to './data/folds_data' (relative to cwd, matching
                    the legacy script convention).

    Returns:
        (train_graphs, test_graphs, val_graphs): three np.ndarray objects
        of shape (subjects, windows) containing PyG Data objects.

    Raises:
        FileNotFoundError: If the expected fold file does not exist.
    """
    path = os.path.join(data_dir, f"graphs_outer{outer_loop}_inner{inner_loop}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fold file not found: {path}\n"
            "Run brainxdl.st.data_prep (step2_prepare_data) first."
        )
    with open(path, "rb") as f:
        loaded = torch.load(f, weights_only=False)
    return loaded["train_graphs"], loaded["test_graphs"], loaded["val_graphs"]


class STGraphDataset(Dataset):
    """PyTorch Dataset for ST-BrainXDL graph sequences.

    Each item is a 1-D array of PyG Data objects representing the temporal
    window sequence for one subject.

    Args:
        data: np.ndarray of shape (subjects, windows) containing PyG Data.

    Note: renamed from 'myDataset' in legacy code for clarity.
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        if data.size > 0 and data[0, 0] is not None:
            self.num_node_features = data[0, 0].num_node_features
            self.num_classes       = data[0, 0].num_classes
            self.num_nodes         = data[0, 0].num_nodes
        else:
            self.num_node_features = 0
            self.num_classes       = 0
            self.num_nodes         = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def padseq(batch):
    """Collate function that re-organises a batch of sequences by time step.

    The DataLoader provides a list of per-subject sequences.  This function
    transposes that into a list of per-time-step Batch objects, which is the
    format expected by ST-BrainXDL.forward().

    Args:
        batch: List of np.ndarray, each of shape (windows,) containing PyG
               Data objects for one subject.

    Returns:
        List of torch_geometric.data.Batch, one per time step.
    """
    batch = np.asarray(batch).T  # (windows, subjects)
    if batch.size > 0 and isinstance(batch[0, 0], BaseData):
        return [Batch.from_data_list(step.tolist()) for step in batch]
    return []
