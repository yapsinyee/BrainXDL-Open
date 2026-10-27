"""ST-BrainXDL: Step 2 preprocessing — cross-validation folds and PyG graph construction.

Converts the LDW correlation data produced by preprocessing.py into
PyTorch Geometric Data objects partitioned into cross-validation folds.

Run as a standalone script (python -m brainxdl.st.data_prep) or import
individual functions.

Scientific behaviour is unchanged from legacy_code/1-ST/step2_prepare_data.py.

Changes from legacy:
- Bare np.random.seed / torch.manual_seed replaced with seed_everything.
- Data paths are parameters; defaults match legacy conventions.
- __main__ block uses argparse.
"""

import os
import pickle

import dill
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from tqdm import tqdm

from brainxdl.common.seed import seed_everything

SEED = 42


def extend_labels(data_idx: np.ndarray, n_win: int) -> np.ndarray:
    """Map subject indices to the corresponding window indices.

    If subject i has n_win windows, windows i*n_win … (i+1)*n_win-1 map to it.

    Args:
        data_idx: Subject-level indices.
        n_win:    Number of windows per subject.

    Returns:
        Window-level indices.
    """
    ex = []
    for idx in data_idx:
        st = idx * n_win
        ex.extend(np.linspace(st, st + n_win - 1, n_win, dtype=int))
    return np.array(ex)


def get_fold_indices(labels: np.ndarray) -> dict:
    """Generate nested 5×5 stratified cross-validation index sets.

    Args:
        labels: Subject-level integer labels.

    Returns:
        dict keyed by 'outerX_innerY' with sub-keys 'train', 'test', 'val'.
        Indices are subject-level (not window-level).

    Note: The random state is fixed at SEED=42, matching the published results.
    """
    fold_idx = {}
    skf_out  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for i, (train_out, test_out) in enumerate(
        skf_out.split(np.zeros(len(labels)), labels)
    ):
        skf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for j, (train_in, val_in) in enumerate(
            skf_in.split(np.zeros(len(train_out)), labels[train_out])
        ):
            fold_idx[f"outer{i+1}_inner{j+1}"] = {
                "train": train_out[train_in],
                "val":   train_out[val_in],
                "test":  test_out,
            }

    return fold_idx


def get_outer_fold_indices(labels: np.ndarray) -> dict:
    """Generate five stratified outer folds with one validation split each.

    The validation split uses the first inner split from the nested 5x5
    protocol. This keeps each outer test fold identical to the nested setup
    while avoiding the extra inner-loop repetitions.
    """
    fold_idx = {}
    skf_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for i, (train_out, test_out) in enumerate(
        skf_out.split(np.zeros(len(labels)), labels)
    ):
        skf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_in, val_in = next(
            skf_in.split(np.zeros(len(train_out)), labels[train_out])
        )
        fold_idx[f"outer{i+1}_inner1"] = {
            "train": train_out[train_in],
            "val": train_out[val_in],
            "test": test_out,
        }

    return fold_idx


def get_single_fold_indices(labels: np.ndarray) -> dict:
    """Generate one deterministic train/val/test split for tiny debug datasets.

    This is intended for smoke-running the ST pipeline on small downloads such as
    five ABIDE AAL subjects. It is not a substitute for the published 5x5
    stratified protocol.
    """
    n_subjects = len(labels)
    if n_subjects < 3:
        raise ValueError("At least 3 subjects are required for a train/val/test split.")

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(n_subjects)
    test_size = max(1, n_subjects // 5)
    val_size = max(1, n_subjects // 5)

    test = np.sort(indices[:test_size])
    val = np.sort(indices[test_size:test_size + val_size])
    train = np.sort(indices[test_size + val_size:])

    if len(train) == 0:
        raise ValueError("Single-fold split produced an empty training set.")

    return {
        "outer1_inner1": {
            "train": train,
            "val": val,
            "test": test,
        }
    }


def load_ldw_data(ldw_dir: str = "./data/ldw_data"):
    """Load the processed LDW data produced by preprocessing.py (step 1).

    Args:
        ldw_dir: Directory containing LDW_abide_data.pkl.

    Returns:
        (data, adj, labels)
        data:   list of per-subject lists of correlation matrices.
        adj:    list of per-subject lists of adjacency matrices.
        labels: np.ndarray of subject labels.

    Note: formerly 'load_data' in step2_prepare_data.py.
    """
    path = os.path.join(ldw_dir, "LDW_abide_data.pkl")
    with open(path, "rb") as f:
        content = dill.load(f)
    return content["node_feat"], content["adj_mat"], content["labels"]


def get_fold_data(
    data, adj, seqlen, label,
    indices: dict,
    split: str,
    outer: int,
    inner: int,
):
    """Slice out one split from the full dataset for a given fold.

    Args:
        data:    Padded node features, indexable by subject.
        adj:     Padded adjacency matrices, indexable by subject.
        seqlen:  Sequence lengths per subject.
        label:   Subject labels.
        indices: Output of get_fold_indices.
        split:   'train', 'val', or 'test'.
        outer:   Outer fold index (1–5).
        inner:   Inner fold index (1–5).

    Returns:
        (Corr, Adj, SeqLen, Label) for the requested split.
    """
    key  = f"outer{outer}_inner{inner}"
    idxs = indices[key][split]
    return (
        [data[k]   for k in idxs],
        [adj[k]    for k in idxs],
        [seqlen[k] for k in idxs],
        [label[k]  for k in idxs],
    )


def pad_graph_seq(data: list):
    """Pad a list of variable-length window sequences to the same length.

    Args:
        data: list of lists; each inner list is a sequence of np.ndarray
              windows for one subject.

    Returns:
        (sequences_padded, seqlengths)
        sequences_padded: FloatTensor (subjects × max_windows × …)
        seqlengths:       LongTensor (subjects,)
    """
    sequences  = [torch.tensor(np.array(seq), dtype=torch.float32) for seq in data]
    seqlengths = torch.LongTensor([len(s) for s in sequences])
    padded     = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, seqlengths


def convert2graphs(
    Corr,
    Adj,
    SeqLen,
    Label,
) -> np.ndarray:
    """Convert padded arrays into a 2-D grid of PyG Data objects.

    Each cell [subject, window] is a torch_geometric.data.Data with:
        x:        node features (correlation matrix row).
        edge_index, edge_attr: non-zero edges from the adjacency matrix.
        adj:      full adjacency matrix (for dense ops in ST-BrainXDL).
        y:        subject label.
        pad:      True if this window is padding.
        last:     True if this is the last *valid* window for the subject.

    Args:
        Corr:   list of (max_windows × nodes × nodes) FloatTensors.
        Adj:    list of (max_windows × nodes × nodes) FloatTensors.
        SeqLen: list of original sequence lengths (LongTensors or ints).
        Label:  list of integer labels.

    Returns:
        np.ndarray of shape (subjects, max_windows), dtype=object.
    """
    n_sub  = len(Corr)
    n_win  = Corr[0].shape[0]
    n_cls  = int(max(Label)) + 1
    Graph  = np.empty([n_sub, n_win], dtype=object)

    for i in tqdm(range(n_sub), desc="Converting to PyG graphs"):
        for j in range(n_win):
            corr   = Corr[i][j]
            adj    = Adj[i][j]
            seqlen = SeqLen[i]
            label  = int(Label[i])

            padding = (torch.sum(adj) == 0)

            node_feat       = corr.float()
            num_nodes       = node_feat.shape[0]
            num_node_feat   = node_feat.shape[1]

            index     = torch.nonzero(adj)
            values    = adj[index[:, 0], index[:, 1]]
            edge_idx  = index.T.long()
            edge_attr = values.unsqueeze(-1).float()

            graph = Data(
                x=node_feat,
                edge_index=edge_idx,
                edge_attr=edge_attr,
                adj=adj.float(),
                y=torch.tensor(label, dtype=torch.long),
                num_nodes=num_nodes,
                num_node_features=num_node_feat,
                num_classes=n_cls,
                pad=padding,
                last=(j + 1 == int(seqlen)),
            )
            Graph[i, j] = graph

    return Graph


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 2: create cross-validation folds of PyG graphs."
    )
    parser.add_argument(
        "--ldw_dir",    default="./data/ldw_data",
        help="Directory containing LDW_abide_data.pkl (output of step 1).",
    )
    parser.add_argument(
        "--output_dir", default="./data/folds_data",
        help="Directory to save graphs_outerX_innerY.pkl files.",
    )
    parser.add_argument(
        "--single_fold",
        action="store_true",
        help=(
            "Create only graphs_outer1_inner1.pkl with a deterministic "
            "train/val/test split. Use for tiny debug datasets."
        ),
    )
    parser.add_argument(
        "--outer_folds_only",
        action="store_true",
        help=(
            "Create graphs_outer1_inner1.pkl through graphs_outer5_inner1.pkl "
            "only. Use for five-fold outer CV without the full nested inner loop."
        ),
    )
    args = parser.parse_args()

    if args.single_fold and args.outer_folds_only:
        parser.error("--single_fold and --outer_folds_only are mutually exclusive.")

    seed_everything(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading LDW data …")
    all_data, all_adj, labels = load_ldw_data(args.ldw_dir)

    print("Padding sequences …")
    all_data_padded, seqlengths = pad_graph_seq(all_data)
    all_adj_padded,  _          = pad_graph_seq(all_adj)

    print("Generating fold indices …")
    if args.single_fold:
        fold_indices = get_single_fold_indices(labels)
        fold_pairs = [(1, 1)]
    elif args.outer_folds_only:
        fold_indices = get_outer_fold_indices(labels)
        fold_pairs = [(i, 1) for i in range(1, 6)]
    else:
        fold_indices = get_fold_indices(labels)
        fold_pairs = [(i, j) for i in range(1, 6) for j in range(1, 6)]

    for i, j in fold_pairs:
        print(f"outer{i}_inner{j} …")
        graphs = {}
        for split in ("train", "test", "val"):
            C, A, S, L = get_fold_data(
                all_data_padded, all_adj_padded, seqlengths, labels,
                fold_indices, split, i, j,
            )
            graphs[f"{split}_graphs"] = convert2graphs(C, A, S, L)

        out_path = os.path.join(args.output_dir, f"graphs_outer{i}_inner{j}.pkl")
        with open(out_path, "wb") as f:
            torch.save(graphs, f)
        print(f"  Saved {out_path}")
