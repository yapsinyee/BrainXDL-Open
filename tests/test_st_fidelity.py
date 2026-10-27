import numpy as np
import torch
from torch_geometric.data import Data

from brainxdl.xai.st_attention import STAttentionRecord
from brainxdl.xai.st_fidelity import (
    aggregate_attention_importance,
    deterministic_random_seed,
    graph_sequence_digest,
    mask_graph_sequence,
    removal_count,
    select_edge_indices,
    undirected_edge_pairs,
)


def test_attention_importance_averages_directions_windows_and_components():
    edges = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    records = [
        STAttentionRecord("enc_z_hidden.module_0", 0, edges, torch.tensor([[0.8], [0.2], [0.4], [0.0]])),
        STAttentionRecord("enc_z_hidden.module_0", 1, edges, torch.tensor([[0.4], [0.4], [0.2], [0.2]])),
        STAttentionRecord("rnn_cell.weight_xz", 0, edges, torch.tensor([[0.2], [0.2], [0.6], [0.2]])),
        STAttentionRecord("rnn_cell.weight_xz", 1, edges, torch.tensor([[0.6], [0.2], [0.0], [0.4]])),
    ]

    scores, components = aggregate_attention_importance(
        records, num_nodes=3, valid_window_mask=np.asarray([True, True])
    )

    assert components == ["posterior_encoder.module_0", "gru.update.input"]
    pairs = undirected_edge_pairs(3)
    values = {tuple(pair): score for pair, score in zip(pairs, scores)}
    assert np.isclose(values[(0, 1)], 0.375)
    assert np.isclose(values[(0, 2)], 0.25)
    assert values[(1, 2)] == 0.0


def test_edge_selection_is_matched_and_random_is_reproducible():
    importance = np.linspace(0.0, 1.0, 100)
    count = removal_count(len(importance), 0.20)
    top = select_edge_indices(importance, 0.20, "top")
    low = select_edge_indices(importance, 0.20, "low")
    seed = deterministic_random_seed(2026, 3, 17, 0.20, 4)
    random_first = select_edge_indices(
        importance, 0.20, "random", rng=np.random.default_rng(seed)
    )
    random_second = select_edge_indices(
        importance, 0.20, "random", rng=np.random.default_rng(seed)
    )

    assert len(top) == len(low) == len(random_first) == count
    assert np.array_equal(top, np.arange(80, 100))
    assert np.array_equal(low, np.arange(20))
    assert np.array_equal(random_first, random_second)


def test_mask_graph_sequence_preserves_original_symmetry_and_diagonal():
    x = torch.tensor(
        [[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]]
    )
    adj = torch.ones((3, 3), dtype=torch.float32)
    edge_index = torch.nonzero(adj, as_tuple=False).T.contiguous()
    graph = Data(
        x=x,
        adj=adj,
        edge_index=edge_index,
        edge_attr=torch.ones((edge_index.shape[1], 1)),
        y=torch.tensor(1),
        pad=False,
        last=True,
        num_nodes=3,
    )
    before = graph_sequence_digest([graph])

    masked_sequence = mask_graph_sequence(
        [graph], np.asarray([[0, 2]], dtype=np.int64)
    )
    assert masked_sequence.shape == (1,)
    assert masked_sequence.dtype == object
    assert isinstance(masked_sequence[0], Data)
    masked = masked_sequence[0]

    assert graph_sequence_digest([graph]) == before
    assert masked.x[0, 2] == masked.x[2, 0] == 0
    assert masked.adj[0, 2] == masked.adj[2, 0] == 0
    assert torch.equal(torch.diagonal(masked.x), torch.ones(3))
    assert torch.equal(torch.diagonal(masked.adj), torch.ones(3))
    remaining = {tuple(edge) for edge in masked.edge_index.T.tolist()}
    assert (0, 2) not in remaining and (2, 0) not in remaining
    assert torch.allclose(masked.x, masked.x.T)
    assert torch.allclose(masked.adj, masked.adj.T)
