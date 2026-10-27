import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from brainxdl.xai.st_fidelity import graph_sequence_digest
from brainxdl.xai.st_temporal import (
    block_scores_to_windows,
    oriented_prediction_margin,
    replace_temporal_blocks,
    select_nonoverlapping_blocks,
    temporal_block_starts,
    threshold_centered_log_odds,
)


def _sequence(num_windows=6):
    graphs = np.empty(num_windows, dtype=object)
    for index in range(num_windows):
        value = float(index + 1)
        x = torch.full((3, 3), value)
        adj = torch.eye(3) * value
        edge_index = torch.nonzero(torch.ones((3, 3)), as_tuple=False).T.contiguous()
        graphs[index] = Data(
            x=x,
            adj=adj,
            edge_index=edge_index,
            edge_attr=torch.full((edge_index.shape[1], 1), value),
            y=torch.tensor(1),
            pad=torch.tensor(False),
            last=torch.tensor(index == num_windows - 1),
            num_nodes=3,
        )
    return graphs


def test_temporal_block_starts_covers_tail():
    assert np.array_equal(temporal_block_starts(10, 4, 3), [0, 3, 6])
    assert np.array_equal(temporal_block_starts(10, 4, 4), [0, 4, 6])


def test_replace_temporal_block_is_non_mutating_and_preserves_controls():
    graphs = _sequence()
    before = graph_sequence_digest(graphs)
    replaced = replace_temporal_blocks(graphs, [(2, 4)])

    assert graph_sequence_digest(graphs) == before
    assert replaced.dtype == object
    assert torch.equal(replaced[2].x, graphs[1].x)
    assert torch.equal(replaced[3].x, graphs[4].x)
    assert bool(replaced[-1].last)
    assert not bool(replaced[3].last)
    replaced[2].x[0, 0] = -1
    assert graphs[1].x[0, 0] != -1


def test_replace_temporal_blocks_rejects_full_sequence():
    with pytest.raises(ValueError, match="remain unselected"):
        replace_temporal_blocks(_sequence(), [(0, 6)])


def test_block_scores_are_averaged_onto_windows():
    result = block_scores_to_windows([2.0, 4.0, 8.0], [0, 1, 2], 2, 4)
    assert np.allclose(result, [2.0, 3.0, 6.0, 8.0])


def test_nonoverlapping_selection_is_ranked_and_reproducible():
    scores = [0.1, 0.9, 0.3, 0.8, -0.2, 0.4, 0.7]
    starts = np.arange(len(scores))
    top = select_nonoverlapping_blocks(scores, starts, 2, 2, "top")
    low = select_nonoverlapping_blocks(scores, starts, 2, 2, "low")
    random_a = select_nonoverlapping_blocks(
        scores, starts, 2, 2, "random", rng=np.random.default_rng(2026)
    )
    random_b = select_nonoverlapping_blocks(
        scores, starts, 2, 2, "random", rng=np.random.default_rng(2026)
    )

    assert np.array_equal(top, [1, 3])
    assert np.array_equal(low, [0, 4])
    assert np.array_equal(random_a, random_b)
    for selected in (top, low, random_a):
        assert starts[selected[1]] >= starts[selected[0]] + 2


def test_threshold_centered_margin_uses_validation_threshold():
    logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    centered = threshold_centered_log_odds(logits, threshold=0.75)
    expected_offset = np.log(0.75 / 0.25)
    expected = torch.tensor(
        [2.0 - expected_offset, -2.0 - expected_offset], dtype=centered.dtype
    )
    assert torch.allclose(centered, expected)

    margins, predicted = oriented_prediction_margin(logits, threshold=0.75)
    assert torch.equal(predicted, torch.tensor([1, 0]))
    assert torch.all(margins >= 0)
