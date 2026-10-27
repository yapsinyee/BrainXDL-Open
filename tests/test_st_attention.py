import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

from brainxdl.xai.st_attention import (
    STAttentionRecord,
    TransformerAttentionCapture,
    records_to_component_series,
    summarize_attention_series,
    top_undirected_edges,
)


class TinyAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = TransformerConv(3, 2)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


def test_attention_capture_preserves_forward_output_and_restores_layer():
    torch.manual_seed(7)
    model = TinyAttentionModel().eval()
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    expected = model(x, edge_index)
    original_forward = model.conv.forward

    with TransformerAttentionCapture(model) as capture:
        observed = model(x, edge_index)

    assert torch.allclose(observed, expected)
    assert len(capture.records) == 1
    assert capture.records[0].module_name == "conv"
    assert capture.records[0].alpha.shape[0] >= edge_index.shape[1]
    assert model.conv.forward == original_forward


def test_encoder_calls_map_one_to_one_to_time_steps():
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    alpha_first = torch.tensor([[0.2], [0.8]])
    alpha_second = torch.tensor([[0.7], [0.3]])
    records = [
        STAttentionRecord("enc_z_hidden.module_0", 0, edges, alpha_first),
        STAttentionRecord("enc_z_hidden.module_0", 1, edges, alpha_second),
    ]

    series = records_to_component_series(records, num_nodes=2)

    assert list(series) == ["posterior_encoder.module_0"]
    assert series["posterior_encoder.module_0"].shape == (2, 2, 2)
    assert np.isclose(series["posterior_encoder.module_0"][0, 0, 1], 0.2)
    assert np.isclose(series["posterior_encoder.module_0"][1, 0, 1], 0.7)


def test_attention_summary_separates_presence_from_conditional_weight():
    series = np.asarray(
        [
            [[0.0, 0.2], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.6], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    mean, conditional, presence = summarize_attention_series(series)

    assert np.isclose(mean[0, 1], 0.8 / 3)
    assert np.isclose(conditional[0, 1], 0.4)
    assert np.isclose(presence[0, 1], 2 / 3)


def test_top_edges_symmetrizes_directional_attention():
    matrix = np.asarray(
        [[0.0, 0.8, 0.1], [0.2, 0.0, 0.3], [0.1, 0.5, 0.0]],
        dtype=np.float32,
    )

    rows = top_undirected_edges(matrix, top_k=2)

    assert rows[0][:2] == (0, 1)
    assert np.isclose(rows[0][2], 0.5)
    assert rows[1][:2] == (1, 2)
