"""Minimal ST-BrainXDL forward pass using two synthetic graph sequences."""

import torch
from torch_geometric.data import Batch, Data

from brainxdl.st.model import STBrainXDL


def main() -> None:
    torch.manual_seed(2026)
    nodes, features = 4, 3
    rows, cols = torch.meshgrid(torch.arange(nodes), torch.arange(nodes), indexing="ij")
    edge_index = torch.stack([rows.flatten(), cols.flatten()])
    steps = []
    for step in range(2):
        graphs = [
            Data(
                x=torch.randn(nodes, features),
                edge_index=edge_index,
                adj=torch.eye(nodes),
                y=torch.tensor(label),
                pad=torch.tensor(False),
                last=torch.tensor(step == 1),
                num_classes=2,
            )
            for label in (0, 1)
        ]
        steps.append(Batch.from_data_list(graphs))

    setting = {
        "device": "cpu", "sparse_on_cpu": True, "recurrent": False,
        "graphRNN": False, "variational": False,
        "model_params": {
            "num_nodes": nodes, "num_classes": 2, "x_dim": features, "y_dim": 2,
            "z_hidden_dim": 4, "z_dim": 3, "z_phi_dim": 3, "x_phi_dim": 4,
            "rnn_dim": 3, "y_hidden_dim": [], "x_hidden_dim": 4,
            "layer_dims": [], "classifier_dropout": 0.0,
        },
    }
    model = STBrainXDL(setting).eval()
    x_nll, z_kld, adjacency_nll, readout, target = model(steps, setting)
    logits = model.classifier.linear(readout)
    print("ST logits shape:", tuple(logits.shape))
    print("ST targets:", target.tolist())
    print("ST losses:", {"x_nll": x_nll.item(), "z_kld": z_kld.item(), "adjacency_nll": adjacency_nll.item()})


if __name__ == "__main__":
    main()
