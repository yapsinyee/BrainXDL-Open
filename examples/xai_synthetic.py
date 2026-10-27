"""Run the project-specific dense edge-mask explainer on synthetic MM input."""

import torch

from brainxdl.mm.models.brainmgf import MMBrainXDL
from brainxdl.xai.explainer import DenseGNNExplainer


def symmetric(nodes: int) -> torch.Tensor:
    values = torch.randn(1, nodes, nodes)
    values = 0.5 * (values + values.transpose(-1, -2))
    return values * (1.0 - torch.eye(nodes)) + torch.eye(nodes)


def main() -> None:
    torch.manual_seed(2026)
    nf, ne = 6, 5
    model = MMBrainXDL(nf, ne, 8, 2, nf, ne, k_fmri=3, k_eeg=3, dropout=0.0).eval()
    inputs = (
        symmetric(nf), symmetric(ne), symmetric(nf),
        symmetric(ne), symmetric(nf), symmetric(ne),
    )
    explainer = DenseGNNExplainer(
        model, torch.device("cpu"), binary_mode=False,
        use_fmri=True, use_eeg=True, epochs=10,
    )
    explanation = explainer.explain(*inputs)
    print("Predicted class:", explanation["prediction"])
    print("Prediction confidence:", explanation["prediction_prob"])
    print("fMRI mask shape:", explanation["edge_masks"]["fmri"].shape)
    print("EEG mask shape:", explanation["edge_masks"]["eeg"].shape)
    print("Method: optimized symmetric edge mask; signed saliency is mask * adjacency")


if __name__ == "__main__":
    main()
