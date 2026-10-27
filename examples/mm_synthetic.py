"""Minimal MM-BrainXDL forward pass with synthetic fMRI and EEG matrices."""

import torch

from brainxdl.mm.models.brainmgf import MMBrainXDL


def symmetric(batch: int, nodes: int) -> torch.Tensor:
    values = torch.randn(batch, nodes, nodes)
    values = 0.5 * (values + values.transpose(-1, -2))
    return values * (1.0 - torch.eye(nodes)) + torch.eye(nodes)


def main() -> None:
    torch.manual_seed(2026)
    nf, ne = 6, 5
    model = MMBrainXDL(nf, ne, 8, 2, nf, ne, k_fmri=3, k_eeg=3, dropout=0.0).eval()
    inputs = (
        symmetric(2, nf), symmetric(2, ne), symmetric(2, nf),
        symmetric(2, ne), symmetric(2, nf), symmetric(2, ne),
    )
    logits = model(*inputs)
    probabilities = torch.softmax(logits, dim=-1)
    print("MM logits shape:", tuple(logits.shape))
    print("MM class probabilities:", probabilities.detach().tolist())
    print("MM gate weights:", model.get_latest_components()["gate_weights"].detach().tolist())


if __name__ == "__main__":
    main()
