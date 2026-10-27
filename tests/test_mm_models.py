"""Smoke tests: MM-BrainXDL init + forward pass on dummy data."""

import argparse

import pytest
import torch


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        model="mmodal_gcn",
        modality="fusion",
        use_fmri=True,
        use_eeg=True,
        embid_dim=16,
        dropout=0.0,
        k_fmri=4,
        k_eeg=4,
        graph_mode="static",
        temporal_pool="mean",
        fusion_mode="gated",
        gcn_hidden_dim=32,
        brainnet_channels=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _dummy_batch(B: int = 2, N_fmri: int = 10, N_eeg: int = 8, device: torch.device = torch.device("cpu")):
    """Return a TensorDataset-style batch tuple matching prepare_data output shapes."""
    pearson_fmri = torch.randn(B, N_fmri, N_fmri, device=device)
    pearson_eeg = torch.randn(B, N_eeg, N_eeg, device=device)
    pcorr_fmri = torch.randn(B, N_fmri, N_fmri, device=device)
    pcorr_eeg = torch.randn(B, N_eeg, N_eeg, device=device)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    pseudo_fmri = torch.randn(B, N_fmri, N_fmri, device=device)
    pseudo_eeg = torch.randn(B, N_eeg, N_eeg, device=device)
    return (pearson_fmri, pearson_eeg, pcorr_fmri, pcorr_eeg, labels, pseudo_fmri, pseudo_eeg)


def test_mm_brainxdl_init_and_forward():
    from brainxdl.mm.models.brainmgf import MMBrainXDL, forward_model

    N_fmri, N_eeg, embed_dim = 10, 8, 16
    model = MMBrainXDL(
        in_channels_fmri=N_fmri,
        in_channels_eeg=N_eeg,
        embed_dim=embed_dim,
        fc_out=2,
        nrois_fmri=N_fmri,
        nrois_eeg=N_eeg,
        k_fmri=4,
        k_eeg=4,
    )
    args = _make_args()
    device = torch.device("cpu")
    batch = _dummy_batch(B=2, N_fmri=N_fmri, N_eeg=N_eeg, device=device)

    logits, labels, latent, gates = forward_model(model, batch, device, args, return_latent=True)

    assert logits.shape == (2, 2)
    assert labels.shape == (2,)
    assert latent is not None and latent.shape[0] == 2


@pytest.mark.parametrize(
    ("fusion_mode", "gate_type", "skip_dim", "classifier_type", "out_dim", "fused_dim"),
    [
        ("adaptive_sum", "scalar", 0, "mlp", 2, 16),
        ("gated_concat", "scalar", 0, "mlp", 2, 32),
        ("gated_concat", "scalar", 0, "linear", 1, 32),
        ("gated_concat", "scalar", 4, "mlp", 2, 40),
        ("gated_concat", "featurewise", 0, "mlp", 2, 32),
    ],
)
def test_refined_mm_variants_expose_finite_components(
    fusion_mode, gate_type, skip_dim, classifier_type, out_dim, fused_dim
):
    from brainxdl.mm.models.brainmgf import MMBrainXDL

    model = MMBrainXDL(
        in_channels_fmri=10, in_channels_eeg=8, embed_dim=16, fc_out=out_dim,
        nrois_fmri=10, nrois_eeg=8, k_fmri=4, k_eeg=4, dropout=0.0,
        fusion_mode=fusion_mode, gate_type=gate_type,
        connectivity_skip_dim=skip_dim, classifier_type=classifier_type,
    ).eval()
    batch = _dummy_batch(B=2, N_fmri=10, N_eeg=8)
    logits = model(batch[0], batch[1], batch[2], batch[3], batch[5], batch[6])
    components = model.get_latest_components()
    assert logits.shape == (2, out_dim)
    assert components is not None
    assert components["z_fmri"].shape == (2, 16)
    assert components["z_eeg"].shape == (2, 16)
    assert components["z_concat"].shape == (2, 32)
    assert components["adaptive_sum"].shape == (2, 16)
    assert components["fused"].shape == (2, fused_dim)
    assert all(torch.isfinite(value).all() for value in components.values())


def test_default_mm_path_preserves_original_parameter_layout_and_fusion_equation():
    from brainxdl.mm.models.brainmgf import MMBrainXDL

    model = MMBrainXDL(
        in_channels_fmri=10, in_channels_eeg=8, embed_dim=16, fc_out=2,
        nrois_fmri=10, nrois_eeg=8, k_fmri=4, k_eeg=4, dropout=0.0,
    ).eval()
    assert not any(key.startswith("skip_") for key in model.state_dict())
    assert not any(key.startswith("graph_decoder_") for key in model.state_dict())
    assert model.get_latest_reconstruction_losses() is None
    assert model.classifier[0].in_features == 16
    batch = _dummy_batch(B=2, N_fmri=10, N_eeg=8)
    model(batch[0], batch[1], batch[2], batch[3], batch[5], batch[6])
    components = model.get_latest_components()
    expected = (
        components["gate_weights"][:, 0:1] * components["z_fmri"]
        + components["gate_weights"][:, 1:2] * components["z_eeg"]
    )
    assert torch.equal(components["adaptive_sum"], expected)


def test_self_preserving_graph_encoder_is_opt_in_and_finite():
    from brainxdl.mm.models.brainmgf import (
        GraphConvModel,
        MMBrainXDL,
        SelfPreservingGraphConvModel,
    )

    published = MMBrainXDL(
        in_channels_fmri=10, in_channels_eeg=8, embed_dim=16, fc_out=2,
        nrois_fmri=10, nrois_eeg=8, k_fmri=4, k_eeg=4, dropout=0.0,
    ).eval()
    refined = MMBrainXDL(
        in_channels_fmri=10, in_channels_eeg=8, embed_dim=16, fc_out=2,
        nrois_fmri=10, nrois_eeg=8, k_fmri=4, k_eeg=4, dropout=0.0,
        graph_encoder_type="self_preserving",
    ).eval()
    assert isinstance(published.gnn_fmri, GraphConvModel)
    assert isinstance(refined.gnn_fmri, SelfPreservingGraphConvModel)
    assert not any("self0" in key for key in published.state_dict())
    assert any("self0" in key for key in refined.state_dict())

    batch = _dummy_batch(B=2, N_fmri=10, N_eeg=8)
    logits = refined(batch[0], batch[1], batch[2], batch[3], batch[5], batch[6])
    components = refined.get_latest_components()
    assert logits.shape == (2, 2)
    assert components is not None
    assert components["z_graph_fmri"].shape == (2, 16)
    assert components["z_graph_eeg"].shape == (2, 16)
    assert all(torch.isfinite(value).all() for value in components.values())


def test_self_preserving_block_uses_separate_self_and_neighbour_paths():
    from brainxdl.mm.models.brainmgf import SelfPreservingGraphConvModel

    encoder = SelfPreservingGraphConvModel(node_input_dim=3, roi_num=3, embed_dim=2).eval()
    node_features = torch.randn(2, 3, 3)
    zero_adjacency = torch.zeros(2, 3, 3)
    output = encoder.forward_components(node_features, zero_adjacency)["block0"]
    assert output.shape == (2, 3, 3)
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, torch.zeros_like(output))


def test_sign_separated_first_block_encoder_is_opt_in_and_finite():
    from brainxdl.mm.models.brainmgf import (
        MMBrainXDL,
        SignSeparatedFirstBlockGraphConvModel,
    )

    model = MMBrainXDL(
        in_channels_fmri=10, in_channels_eeg=8, embed_dim=16, fc_out=2,
        nrois_fmri=10, nrois_eeg=8, k_fmri=4, k_eeg=4, dropout=0.0,
        graph_encoder_type="sign_separated_first",
    ).eval()
    assert isinstance(model.gnn_fmri, SignSeparatedFirstBlockGraphConvModel)
    assert any("positive0" in key for key in model.state_dict())
    assert any("negative0" in key for key in model.state_dict())

    batch = _dummy_batch(B=2, N_fmri=10, N_eeg=8)
    logits = model(batch[0], batch[1], batch[2], batch[3], batch[5], batch[6])
    components = model.get_latest_components()
    assert logits.shape == (2, 2)
    assert components is not None
    assert components["z_graph_fmri"].shape == (2, 16)
    assert components["z_graph_eeg"].shape == (2, 16)
    assert all(torch.isfinite(value).all() for value in components.values())


def test_sign_separated_encoder_handles_empty_sign_rows():
    from brainxdl.mm.models.brainmgf import SignSeparatedFirstBlockGraphConvModel

    encoder = SignSeparatedFirstBlockGraphConvModel(
        node_input_dim=3, roi_num=3, embed_dim=2
    ).eval()
    node_features = torch.randn(2, 3, 3)
    adjacency = torch.zeros(2, 3, 3)
    adjacency[0, 0, 1] = 0.5
    adjacency[0, 1, 0] = -0.5
    components = encoder.forward_components(node_features, adjacency)
    assert components["block0"].shape == (2, 3, 3)
    assert components["block1"].shape == (2, 3, 3)
    assert components["block2"].shape == (2, 3, 2)
    assert all(torch.isfinite(value).all() for value in components.values())


def test_graph_reconstruction_head_uses_symmetric_absolute_support():
    from brainxdl.mm.models.brainmgf import GraphReconstructionHead

    decoder = GraphReconstructionHead(dropout=0.0)
    adjacency = torch.tensor(
        [[[1.0, -0.5, 0.0], [0.0, 1.0, 0.25], [0.0, 0.0, 1.0]]]
    )
    expected = torch.tensor(
        [[[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]]
    )
    assert torch.equal(decoder.target(adjacency), expected)

    embeddings = torch.randn(1, 3, 4, requires_grad=True)
    loss = decoder.loss(embeddings, adjacency)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.norm() > 0


def test_graph_autoencoder_regularizer_is_opt_in_and_reaches_both_encoders():
    from brainxdl.mm.models.brainmgf import MMBrainXDL
    from brainxdl.mm.refined_benchmark import VARIANTS

    variant = VARIANTS["mm_g3_graph_autoencoder"]
    assert variant.graph_encoder_type == "sign_separated_first"
    assert variant.graph_reconstruction_weight == pytest.approx(0.1)

    model = MMBrainXDL(
        in_channels_fmri=10, in_channels_eeg=8, embed_dim=16, fc_out=2,
        nrois_fmri=10, nrois_eeg=8, k_fmri=4, k_eeg=4, dropout=0.0,
        graph_encoder_type="sign_separated_first", graph_reconstruction=True,
        graph_decoder_dropout=0.0,
    ).train()
    batch = _dummy_batch(B=2, N_fmri=10, N_eeg=8)
    logits = model(batch[0], batch[1], batch[2], batch[3], batch[5], batch[6])
    losses = model.get_latest_reconstruction_losses()
    assert logits.shape == (2, 2)
    assert losses is not None and set(losses) == {"fmri", "eeg"}
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())

    torch.stack(list(losses.values())).mean().backward()
    assert model.gnn_fmri.positive0.weight.grad is not None
    assert model.gnn_fmri.positive0.weight.grad.norm() > 0
    assert model.gnn_eeg.positive0.weight.grad is not None
    assert model.gnn_eeg.positive0.weight.grad.norm() > 0
    assert model.gnn_eeg.negative0.weight.grad is not None
    assert model.gnn_eeg.negative0.weight.grad.norm() > 0


@pytest.mark.parametrize("fusion_mode", ["average", "adaptive_sum"])
def test_residual_mm_adds_projected_connectivity_before_fusion(fusion_mode):
    from brainxdl.mm.models.brainmgf import MMBrainXDL

    model = MMBrainXDL(
        in_channels_fmri=10, in_channels_eeg=8, embed_dim=16, fc_out=2,
        nrois_fmri=10, nrois_eeg=8, k_fmri=4, k_eeg=4, dropout=0.0,
        fusion_mode=fusion_mode, residual_connectivity_dim=4, residual_lambda=1.0,
    ).eval()
    batch = _dummy_batch(B=2, N_fmri=10, N_eeg=8)
    logits = model(batch[0], batch[1], batch[2], batch[3], batch[5], batch[6])
    components = model.get_latest_components()
    assert logits.shape == (2, 2)
    assert torch.allclose(
        components["z_refined_fmri"],
        components["z_graph_fmri"] + components["z_residual_fmri"],
    )
    assert torch.allclose(
        components["z_refined_eeg"],
        components["z_graph_eeg"] + components["z_residual_eeg"],
    )
    for modality in ("fmri", "eeg"):
        for block in range(3):
            summary = components[f"{modality}_block{block}_summary"]
            assert summary.shape[0] == 2
            assert torch.isfinite(summary).all()


def test_mm_brainxdl_fmri_only():
    from brainxdl.mm.models.brainmgf import build_model, forward_model

    args = _make_args(model="mmodal_gcn", modality="fmri", use_fmri=True, use_eeg=False, out_channels=2)
    device = torch.device("cpu")
    model = build_model(args, n_fmri=10, n_eeg=8, model_out_dim=2, device=device)
    batch = _dummy_batch(B=2, N_fmri=10, N_eeg=8, device=device)
    logits, labels, _, _ = forward_model(model, batch, device, args)
    assert logits.shape == (2, 2)


def test_dynamic_mm_brainxdl_forward():
    from brainxdl.mm.models.dynamic_brainmgf import DynamicMMBrainXDL

    N_fmri, N_eeg, T = 10, 8, 3
    model = DynamicMMBrainXDL(
        in_channels_fmri=N_fmri,
        in_channels_eeg=N_eeg,
        embed_dim=16,
        fc_out=2,
        nrois_fmri=N_fmri,
        nrois_eeg=N_eeg,
        k_fmri=4,
        k_eeg=4,
    )
    B = 2
    pearson_fmri = torch.randn(B, T, N_fmri, N_fmri)
    pearson_eeg = torch.randn(B, T, N_eeg, N_eeg)
    pcorr_fmri = torch.randn(B, T, N_fmri, N_fmri)
    pcorr_eeg = torch.randn(B, T, N_eeg, N_eeg)
    pseudo_fmri = torch.randn(B, T, N_fmri, N_fmri)  # same layout as pearson
    pseudo_eeg = torch.randn(B, T, N_eeg, N_eeg)

    out = model(pearson_fmri, pearson_eeg, pcorr_fmri, pcorr_eeg, pseudo_fmri, pseudo_eeg)
    assert out.shape == (B, 2)


def test_compute_probs_and_preds_ce():
    from brainxdl.mm.training.trainer import compute_probs_and_preds

    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    probs, preds = compute_probs_and_preds(logits, binary_mode=False)
    assert preds.tolist() == [1, 0]


def test_compute_probs_and_preds_bce():
    from brainxdl.mm.training.trainer import compute_probs_and_preds

    logits = torch.tensor([[2.0], [-1.0]])
    probs, preds = compute_probs_and_preds(logits, binary_mode=True)
    assert preds.tolist() == [1, 0]
