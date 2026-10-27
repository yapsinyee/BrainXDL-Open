"""Integration smoke tests for tiny ST and MM train/eval runs."""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _make_mm_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        model="mmodal_gcn",
        modality="fmri",
        use_fmri=True,
        use_eeg=False,
        embid_dim=8,
        dropout=0.0,
        k_fmri=2,
        k_eeg=2,
        graph_mode="static",
        temporal_pool="mean",
        fusion_mode="gated",
        gcn_hidden_dim=16,
        brainnet_channels=None,
        batch_size=2,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _symmetric_stack(num_subjects: int, num_nodes: int) -> torch.Tensor:
    mats = torch.randn(num_subjects, num_nodes, num_nodes)
    mats = 0.5 * (mats + mats.transpose(-1, -2))
    eye = torch.eye(num_nodes).unsqueeze(0)
    return mats * (1.0 - eye) + eye


def test_tiny_mm_training_and_evaluation_smoke():
    from brainxdl.mm.models.brainmgf import build_model
    from brainxdl.mm.training.trainer import evaluate, evaluate_metrics, train_one_epoch

    torch.manual_seed(7)
    device = torch.device("cpu")
    args = _make_mm_args()
    num_subjects = 4
    n_fmri = 6
    n_eeg = 5

    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(
        _symmetric_stack(num_subjects, n_fmri),
        _symmetric_stack(num_subjects, n_eeg),
        _symmetric_stack(num_subjects, n_fmri),
        _symmetric_stack(num_subjects, n_eeg),
        labels,
        _symmetric_stack(num_subjects, n_fmri),
        _symmetric_stack(num_subjects, n_eeg),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = build_model(args, n_fmri=n_fmri, n_eeg=n_eeg, model_out_dim=2, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loss, train_acc = train_one_epoch(
        loader,
        model,
        criterion,
        optimizer,
        device,
        args,
        grad_clip=0.5,
        binary_mode=False,
    )
    eval_loss, eval_acc = evaluate(loader, model, criterion, device, args, binary_mode=False)
    metrics = evaluate_metrics(loader, model, criterion, device, args, binary_mode=False)

    assert np.isfinite(train_loss)
    assert 0.0 <= train_acc <= 1.0
    assert np.isfinite(eval_loss)
    assert 0.0 <= eval_acc <= 1.0
    assert set(metrics) == {"loss", "log_loss", "acc", "pre", "sen", "spe", "f1", "roc"}
    assert np.isfinite(metrics["loss"])


def _make_st_graph(label: int, *, last: bool, num_nodes: int, x_dim: int):
    from torch_geometric.data import Data

    rows, cols = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
    edge_index = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=0)
    adj = torch.ones(num_nodes, num_nodes, dtype=torch.float32)
    x = torch.randn(num_nodes, x_dim, dtype=torch.float32)
    return Data(
        x=x,
        edge_index=edge_index,
        adj=adj,
        y=torch.tensor(label, dtype=torch.long),
        pad=torch.tensor(False),
        last=torch.tensor(last),
        num_classes=2,
    )


def _make_st_sequences(labels, *, num_windows: int = 2, num_nodes: int = 4, x_dim: int = 3) -> np.ndarray:
    sequences = np.empty((len(labels), num_windows), dtype=object)
    for subject_idx, label in enumerate(labels):
        for step in range(num_windows):
            sequences[subject_idx, step] = _make_st_graph(
                label,
                last=step == num_windows - 1,
                num_nodes=num_nodes,
                x_dim=x_dim,
            )
    return sequences


def test_tiny_st_training_and_evaluation_smoke(tmp_path):
    from brainxdl.common.checkpointing import load_checkpoint
    from brainxdl.st.visualize import analyze_checkpoint
    from brainxdl.st.dataset import STGraphDataset, padseq
    from brainxdl.st.model import STBrainXDL
    from brainxdl.st.trainer import init_checkpoint, train, validate

    torch.manual_seed(11)
    device = torch.device("cpu")
    train_dataset = STGraphDataset(_make_st_sequences([0, 1]))
    val_dataset = STGraphDataset(_make_st_sequences([0, 1]))
    test_dataset = STGraphDataset(_make_st_sequences([0, 1]))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=padseq, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=padseq, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=padseq, drop_last=False)

    setting = {
        "device": str(device),
        "prefer_mps": False,
        "sparse_on_cpu": True,
        "rngPATH": None,
        "model_params": {
            "num_nodes": 4,
            "num_classes": 2,
            "x_dim": 3,
            "y_dim": 2,
            "z_hidden_dim": 4,
            "z_dim": 3,
            "z_phi_dim": 3,
            "x_phi_dim": 4,
            "rnn_dim": 3,
            "y_hidden_dim": [],
            "x_hidden_dim": 4,
            "layer_dims": [],
        },
        "recurrent": False,
        "graphRNN": False,
        "variational": False,
        "DAT": False,
        "bce_weight": 2.0,
        "mce_weight": 0.5,
        "learnRate": [1e-3, 1e-3],
        "yBCEMultiplier": [1, 1],
        "l2factor": [0.0, 0.0],
        "lr_annealType": [None, None],
        "lr_annealFactor": [0.8, 0.8],
        "lr_annealPatience": [30, 30],
        "partition": [len(train_dataset), len(val_dataset), len(test_dataset)],
    }
    checkpoint_path = str(tmp_path / "tiny_st_smoke.pth")

    model, optimizers, schedulers, epoch_start, train_losses, val_losses, test_losses = init_checkpoint(
        setting,
        checkpoint_path,
        checkpoint_path,
    )
    model, train_losses, val_losses, test_losses = train(
        model,
        optimizers,
        schedulers,
        setting,
        checkpoint_path,
        train_losses,
        val_losses,
        test_losses,
        train_loader,
        val_loader,
        test_loader,
        epoch_start=epoch_start,
        num_epochs=1,
        grad_threshold=0.5,
        gradient_clip=True,
        verbose=False,
        val_freq=1,
        validation=True,
        testing=True,
        early_stop=False,
        device=device,
    )
    eval_losses = validate(model, setting, val_loader)

    assert len(train_losses["Total"]) == 1
    assert len(val_losses["Total"]) == 1
    assert len(test_losses["Total"]) == 1
    assert torch.isfinite(eval_losses["Total"]).item()
    expected_total = (
        eval_losses["x_NLL"]
        + eval_losses["z_KLD"]
        + 0.5 * eval_losses["a_NLL"]
        + 2.0 * eval_losses["y_BCE"]
    )
    assert torch.allclose(eval_losses["Total"], expected_total)
    assert 0.0 <= float(eval_losses["y_ACC"]) <= 1.0
    assert (tmp_path / "tiny_st_smoke.pth").is_file()

    ckpt = load_checkpoint(checkpoint_path, device=device)
    loaded_setting = ckpt["training_setting"]
    loaded_setting["device"] = str(device)
    loaded_setting["sparse_on_cpu"] = True
    loaded_model = STBrainXDL(loaded_setting)
    loaded_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    loaded_model.to(device)
    loaded_model.apply_device_split(device, sparse_on_cpu=True)
    loaded_eval_losses = validate(loaded_model, loaded_setting, val_loader)
    analyze_checkpoint(checkpoint_path)

    assert torch.isfinite(loaded_eval_losses["Total"]).item()
    assert 0.0 <= float(loaded_eval_losses["y_ACC"]) <= 1.0


def test_st_training_workflow_smoke(tmp_path, monkeypatch):
    from brainxdl.pipelines import st_workflow

    data_dir = tmp_path / "folds_data"
    save_dir = tmp_path / "saved_models"
    data_dir.mkdir()

    train_graphs = _make_st_sequences([0, 1], num_nodes=3, x_dim=2)
    val_graphs = _make_st_sequences([1], num_nodes=3, x_dim=2)
    test_graphs = _make_st_sequences([0], num_nodes=3, x_dim=2)
    torch.save(
        {
            "train_graphs": train_graphs,
            "val_graphs": val_graphs,
            "test_graphs": test_graphs,
        },
        data_dir / "graphs_outer1_inner1.pkl",
    )

    calls = {}

    class TinyModel:
        def __str__(self):
            return "TinyModel()"

    def fake_init_checkpoint(setting, load_path, save_path):
        calls["setting"] = setting
        calls["load_path"] = load_path
        calls["save_path"] = save_path
        return (
            TinyModel(),
            ["optimizer"],
            ["scheduler"],
            0,
            {"Total": []},
            {"Total": []},
            {"Total": []},
        )

    def fake_train(
        model,
        optimizers,
        schedulers,
        setting,
        checkpoint_path,
        train_losses,
        val_losses,
        test_losses,
        train_loader,
        val_loader,
        test_loader,
        **kwargs,
    ):
        calls["train_batches"] = len(train_loader)
        calls["val_batches"] = len(val_loader)
        calls["test_batches"] = len(test_loader)
        calls["train_kwargs"] = kwargs
        train_losses["Total"].append(1.0)
        val_losses["Total"].append(1.0)
        test_losses["Total"].append(1.0)
        return model, train_losses, val_losses, test_losses

    monkeypatch.setattr(st_workflow, "init_checkpoint", fake_init_checkpoint)
    monkeypatch.setattr(st_workflow, "train", fake_train)

    args = argparse.Namespace(
        outer=1,
        inner=1,
        version="smoke",
        data_dir=str(data_dir),
        save_dir=str(save_dir),
        prefer_mps=False,
        batch_size=2,
        num_epochs=1,
        learn_rate=1e-3,
        l2factor=0.0,
        kl_weight=1.0,
        kl_start_weight=None,
        kl_warmup_epochs=0,
        bce_weight=2.0,
        mce_weight=0.5,
        patience=1,
        val_freq=1,
        grad_clip=0.5,
        no_early_stop=True,
        no_verbose=True,
        no_variational=True,
        no_recurrent=True,
        no_graphRNN=True,
        dat=False,
    )

    result = st_workflow.run_st_training_workflow(args)

    assert result.partition == [3, 1, 1]
    assert result.setting["model_params"]["num_nodes"] == 3
    assert result.setting["model_params"]["x_dim"] == 2
    assert result.setting["model_params"]["y_dim"] == 2
    assert result.setting["recurrent"] is False
    assert result.setting["graphRNN"] is False
    assert result.setting["variational"] is False
    assert result.setting["bce_weight"] == 2.0
    assert result.setting["mce_weight"] == 0.5
    assert result.save_path == str(save_dir / "ST-BrainXDL_softmax_adv_fold11_smoke.pth")
    assert calls["load_path"] == result.save_path
    assert calls["save_path"] == result.save_path
    assert calls["train_batches"] == 1
    assert calls["val_batches"] == 1
    assert calls["test_batches"] == 1
    assert calls["train_kwargs"]["num_epochs"] == 1
    assert result.train_losses["Total"] == [1.0]

    args.strict_validation = True
    args.no_test_during_training = True
    args.version = "strict"
    strict_result = st_workflow.run_st_training_workflow(args)
    assert strict_result.partition == [2, 1, 1]
    assert strict_result.setting["strict_validation"] is True
    assert strict_result.setting["test_during_training"] is False
    assert calls["train_kwargs"]["testing"] is False


def test_st_report_includes_loss_weights():
    from scripts.st.collect_st_fold_results import markdown_table

    row = {
        "fold": 1,
        "kl_weight": 0.1,
        "bce_weight": 2.0,
        "mce_weight": 0.5,
    }
    table = markdown_table(
        ["Fold", "KL Weight", "BCE Weight", "MCE Weight"],
        [
            [
                str(int(row["fold"])),
                f"{row['kl_weight']:.4f}",
                f"{row['bce_weight']:.4f}",
                f"{row['mce_weight']:.4f}",
            ]
        ],
    )

    assert "KL Weight" in table
    assert "| 1 | 0.1000 | 2.0000 | 0.5000 |" in table


def test_mm_training_workflow_smoke(tmp_path, monkeypatch):
    from brainxdl.pipelines import mm_workflow

    num_subjects = 10
    labels = np.array([0, 1] * 5, dtype=np.int64)
    tasks = np.array(["rest"] * num_subjects, dtype=object)
    corr_fmri = np.stack([np.eye(4, dtype=np.float32) for _ in range(num_subjects)])
    pcorr_fmri = corr_fmri.copy()
    corr_eeg = np.stack([np.eye(3, dtype=np.float32) for _ in range(num_subjects)])
    pcorr_eeg = corr_eeg.copy()
    pseudo_fmri = corr_fmri.copy()
    pseudo_eeg = corr_eeg.copy()

    calls = {}

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

    def fake_prepare_data(args):
        calls["prepare_model"] = args.model
        return corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels, pseudo_fmri, pseudo_eeg, tasks

    def fake_build_model(args, n_fmri, n_eeg, model_out_dim, device):
        calls["build"] = (args.model, args.modality, args.use_fmri, args.use_eeg, n_fmri, n_eeg, model_out_dim, str(device))
        return TinyModel()

    def fake_train_one_epoch(loader, model, criterion, optimizer, device, args, **kwargs):
        calls["train_loader_len"] = len(loader.dataset)
        calls["train_kwargs"] = kwargs
        return 0.5, 0.75

    def fake_evaluate(loader, model, criterion, device, args, **kwargs):
        calls["val_loader_len"] = len(loader.dataset)
        return 0.4, 0.8

    def fake_evaluate_metrics(loader, model, criterion, device, args, **kwargs):
        calls["test_loader_len"] = len(loader.dataset)
        return {
            "loss": 0.3,
            "log_loss": 0.3,
            "acc": 0.9,
            "pre": 0.9,
            "sen": 0.9,
            "spe": 0.9,
            "f1": 0.9,
            "roc": 0.9,
        }

    def fake_analyze_fusion_gate_bias(model, dataset, args, device, all_tasks, split_name):
        calls["gate_analysis"] = (len(dataset), list(all_tasks[:2]), split_name)

    def fake_maybe_run_gnn_explanations(model, split_map, args, device, binary_mode, labels_np):
        calls["explain_splits"] = sorted(split_map)
        calls["binary_mode"] = binary_mode
        calls["labels_len"] = len(labels_np)

    monkeypatch.setattr(mm_workflow, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(mm_workflow, "build_model", fake_build_model)
    monkeypatch.setattr(mm_workflow, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(mm_workflow, "evaluate", fake_evaluate)
    monkeypatch.setattr(mm_workflow, "evaluate_metrics", fake_evaluate_metrics)
    monkeypatch.setattr(mm_workflow, "analyze_fusion_gate_bias", fake_analyze_fusion_gate_bias)
    monkeypatch.setattr(mm_workflow, "maybe_run_gnn_explanations", fake_maybe_run_gnn_explanations)

    args = argparse.Namespace(
        model="brainmgf_dynamic",
        graph_mode="dynamic",
        modality="fusion",
        seed=3,
        loss_type="ce",
        use_cache=False,
        cache_path=None,
        cache_dir=str(tmp_path / "cache"),
        plot_latents=False,
        latent_dir=str(tmp_path / "latents"),
        plot_debug=False,
        n_folds=1,
        val_ratio=0.25,
        balance_threshold=0.0,
        balance_batches=False,
        batch_size=2,
        device="cpu",
        out_channels=2,
        label_smoothing=0.0,
        class_weighted=False,
        pos_weight=None,
        lr=1e-3,
        wd=0.0,
        save_best_model=False,
        save_metric="acc",
        es_metric="loss",
        es_min_delta=0.0,
        epochs=1,
        min_epochs=1,
        no_early_stop=True,
        patience=1,
        grad_clip=0.2,
        task="REST",
        tasks=None,
        command="train_mm smoke",
        gnn_explain=False,
        latent_split="test",
    )

    mm_workflow.run_mm_training_workflow(args)

    assert args.model == "mm_brainxdl_dynamic"
    assert args.graph_mode == "dynamic"
    assert args.use_fmri is True
    assert args.use_eeg is True
    assert calls["prepare_model"] == "mm_brainxdl_dynamic"
    assert calls["build"] == ("mm_brainxdl_dynamic", "fusion", True, True, 4, 3, 2, "cpu")
    assert calls["train_loader_len"] == 6
    assert calls["val_loader_len"] == 2
    assert calls["test_loader_len"] == 2
    assert calls["train_kwargs"]["grad_clip"] == 0.2
    assert calls["gate_analysis"] == (2, ["rest", "rest"], "test")
    assert calls["explain_splits"] == ["test", "train", "val"]
    assert calls["binary_mode"] is False
    assert calls["labels_len"] == num_subjects


def test_mm_kfold_uses_participant_groups(tmp_path, monkeypatch):
    from brainxdl.mm.training import trainer
    from brainxdl.mm.models import brainmgf

    num_groups = 12
    labels = np.tile(np.array([0, 1], dtype=np.int64), num_groups)
    groups = np.repeat(np.array([f"PC{i:03d}" for i in range(num_groups)], dtype=object), 2)
    corr_fmri = torch.stack([torch.eye(4) for _ in range(labels.size)])
    corr_eeg = torch.stack([torch.eye(3) for _ in range(labels.size)])
    dataset = torch.utils.data.TensorDataset(
        corr_fmri,
        corr_eeg,
        corr_fmri.clone(),
        corr_eeg.clone(),
        torch.tensor(labels, dtype=torch.long),
        corr_fmri.clone(),
        corr_eeg.clone(),
    )
    split_groups = []

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

    def fake_build_model(args, n_fmri, n_eeg, model_out_dim, device):
        return TinyModel()

    def record_subset(loader):
        subset = loader.dataset
        assert isinstance(subset, torch.utils.data.Subset)
        return set(groups[np.asarray(subset.indices)])

    def fake_train_one_epoch(loader, model, criterion, optimizer, device, args, **kwargs):
        trainer._current_train_groups = record_subset(loader)
        return 0.5, 0.5

    def fake_evaluate(loader, model, criterion, device, args, **kwargs):
        if not hasattr(trainer, "_current_val_groups"):
            trainer._current_val_groups = []
        trainer._current_val_groups.append(record_subset(loader))
        return 0.4, 0.5

    def fake_evaluate_metrics(loader, model, criterion, device, args, **kwargs):
        train_groups = trainer._current_train_groups
        val_groups = trainer._current_val_groups.pop()
        test_groups = record_subset(loader)
        split_groups.append((train_groups, val_groups, test_groups))
        return {
            "loss": 0.3,
            "log_loss": 0.3,
            "acc": 0.9,
            "pre": 0.9,
            "sen": 0.9,
            "spe": 0.9,
            "f1": 0.9,
            "roc": 0.9,
        }

    monkeypatch.setattr(brainmgf, "build_model", fake_build_model)
    monkeypatch.setattr(trainer, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    monkeypatch.setattr(trainer, "evaluate_metrics", fake_evaluate_metrics)

    args = argparse.Namespace(
        model="mmodal_gcn",
        graph_mode="static",
        modality="fusion",
        use_fmri=True,
        use_eeg=True,
        seed=11,
        loss_type="ce",
        n_folds=3,
        val_ratio=0.25,
        balance_threshold=0.0,
        balance_batches=False,
        batch_size=2,
        device="cpu",
        out_channels=2,
        label_smoothing=0.0,
        class_weighted=False,
        pos_weight=None,
        lr=1e-3,
        wd=0.0,
        save_best_model=False,
        save_metric="acc",
        es_metric="loss",
        es_min_delta=0.0,
        epochs=1,
        min_epochs=1,
        no_early_stop=True,
        patience=1,
        grad_clip=0.0,
        task="REST",
        tasks=None,
        metrics_dir=str(tmp_path / "metrics"),
        plot_dir=str(tmp_path / "plots"),
        plot_curves=False,
        plot_latents=False,
        latent_split="test",
        command="grouped mm smoke",
    )

    trainer.run_kfold_cv(dataset, labels, args, tasks=np.array(["rest"] * labels.size, dtype=object), groups=groups)

    assert len(split_groups) == 3
    all_test_groups = []
    for train_groups, val_groups, test_groups in split_groups:
        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)
        all_test_groups.extend(test_groups)
    assert sorted(all_test_groups) == sorted(set(groups))


def test_mm_checkpoint_evaluation_workflow_smoke(tmp_path, monkeypatch):
    from brainxdl.mm.models.brainmgf import build_model
    from brainxdl.pipelines import mm_evaluation

    torch.manual_seed(13)
    device = torch.device("cpu")
    num_subjects = 4
    n_fmri = 5
    n_eeg = 4
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    corr_fmri = _symmetric_stack(num_subjects, n_fmri).numpy()
    corr_eeg = _symmetric_stack(num_subjects, n_eeg).numpy()
    pcorr_fmri = corr_fmri.copy()
    pcorr_eeg = corr_eeg.copy()
    pseudo_fmri = corr_fmri.copy()
    pseudo_eeg = corr_eeg.copy()

    def fake_prepare_data(args):
        return corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels, pseudo_fmri, pseudo_eeg, None

    monkeypatch.setattr(mm_evaluation, "prepare_data", fake_prepare_data)

    save_args = _make_mm_args(out_channels=2, loss_type="ce")
    model = build_model(save_args, n_fmri=n_fmri, n_eeg=n_eeg, model_out_dim=2, device=device)
    checkpoint_path = tmp_path / "tiny_mm_best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metric_name": "acc",
            "metric_value": 0.5,
            "metrics": {"acc": 0.5},
            "command": "train_mm smoke",
        },
        checkpoint_path,
    )

    eval_args = argparse.Namespace(
        checkpoint=str(checkpoint_path),
        fmri_path="unused",
        eeg_path="unused",
        model="mmodal_gcn",
        modality="fmri",
        loss_type="ce",
        out_channels=2,
        embid_dim=8,
        k_fmri=2,
        k_eeg=2,
        dropout=0.0,
        batch_size=2,
        device="cpu",
        seed=13,
        task="REST",
        tasks=None,
        graph_mode="static",
        window_len=50,
        window_stride=25,
        temporal_pool="mean",
        use_cache=False,
        cache_dir=str(tmp_path / "cache"),
        cache_path=None,
        gcn_hidden_dim=16,
        brainnet_channels=None,
        pos_weight=None,
        label_smoothing=0.0,
        command="eval_mm smoke",
    )

    metrics = mm_evaluation.evaluate_mm_checkpoint(eval_args)

    assert set(metrics) == {"loss", "log_loss", "acc", "pre", "sen", "spe", "f1", "roc"}
    assert np.isfinite(metrics["loss"])
    assert 0.0 <= metrics["acc"] <= 1.0
    assert eval_args.use_fmri is True
    assert eval_args.use_eeg is False
