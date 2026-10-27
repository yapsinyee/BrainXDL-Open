from pathlib import Path
from argparse import Namespace

import numpy as np
import pytest
import torch
from scipy.io import savemat
from sklearn.covariance import LedoitWolf

from brainxdl.st.artifacts import condition_name, resolve_checkpoint, split_metadata
from brainxdl.st.checkpoints import load_model
from brainxdl.st.psiconnect import (
    get_grouped_outer_fold_indices,
    load_psiconnect_timeseries,
    prepare_psiconnect_st,
)
from brainxdl.st.preprocessing import (
    extract_ldw_corr,
    extract_ldw_corr_torch,
    ledoit_wolf_covariance_torch,
)
from scripts.st.train_static_aal_outer5 import load_source_dataset


def test_public_st_artifact_helpers_validate_grouped_splits(tmp_path):
    metadata = {
        "sample_ids": ["s1a", "s1b", "s2a", "s2b", "s3a", "s3b"],
        "participant_groups": ["p1", "p1", "p2", "p2", "p3", "p3"],
        "split_indices": {"train": [0, 1], "val": [2, 3], "test": [4, 5]},
    }
    sample_ids, groups, indices = split_metadata(metadata, fold=1)
    assert sample_ids == metadata["sample_ids"]
    assert groups == metadata["participant_groups"]
    assert indices["test"].tolist() == [4, 5]
    assert condition_name({"1": "psilocybin (ses-01)"}, 1) == "psilocybin"
    assert callable(load_model)

    checkpoint = tmp_path / "checkpoints" / "model.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    assert resolve_checkpoint(tmp_path / "metrics.json", {}) == checkpoint.resolve()


def test_public_st_artifact_helpers_reject_participant_leakage():
    metadata = {
        "sample_ids": ["s1a", "s1b", "s2a", "s2b"],
        "participant_groups": ["p1", "p1", "p2", "p2"],
        "split_indices": {"train": [0], "val": [1], "test": [2, 3]},
    }
    with pytest.raises(ValueError, match="participant leakage"):
        split_metadata(metadata, fold=1)


def _write_timeseries(root: Path, subject: str, session: str, value: float) -> None:
    directory = root / f"sub-{subject}" / f"ses-{session}" / "task-rest"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"sub-{subject}_ses-{session}_task-rest_run-1_"
        "atlas-Schaefer4Parcels7Networks_timeseries-tedanaGLM.mat"
    )
    rng = np.random.default_rng(int(subject) * 10 + int(session))
    savemat(path, {"time_series": rng.normal(value, 1.0, size=(8, 4))})


def _tiny_dataset(root: Path, participants: int = 10) -> None:
    for index in range(1, participants + 1):
        subject = f"{index:03d}"
        _write_timeseries(root, subject, "01", 1.0)
        _write_timeseries(root, subject, "02", 0.0)


def test_load_psiconnect_timeseries_uses_mm_session_labels(tmp_path):
    _tiny_dataset(tmp_path, participants=2)
    data, labels, groups, sample_ids = load_psiconnect_timeseries(
        tmp_path, expected_rois=4
    )
    assert len(data) == 4
    assert labels.tolist() == [1, 0, 1, 0]
    assert groups.tolist() == ["001", "001", "002", "002"]
    assert sample_ids[0] == "sub-001_ses-01_task-rest"
    assert all(item.shape == (8, 4) for item in data)


def test_load_psiconnect_timeseries_supports_flat_mm_layout(tmp_path):
    for subject, session in (("SYN001", "01"), ("SYN001", "02")):
        rng = np.random.default_rng(int(session))
        savemat(
            tmp_path
            / (
                f"sub-{subject}_ses-{session}_task-rest_run-1_"
                "atlas-Schaefer4Parcels7Networks_timeseries-tedanaGLM.mat"
            ),
            {"time_series": rng.normal(size=(8, 4))},
        )

    data, labels, groups, sample_ids = load_psiconnect_timeseries(
        tmp_path,
        pattern="*_task-rest_*_timeseries-tedanaGLM.mat",
        expected_rois=4,
    )

    assert len(data) == 2
    assert labels.tolist() == [1, 0]
    assert groups.tolist() == ["SYN001", "SYN001"]
    assert sample_ids == [
        "sub-SYN001_ses-01_task-rest",
        "sub-SYN001_ses-02_task-rest",
    ]


def test_load_psiconnect_timeseries_can_match_mm_eeg_cohort(tmp_path):
    source = tmp_path / "timeseries"
    eeg = tmp_path / "eeg"
    eeg.mkdir()
    _tiny_dataset(source, participants=2)
    (eeg / "sub-001_ses-01_task-rest_Clean-ft.mat").touch()
    (eeg / "sub-001_ses-02_task-rest_Clean-ft.mat").touch()

    _, labels, groups, _ = load_psiconnect_timeseries(
        source, expected_rois=4, matched_eeg_root=eeg
    )

    assert labels.tolist() == [1, 0]
    assert groups.tolist() == ["001", "001"]


def test_grouped_outer_folds_keep_sessions_together():
    labels = np.tile([1, 0], 10)
    groups = np.repeat([f"PC{i:03d}" for i in range(10)], 2)
    folds = get_grouped_outer_fold_indices(labels, groups, n_splits=5)
    assert set(folds) == {f"outer{i}_inner1" for i in range(1, 6)}
    all_test_groups = []
    for split_indices in folds.values():
        split_groups = {
            split: set(groups[indices]) for split, indices in split_indices.items()
        }
        assert split_groups["train"].isdisjoint(split_groups["val"])
        assert split_groups["train"].isdisjoint(split_groups["test"])
        assert split_groups["val"].isdisjoint(split_groups["test"])
        all_test_groups.extend(split_groups["test"])
    assert sorted(all_test_groups) == sorted(np.unique(groups))


def test_static_outer5_psiconnect_source_uses_grouped_folds(tmp_path):
    source = tmp_path / "timeseries"
    eeg = tmp_path / "eeg"
    eeg.mkdir()
    _tiny_dataset(source, participants=10)
    for index in range(1, 11):
        subject = f"{index:03d}"
        for session in ("01", "02"):
            (eeg / f"sub-{subject}_ses-{session}_task-rest_Clean-ft.mat").touch()

    args = Namespace(
        source="psiconnect",
        psiconnect_timeseries_root=str(source),
        psiconnect_pattern=(
            "sub-*/ses-*/task-rest/"
            "*_task-rest_*_timeseries-tedanaGLM.mat"
        ),
        psiconnect_expected_rois=4,
        psiconnect_matched_eeg_root=str(eeg),
        seed=42,
    )
    timeseries, labels, label_map, folds = load_source_dataset(args)

    assert len(timeseries) == 20
    assert labels.tolist() == [value for _ in range(10) for value in (1, 0)]
    assert label_map == {0: "control", 1: "psilocybin"}
    assert set(folds) == {f"outer{i}_inner1" for i in range(1, 6)}
    groups = np.repeat([f"{index:03d}" for index in range(1, 11)], 2)
    for split in folds.values():
        train_groups = set(groups[split["train"]])
        val_groups = set(groups[split["val"]])
        test_groups = set(groups[split["test"]])
        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)


def test_torch_ledoit_wolf_matches_sklearn():
    samples = np.random.default_rng(42).normal(size=(20, 12))
    expected = LedoitWolf(assume_centered=False).fit(samples).covariance_
    actual = ledoit_wolf_covariance_torch(samples, device="cpu").numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_torch_ldw_graphs_match_established_cpu_path():
    samples = [np.random.default_rng(42).normal(size=(12, 8))]
    expected_features, expected_adjacency, expected_windows = extract_ldw_corr(
        samples, w_size=4, shift=2, threshold_p=0.40
    )
    actual_features, actual_adjacency, actual_windows = extract_ldw_corr_torch(
        samples, w_size=4, shift=2, threshold_p=0.40, device="cpu"
    )
    assert actual_windows == expected_windows
    np.testing.assert_allclose(actual_features, expected_features, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_adjacency, expected_adjacency, rtol=1e-12, atol=1e-12)


def test_prepare_psiconnect_st_writes_loadable_fold_files(tmp_path):
    source = tmp_path / "timeseries"
    output = tmp_path / "folds"
    _tiny_dataset(source, participants=6)
    manifest = prepare_psiconnect_st(
        source,
        output,
        expected_rois=4,
        window_size=4,
        shift=2,
        threshold_p=0.40,
        n_splits=3,
        device="cpu",
    )
    assert manifest["n_samples"] == 12
    assert manifest["n_participants"] == 6
    assert manifest["class_counts"] == {"0": 6, "1": 6}
    assert (output / "LDW_psiconnect_data.pkl").is_file()
    assert (output / "psiconnect_st_manifest.json").is_file()

    payload = torch.load(output / "graphs_outer1_inner1.pkl", weights_only=False)
    assert set(payload) == {
        "train_graphs", "val_graphs", "test_graphs", "metadata"
    }
    assert payload["train_graphs"].shape[1] == 3
    groups = np.asarray(payload["metadata"]["participant_groups"])
    indices = payload["metadata"]["split_indices"]
    train_groups = set(groups[indices["train"]])
    val_groups = set(groups[indices["val"]])
    test_groups = set(groups[indices["test"]])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
