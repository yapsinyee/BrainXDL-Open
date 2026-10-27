"""Regression tests for the corrected participant-grouped MM benchmark."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from brainxdl.mm.participant_grouped_benchmark import (
    BenchmarkData,
    PyGGATClassifier,
    atomic_write_csv,
    build_neural_model,
    confusion_counts,
    metric_dict,
    validate_manifest,
)


def _synthetic_grouped_manifest():
    participants = [f"p{i:02d}" for i in range(10)]
    sample_ids = []
    groups = []
    for participant in participants:
        sample_ids.extend([
            f"sub-{participant}_ses-01_task-rest",
            f"sub-{participant}_ses-02_task-rest",
        ])
        groups.extend([participant, participant])

    folds = {}
    for fold in range(5):
        test_participants = {participants[2 * fold], participants[2 * fold + 1]}
        val_participants = {
            participants[(2 * fold + 2) % 10],
            participants[(2 * fold + 3) % 10],
        }
        split_participants = {
            "test": test_participants,
            "val": val_participants,
            "train": set(participants) - test_participants - val_participants,
        }
        sections = {}
        for split, selected in split_participants.items():
            indices = [index for index, group in enumerate(groups) if group in selected]
            sections[split] = {
                "indices": indices,
                "sample_ids": [sample_ids[index] for index in indices],
                "participant_groups": [groups[index] for index in indices],
            }
        folds[f"outer{fold + 1}_inner1"] = sections
    return {"n_samples": len(sample_ids), "folds": folds}


def test_verified_manifest_is_participant_disjoint_in_every_fold():
    audit = validate_manifest(_synthetic_grouped_manifest(), enforce_expected_cohort=False)
    assert audit["cohort"]["n_observations"] == 20
    assert audit["cohort"]["n_participants"] == 10
    assert audit["all_checks_passed"] is True
    for fold in audit["folds"].values():
        assert fold["participant_intersections"] == {"train_val": [], "train_test": [], "val_test": []}


def test_manifest_alignment_mutation_fails_closed():
    manifest = _synthetic_grouped_manifest()
    corrupted = copy.deepcopy(manifest)
    corrupted["folds"]["outer1_inner1"]["test"]["participant_groups"][0] = "sub-leaked"
    with pytest.raises(ValueError, match="participant IDs are not index-aligned"):
        validate_manifest(corrupted, enforce_expected_cohort=False)


def test_metric_suite_includes_balanced_accuracy_and_log_loss():
    metrics = metric_dict(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.7, 0.8, 0.9]), 0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert set(metrics) == {
        "accuracy", "balanced_accuracy", "roc_auc", "f1", "precision",
        "sensitivity", "specificity", "log_loss",
    }


def test_confusion_counts_use_negative_then_positive_label_semantics():
    counts = confusion_counts(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 1]))
    assert counts == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_gat_forward_shape_and_finite_values():
    pytest.importorskip("torch_geometric")
    model = PyGGATClassifier(num_nodes=6, hidden_dim=4, dropout=0.0, k=3)
    pearson = torch.randn(2, 6, 6)
    pcorr = torch.randn(2, 6, 6)
    pseudo = torch.eye(6).repeat(2, 1, 1)
    output = model(pearson, pcorr, pseudo)
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


def test_atomic_csv_supports_mixed_comparison_row_schemas(tmp_path):
    destination = tmp_path / "mixed.csv"
    atomic_write_csv(destination, [{"kind": "fold", "delta": 0.1}, {"kind": "aggregate", "n": 5}])
    assert destination.read_text(encoding="utf-8").splitlines() == [
        "kind,delta,n",
        "fold,0.1,",
        "aggregate,,5",
    ]


def _synthetic_benchmark_data() -> BenchmarkData:
    rng = np.random.default_rng(11)

    def symmetric(n_samples: int, n_nodes: int) -> np.ndarray:
        raw = rng.normal(size=(n_samples, n_nodes, n_nodes)).astype(np.float32)
        return (raw + raw.transpose(0, 2, 1)) / 2

    fmri = symmetric(4, 7)
    eeg = symmetric(4, 5)
    return BenchmarkData(
        corr_fmri=fmri,
        pcorr_fmri=fmri,
        corr_eeg=eeg,
        pcorr_eeg=eeg,
        pseudo_fmri=np.repeat(np.eye(7, dtype=np.float32)[None], 4, axis=0),
        pseudo_eeg=np.repeat(np.eye(5, dtype=np.float32)[None], 4, axis=0),
        labels=np.asarray([0, 1, 0, 1]),
        groups=np.asarray(["1", "1", "2", "2"]),
        tasks=np.asarray(["rest"] * 4),
        sample_ids=np.asarray([f"sample-{i}" for i in range(4)]),
        sessions=np.asarray(["ses-02", "ses-01", "ses-02", "ses-01"]),
    )


def test_eeg_gcn_is_dimension_agnostic():
    data = _synthetic_benchmark_data()
    model = build_neural_model("gcn_eeg", data, input_dim=0)
    output = model(
        torch.from_numpy(data.corr_eeg),
        torch.from_numpy(data.pcorr_eeg),
        torch.from_numpy(data.pseudo_eeg),
    )
    assert output.shape == (4, 2)
    assert torch.isfinite(output).all()
