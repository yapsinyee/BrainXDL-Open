import numpy as np
import pytest

from brainxdl.common.metrics import (
    binary_threshold_candidates,
    choose_binary_threshold,
    compute_classification_metrics,
)


@pytest.mark.parametrize("objective", ["balanced_accuracy", "f1"])
def test_choose_binary_threshold_maximizes_validation_objective(objective):
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.10, 0.30, 0.55, 0.40, 0.65, 0.90])

    threshold, selected_value = choose_binary_threshold(labels, scores, objective)

    candidate_values = []
    for candidate in binary_threshold_candidates(scores):
        predictions = (scores >= candidate).astype(int)
        metrics = compute_classification_metrics(labels, predictions, scores)
        value = (
            metrics["f1"]
            if objective == "f1"
            else (metrics["sen"] + metrics["spe"]) / 2.0
        )
        candidate_values.append(value)
    assert selected_value == pytest.approx(max(candidate_values))
    assert threshold in binary_threshold_candidates(scores)


def test_choose_binary_threshold_tie_prefers_default_boundary():
    labels = np.asarray([0, 1])
    scores = np.asarray([0.25, 0.75])
    threshold, value = choose_binary_threshold(labels, scores, "balanced_accuracy")
    assert threshold == pytest.approx(0.5)
    assert value == pytest.approx(1.0)


def test_choose_binary_threshold_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="labels 0 and 1"):
        choose_binary_threshold(
            np.asarray([0, 2]), np.asarray([0.2, 0.8]), "balanced_accuracy"
        )
