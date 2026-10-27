import numpy as np
import pytest

from brainxdl.pipelines.mm_workflow import validate_participant_grouping


def test_public_workflow_rejects_missing_participant_groups():
    labels = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="Participant identifiers are required"):
        validate_participant_grouping(labels, None, n_folds=5, required=True)


def test_public_workflow_rejects_single_split_for_repeated_participants():
    labels = np.array([0, 1, 0, 1])
    groups = np.array(["p1", "p1", "p2", "p2"])
    with pytest.raises(ValueError, match="grouped cross-validation"):
        validate_participant_grouping(labels, groups, n_folds=1, required=True)


def test_public_workflow_accepts_grouped_cross_validation():
    labels = np.array([0, 1, 0, 1])
    groups = np.array(["p1", "p1", "p2", "p2"])
    checked = validate_participant_grouping(labels, groups, n_folds=2, required=True)
    assert np.array_equal(checked, groups)
