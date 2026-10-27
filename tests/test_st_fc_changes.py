import numpy as np

from brainxdl.xai.st_fc_changes import (
    benjamini_hochberg,
    cohens_dz,
    fit_random_intercept_condition_model,
    paired_values,
)


def test_benjamini_hochberg_preserves_order_and_monotonicity() -> None:
    values = np.asarray([0.04, 0.001, 0.03, 0.20])
    adjusted = benjamini_hochberg(values)
    np.testing.assert_allclose(adjusted, [0.0533333333, 0.004, 0.0533333333, 0.2])


def test_paired_values_and_cohens_dz_use_psilocybin_minus_control() -> None:
    control, psilocybin, subjects = paired_values(
        [1.0, 3.0, 2.0, 5.0, 9.0],
        [0, 1, 0, 1, 1],
        ["A", "A", "B", "B", "C"],
    )
    assert subjects == ["A", "B"]
    np.testing.assert_array_equal(control, [1.0, 2.0])
    np.testing.assert_array_equal(psilocybin, [3.0, 5.0])
    assert np.isclose(cohens_dz(control, psilocybin), 2.5 / np.sqrt(0.5))


def test_random_intercept_model_recovers_positive_condition_effect() -> None:
    rng = np.random.default_rng(42)
    groups, conditions, values = [], [], []
    for index in range(40):
        random_intercept = rng.normal(0, 0.8)
        for condition in (0, 1):
            groups.append(f"S{index:02d}")
            conditions.append(condition)
            values.append(1.0 + random_intercept + 0.5 * condition + rng.normal(0, 0.15))
    result = fit_random_intercept_condition_model(values, conditions, groups)
    assert result["converged"] is True
    assert 0.40 < result["condition_effect"] < 0.60
    assert result["p"] < 0.001
