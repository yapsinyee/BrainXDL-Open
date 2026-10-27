"""Statistics for condition differences in attention-selected dynamic FC edges."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import optimize, stats


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("p_values must be a non-empty finite one-dimensional array.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must lie in [0, 1].")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def paired_values(
    values: Sequence[float], conditions: Sequence[int], groups: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return control/psilocybin vectors for groups observed in both conditions."""
    y = np.asarray(values, dtype=float)
    condition = np.asarray(conditions, dtype=np.int64)
    group = np.asarray(groups, dtype=str)
    if y.shape != condition.shape or y.shape != group.shape:
        raise ValueError("values, conditions, and groups must have identical shapes.")
    pairs: list[tuple[str, float, float]] = []
    for subject in sorted(set(group.tolist())):
        mask = group == subject
        control = y[mask & (condition == 0)]
        psilocybin = y[mask & (condition == 1)]
        if control.size > 1 or psilocybin.size > 1:
            raise ValueError(f"Participant {subject} has duplicate condition observations.")
        if control.size == 1 and psilocybin.size == 1:
            pairs.append((subject, float(control[0]), float(psilocybin[0])))
    return (
        np.asarray([row[1] for row in pairs], dtype=float),
        np.asarray([row[2] for row in pairs], dtype=float),
        [row[0] for row in pairs],
    )


def cohens_dz(control: Sequence[float], psilocybin: Sequence[float]) -> float:
    """Cohen's dz for psilocybin-minus-control paired differences."""
    control_array = np.asarray(control, dtype=float)
    psilocybin_array = np.asarray(psilocybin, dtype=float)
    if control_array.shape != psilocybin_array.shape or control_array.size < 2:
        raise ValueError("Paired vectors must have the same length and at least two values.")
    difference = psilocybin_array - control_array
    standard_deviation = float(difference.std(ddof=1))
    if standard_deviation == 0:
        return float(np.sign(difference.mean()) * np.inf) if difference.mean() else 0.0
    return float(difference.mean() / standard_deviation)


def bootstrap_paired_dz_ci(
    control: Sequence[float],
    psilocybin: Sequence[float],
    *,
    repetitions: int = 5000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for paired Cohen's dz."""
    control_array = np.asarray(control, dtype=float)
    psilocybin_array = np.asarray(psilocybin, dtype=float)
    if control_array.shape != psilocybin_array.shape or control_array.size < 2:
        raise ValueError("Paired vectors must have the same length and at least two values.")
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required.")
    differences = psilocybin_array - control_array
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sample = differences[rng.integers(0, len(differences), len(differences))]
        sd = float(sample.std(ddof=1))
        if sd > 0:
            estimates.append(float(sample.mean() / sd))
    if len(estimates) < repetitions * 0.95:
        raise ValueError("Too many undefined bootstrap effect sizes.")
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def cluster_bootstrap_mean_difference_ci(
    values: Sequence[float],
    conditions: Sequence[int],
    groups: Sequence[str],
    *,
    repetitions: int = 5000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Bootstrap the raw mean difference by resampling participant clusters."""
    y = np.asarray(values, dtype=float)
    condition = np.asarray(conditions, dtype=np.int64)
    group = np.asarray(groups, dtype=str)
    subjects = np.asarray(sorted(set(group.tolist())), dtype=str)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    by_subject = [(y[group == subject], condition[group == subject]) for subject in subjects]
    for repetition in range(repetitions):
        sampled = rng.integers(0, len(subjects), len(subjects))
        sample_y = np.concatenate([by_subject[index][0] for index in sampled])
        sample_condition = np.concatenate([by_subject[index][1] for index in sampled])
        estimates[repetition] = (
            sample_y[sample_condition == 1].mean()
            - sample_y[sample_condition == 0].mean()
        )
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def _gls_components(
    y: np.ndarray,
    x: np.ndarray,
    group_indices: list[np.ndarray],
    residual_variance: float,
    random_intercept_variance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    xt_vinv_x = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    xt_vinv_y = np.zeros(x.shape[1], dtype=float)
    y_vinv_y = 0.0
    log_determinant = 0.0
    for indices in group_indices:
        group_x = x[indices]
        group_y = y[indices]
        size = len(indices)
        inverse = (
            np.eye(size) / residual_variance
            - random_intercept_variance
            * np.ones((size, size))
            / (residual_variance * (residual_variance + size * random_intercept_variance))
        )
        xt_vinv_x += group_x.T @ inverse @ group_x
        xt_vinv_y += group_x.T @ inverse @ group_y
        y_vinv_y += float(group_y.T @ inverse @ group_y)
        log_determinant += (size - 1) * np.log(residual_variance) + np.log(
            residual_variance + size * random_intercept_variance
        )
    return xt_vinv_x, xt_vinv_y, y_vinv_y, float(log_determinant)


def fit_random_intercept_condition_model(
    values: Sequence[float], conditions: Sequence[int], groups: Sequence[str]
) -> dict[str, float | bool]:
    """Fit y ~ condition + (1|participant) by REML without external LMM packages.

    The condition coefficient is psilocybin minus control. A t reference with
    participant-level residual degrees of freedom is used for conservative
    inference in this two-condition, partially paired design.
    """
    y = np.asarray(values, dtype=float)
    condition = np.asarray(conditions, dtype=np.int64)
    group = np.asarray(groups, dtype=str)
    if y.ndim != 1 or y.shape != condition.shape or y.shape != group.shape:
        raise ValueError("values, conditions, and groups must be equal-length vectors.")
    if not np.isfinite(y).all() or set(np.unique(condition)) != {0, 1}:
        raise ValueError("The model requires finite values and both binary conditions.")
    subjects = sorted(set(group.tolist()))
    if len(subjects) < 4:
        raise ValueError("At least four participant groups are required.")
    group_indices = [np.flatnonzero(group == subject) for subject in subjects]
    if any(len(indices) > 2 for indices in group_indices):
        raise ValueError("This implementation expects at most two observations per participant.")
    x = np.column_stack((np.ones(len(y)), condition.astype(float)))
    response_variance = max(float(y.var(ddof=1)), 1e-8)

    def objective(log_variances: np.ndarray) -> float:
        residual_variance, random_variance = np.exp(log_variances)
        xtx, xty, yvy, logdet = _gls_components(
            y, x, group_indices, residual_variance, random_variance
        )
        sign, logdet_xtx = np.linalg.slogdet(xtx)
        if sign <= 0:
            return float("inf")
        beta = np.linalg.solve(xtx, xty)
        residual_quadratic = max(float(yvy - beta.T @ xty), np.finfo(float).tiny)
        return 0.5 * (
            logdet
            + logdet_xtx
            + residual_quadratic
            + (len(y) - x.shape[1]) * np.log(2.0 * np.pi)
        )

    starts = (
        np.log([response_variance * 0.5, response_variance * 0.5]),
        np.log([response_variance * 0.8, response_variance * 0.2]),
        np.log([response_variance * 0.2, response_variance * 0.8]),
    )
    fits = [
        optimize.minimize(objective, start, method="L-BFGS-B", bounds=[(-20, 5), (-20, 5)])
        for start in starts
    ]
    fit = min(fits, key=lambda result: float(result.fun))
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Random-intercept model did not converge: {fit.message}")
    residual_variance, random_variance = np.exp(fit.x)
    xtx, xty, _, _ = _gls_components(
        y, x, group_indices, residual_variance, random_variance
    )
    beta = np.linalg.solve(xtx, xty)
    covariance = np.linalg.inv(xtx)
    standard_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = float(len(subjects) - x.shape[1])
    statistic = float(beta[1] / standard_error)
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=degrees_freedom))
    critical = float(stats.t.ppf(0.975, df=degrees_freedom))
    return {
        "intercept": float(beta[0]),
        "condition_effect": float(beta[1]),
        "standard_error": standard_error,
        "ci_low": float(beta[1] - critical * standard_error),
        "ci_high": float(beta[1] + critical * standard_error),
        "t": statistic,
        "df": degrees_freedom,
        "p": p_value,
        "residual_variance": float(residual_variance),
        "random_intercept_variance": float(random_variance),
        "converged": bool(fit.success),
    }
