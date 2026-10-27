"""Brain connectivity matrix utilities shared across ST and MM modules.

Provides numpy-level functions for computing correlation / partial
correlation matrices from fMRI or EEG time-series, plus a PyTorch
k-NN thresholding helper for use inside model forward passes.

Sources: consolidated from
  legacy_code/2-MM/main.py  (_compute_correlation, _compute_partial_correlation,
                              stabilise_timeseries, generate_instructor_topology,
                              knn_thresholding)
  legacy_code/1-ST/step1_compute_ldw.py  (cov2corr)
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Numpy-level preprocessing
# ---------------------------------------------------------------------------

def stabilise_timeseries(timeseries: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Ensure every feature column has non-zero variance.

    Features with std < *eps* receive a tiny amount of N(0, eps) noise so
    that downstream covariance estimates remain well-conditioned.

    Args:
        timeseries: 2-D array of shape (timepoints, features).
        eps:        Variance floor; features below this threshold are jittered.

    Returns:
        Copy of the input with low-variance features stabilised.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    ts = np.asarray(timeseries, dtype=np.float32)
    if ts.ndim != 2:
        raise ValueError("timeseries must be 2-D with shape (timepoints, features)")

    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    std = ts.std(axis=0)
    low_var_mask = std < eps
    if low_var_mask.any():
        ts = ts.copy()
        rng = np.random.default_rng(0)
        noise = rng.normal(
            loc=0.0, scale=eps, size=(ts.shape[0], int(low_var_mask.sum()))
        ).astype(np.float32)
        ts[:, low_var_mask] += noise
    return ts


def compute_correlation(timeseries: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute the Pearson correlation matrix from a time-series array.

    Args:
        timeseries: 2-D array, shape (timepoints, features / ROIs).
        eps:        Small constant added to the standard deviation to avoid
                    division by zero.

    Returns:
        Symmetric correlation matrix, shape (features, features), clipped
        to [-1, 1].

    Source: extracted from legacy_code/2-MM/main.py (_compute_correlation).
    """
    ts = timeseries - timeseries.mean(axis=0, keepdims=True)
    std = timeseries.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    ts = ts / std
    cov = np.dot(ts.T, ts) / max(ts.shape[0] - 1, 1)
    return np.clip(cov, -1.0, 1.0)


def compute_partial_correlation(
    timeseries: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """Compute the partial correlation matrix via precision matrix inversion.

    A small ridge *eps* is added to the covariance diagonal before inversion
    to keep the matrix numerically stable.

    Args:
        timeseries: 2-D array, shape (timepoints, features / ROIs).
        eps:        Ridge regularisation added to the covariance diagonal.

    Returns:
        Symmetric partial correlation matrix, shape (features, features),
        clipped to [-1, 1].

    Source: extracted from legacy_code/2-MM/main.py (_compute_partial_correlation).
    """
    ts = timeseries - timeseries.mean(axis=0, keepdims=True)
    cov = np.dot(ts.T, ts) / max(ts.shape[0] - 1, 1)
    cov += eps * np.eye(cov.shape[0], dtype=cov.dtype)
    precision = np.linalg.pinv(cov)
    d = np.sqrt(np.clip(np.diag(precision), eps, None))
    denom = np.outer(d, d)
    partial = -precision / denom
    np.fill_diagonal(partial, 1.0)
    return np.clip(partial, -1.0, 1.0)


def generate_connectivity_matrix(
    timeseries: np.ndarray, corr_type: str = "correlation"
) -> np.ndarray:
    """Build a connectivity matrix from a raw time-series array.

    Runs stabilise_timeseries first, then computes the requested matrix
    type, symmetrises the result, fills the diagonal with 1, and replaces
    any NaN / Inf with 0.

    Args:
        timeseries: 2-D array, shape (timepoints, features / ROIs).
        corr_type:  One of "correlation" or "partial correlation".

    Returns:
        Float32 connectivity matrix, shape (features, features).

    Source: extracted from legacy_code/2-MM/main.py (generate_instructor_topology).
    """
    stable_ts = stabilise_timeseries(timeseries)
    if corr_type == "correlation":
        matrix = compute_correlation(stable_ts)
    elif corr_type == "partial correlation":
        matrix = compute_partial_correlation(stable_ts)
    else:
        raise ValueError(f"Unsupported corr_type: '{corr_type}'. "
                         "Expected 'correlation' or 'partial correlation'.")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    return matrix.astype(np.float32)


def cov2corr(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix (NumPy).

    Args:
        covariance: Square covariance matrix, shape (N, N).

    Returns:
        Correlation matrix of the same shape.  Entries where the
        covariance is zero are set to zero rather than producing NaN.

    Source: extracted from legacy_code/1-ST/step1_compute_ldw.py.
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


# ---------------------------------------------------------------------------
# PyTorch-level adjacency thresholding (used inside model forward passes)
# ---------------------------------------------------------------------------

def knn_thresholding(adjacency: torch.Tensor, k: int = 40) -> torch.Tensor:
    """Keep only the top-k edges per node in a batched adjacency tensor.

    Args:
        adjacency: Dense adjacency matrix, shape (batch, nodes, nodes).
        k:         Number of strongest connections to retain per node.

    Returns:
        Thresholded adjacency of the same shape.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    batch_size, num_nodes, _ = adjacency.shape
    k = max(1, min(k, num_nodes))
    flat = adjacency.view(-1, num_nodes)
    _, topk_indices = torch.topk(flat, k, dim=1)
    mask = torch.zeros_like(flat)
    mask.scatter_(1, topk_indices, 1.0)
    thresholded = flat * mask
    return thresholded.view(batch_size, num_nodes, num_nodes)
