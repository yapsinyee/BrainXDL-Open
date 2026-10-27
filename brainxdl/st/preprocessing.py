"""ST-BrainXDL: Step 1 preprocessing — Ledoit-Wolf covariance & adjacency matrices.

Runs as a standalone script (python -m brainxdl.st.preprocessing) or is
imported by the data preparation pipeline.

Scientific behaviour is unchanged from legacy_code/1-ST/step1_compute_ldw.py.

Changes from legacy:
- cov2corr imported from brainxdl.common.connectivity (identical implementation).
- Data paths are parameters rather than hardcoded strings.
- __main__ block uses argparse so paths can be overridden without editing.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

from brainxdl.common.connectivity import cov2corr


def load_timeseries(
    data_dir: str = "./data/timeseries",
    asd_file: str = "power_asd.npy",
    td_file: str = "power_td.npy",
    n_rois: int = 264,
):
    """Load raw fMRI time-series and remove subjects with missing ROIs.

    Args:
        data_dir: Directory containing the .npy time-series files.
        asd_file: Filename for the ASD group (label = 1).
        td_file:  Filename for the TD group (label = 0).
        n_rois:   Expected number of ROIs (264 for Power atlas).

    Returns:
        (data, labels): data is a list of np.ndarray (timepoints × ROIs),
                        labels is int np.ndarray (1 = ASD, 0 = TD).

    Note: formerly 'load_data' in step1_compute_ldw.py. Renamed to avoid
    clashing with the identically-named function in data_prep.py.
    """
    asd_data = np.load(os.path.join(data_dir, asd_file), allow_pickle=True)
    td_data  = np.load(os.path.join(data_dir, td_file),  allow_pickle=True)

    data   = np.concatenate((asd_data, td_data))
    labels = np.concatenate(
        (np.ones(len(asd_data)), np.zeros(len(td_data)))
    ).astype(int)

    to_remove = []
    for i, x in enumerate(data):
        if x.shape[1] == n_rois:
            if np.any(np.all(x == 0, axis=0)):
                to_remove.append(i)
                print(f"Subject {i} removed: missing column ROI(s).")
        else:
            if np.any(np.all(x == 0, axis=1)):
                to_remove.append(i)
                print(f"Subject {i} removed: missing row ROI(s).")

    if to_remove:
        data   = np.delete(data,   to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)

    return data, labels


def load_abide_aal_timeseries(
    rois_dir: str = "./data/nilearn/ABIDE_pcp/cpac/nofilt_noglobal",
    phenotypic_csv: str = "./data/manifests/abide_rois_aal_phenotypic.csv",
    pattern: str = "*_rois_aal.1D",
):
    """Load Nilearn ABIDE PCP AAL ROI time-series and DX labels.

    Args:
        rois_dir: Directory containing Nilearn ``*_rois_aal.1D`` files.
        phenotypic_csv: CSV manifest produced by scripts/st/download_abide_aal.py.
        pattern: Glob pattern for AAL ROI time-series files.

    Returns:
        (data, labels): data is a list of np.ndarray (timepoints x 116),
                        labels use the ST convention (ASD=1, control=0).
    """
    rois_paths = sorted(Path(rois_dir).glob(pattern))
    if not rois_paths:
        raise FileNotFoundError(f"No AAL ROI files matching {pattern!r} in {rois_dir}")

    pheno = pd.read_csv(phenotypic_csv)
    if "FILE_ID" not in pheno.columns or "DX_GROUP" not in pheno.columns:
        raise ValueError(
            "Phenotypic CSV must contain FILE_ID and DX_GROUP columns."
        )

    labels_by_file_id = {
        str(row.FILE_ID): 1 if int(row.DX_GROUP) == 1 else 0
        for row in pheno.itertuples(index=False)
    }

    data = []
    labels = []
    missing_labels = []
    for path in rois_paths:
        file_id = path.name.removesuffix("_rois_aal.1D")
        if file_id not in labels_by_file_id:
            missing_labels.append(file_id)
            continue
        data.append(np.loadtxt(path))
        labels.append(labels_by_file_id[file_id])

    if missing_labels:
        raise ValueError(
            "Missing phenotypic labels for ROI files: "
            + ", ".join(missing_labels[:10])
        )

    return data, np.asarray(labels, dtype=int)


def _normalise_cobre_subject_id(value: str) -> str:
    """Normalise COBRE subject/file labels for matching metadata to ROI files."""
    subject_id = Path(str(value)).name.strip()
    for suffix in (".nii.gz", ".nii", ".1D", ".csv", ".mat"):
        if subject_id.endswith(suffix):
            subject_id = subject_id[: -len(suffix)]
    for suffix in ("_rois_aal", "_extra", "_session1_run1"):
        if subject_id.endswith(suffix):
            subject_id = subject_id[: -len(suffix)]
    if subject_id.startswith("fmri_"):
        subject_id = subject_id[len("fmri_"):]
    return subject_id


def load_cobre_aal_timeseries(
    rois_dir: str = "./data/cobre/rois_aal",
    phenotypic_csv: str = "./data/cobre/metadata/cobre_model_group.csv",
    pattern: str = "*_rois_aal.1D",
    subject_column: str | None = None,
    label_column: str = "sz",
    expected_rois: int = 116,
):
    """Load COBRE AAL ROI time-series and schizophrenia/control labels.

    Args:
        rois_dir: Directory containing COBRE ``*_rois_aal.1D`` files.
        phenotypic_csv: COBRE group CSV from the Figshare release.
        pattern: Glob pattern for extracted AAL ROI time-series files.
        subject_column: Optional column holding subject/file labels. If omitted,
            the first CSV column is used.
        label_column: CSV column with schizophrenia labels (1=SZ, 0=control).
        expected_rois: Expected AAL ROI count.

    Returns:
        (data, labels): data is a list of np.ndarray (timepoints x 116),
                        labels use 1=schizophrenia, 0=control.
    """
    rois_paths = sorted(Path(rois_dir).glob(pattern))
    if not rois_paths:
        raise FileNotFoundError(f"No COBRE AAL ROI files matching {pattern!r} in {rois_dir}")

    pheno = pd.read_csv(phenotypic_csv)
    pheno.columns = [str(column).strip() for column in pheno.columns]
    if subject_column is not None:
        subject_column = subject_column.strip()
    label_column = label_column.strip()
    if label_column not in pheno.columns:
        raise ValueError(f"COBRE phenotypic CSV must contain {label_column!r}.")

    if subject_column is None:
        subject_column = pheno.columns[0]
    if subject_column not in pheno.columns:
        raise ValueError(f"COBRE phenotypic CSV does not contain {subject_column!r}.")

    labels_by_subject = {}
    for _, row in pheno.iterrows():
        subject_id = _normalise_cobre_subject_id(row[subject_column])
        if not subject_id:
            continue
        labels_by_subject[subject_id] = int(row[label_column])

    data = []
    labels = []
    missing_labels = []
    invalid_shapes = []
    for path in rois_paths:
        subject_id = _normalise_cobre_subject_id(path.name)
        if subject_id not in labels_by_subject:
            missing_labels.append(subject_id)
            continue

        timeseries = np.loadtxt(path)
        if timeseries.ndim != 2 or timeseries.shape[1] != expected_rois:
            invalid_shapes.append((path.name, timeseries.shape))
            continue

        data.append(timeseries)
        labels.append(labels_by_subject[subject_id])

    if missing_labels:
        raise ValueError(
            "Missing COBRE phenotypic labels for ROI files: "
            + ", ".join(missing_labels[:10])
        )
    if invalid_shapes:
        examples = ", ".join(f"{name}: {shape}" for name, shape in invalid_shapes[:10])
        raise ValueError(f"COBRE ROI files with invalid shapes: {examples}")

    return data, np.asarray(labels, dtype=int)


def threshold_proportional(W: np.ndarray, p: float, copy: bool = True) -> np.ndarray:
    """Threshold a connectivity matrix to keep only the top-p fraction of weights.

    All sub-threshold weights and the diagonal are set to zero.

    Args:
        W:    Square connectivity matrix.
        p:    Proportion of strongest weights to preserve (0 < p < 1).
        copy: If True, operate on a copy of W.

    Returns:
        Thresholded matrix (symmetric if the input is symmetric).

    Source: unchanged from legacy_code/1-ST/step1_compute_ldw.py.
    """
    assert 0 < p < 1, "p must be in (0, 1)."
    if copy:
        W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)

    if np.all(W == W.T):
        W[np.tril_indices(n)] = 0
        ud = 2
    else:
        ud = 1

    ind = np.where(W)
    I   = np.argsort(W[ind])[::-1]
    en  = round((n * n - n) * p / ud)
    W[(ind[0][I][en:], ind[1][I][en:])] = 0

    if ud == 2:
        W[:, :] = W + W.T

    W[W > 0.9999] = 1
    return W


def build_static_connectivity(
    timeseries: list[np.ndarray],
    threshold_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one full-series Ledoit-Wolf graph per observation.

    This preserves the static-baseline preprocessing used by the original
    training scripts: signed correlations are node features, while absolute
    correlations are proportionally thresholded for adjacency matrices.
    """
    correlations = []
    adjacencies = []
    for subject_ts in timeseries:
        cov = LedoitWolf(assume_centered=False).fit(subject_ts.squeeze()).covariance_
        corr = np.nan_to_num(cov2corr(cov), nan=0.0, posinf=0.0, neginf=0.0)
        adjacency = threshold_proportional(np.abs(corr), threshold_p)
        np.fill_diagonal(adjacency, 1)
        correlations.append(corr.astype(np.float32))
        adjacencies.append(adjacency.astype(np.float32))
    return np.stack(correlations), np.stack(adjacencies)


def extract_ldw_corr(
    data: list,
    w_size: int = 20,
    shift: int = 10,
    threshold_p: float = 0.40,
):
    """Compute sliding-window Ledoit-Wolf correlation matrices and adjacency matrices.

    For each subject and each temporal window:
    1. Fit Ledoit-Wolf covariance.
    2. Convert to correlation via cov2corr.
    3. Apply proportional thresholding to create the adjacency matrix.

    Args:
        data:        List of per-subject time-series, each (timepoints × ROIs).
        w_size:      Sliding window length in time points.
        shift:       Window step size.
        threshold_p: Proportion of edges to keep (0 < p < 1).

    Returns:
        (node_feats, adj_mats, n_win)
        - node_feats: list[list[np.ndarray]] — correlation matrix per (subject, window).
        - adj_mats:   list[list[np.ndarray]] — thresholded adjacency per (subject, window).
        - n_win:      list[int] — number of windows per subject.

    Source: unchanged from legacy_code/1-ST/step1_compute_ldw.py (extract_ldw_corr).
    """
    overlap = w_size - shift
    n_win   = [int((item.shape[0] - overlap) / (w_size - overlap)) for item in data]

    node_feats = []
    adj_mats   = []

    for sub in tqdm(range(len(data)), desc="Processing subjects"):
        corr_sub = []
        adj_sub  = []

        for wi in range(n_win[sub]):
            st     = wi * (w_size - overlap)
            en     = st + w_size
            w_data = data[sub][st:en, :]

            lw  = LedoitWolf(assume_centered=False)
            cov = lw.fit(w_data.squeeze()).covariance_

            corr_neg = cov2corr(cov)
            corr     = np.abs(corr_neg)
            corr_sub.append(corr_neg)

            th_corr = threshold_proportional(corr, threshold_p)
            np.fill_diagonal(th_corr, 1)
            adj_sub.append(th_corr)

            assert not np.all(np.all(th_corr == 0, axis=1)), \
                "Adjacency matrix has all-zero rows."
            assert not np.all(np.all(th_corr == 0, axis=0)), \
                "Adjacency matrix has all-zero columns."
            assert np.all(th_corr >= 0), \
                "Adjacency matrix contains negative values."

        node_feats.append(corr_sub)
        adj_mats.append(adj_sub)

    return node_feats, adj_mats, n_win


def ledoit_wolf_covariance_torch(
    samples: np.ndarray | "torch.Tensor",
    device: str | "torch.device" = "cuda",
) -> "torch.Tensor":
    """Compute scikit-learn-compatible Ledoit-Wolf covariance with PyTorch.

    Float64 is used deliberately to keep the GPU path numerically aligned with
    the established scikit-learn implementation used by :func:`extract_ldw_corr`.
    """
    import torch

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA preprocessing was requested, but no CUDA device is available. "
            "Run this command inside a GPU allocation or pass --device cpu."
        )
    x = torch.as_tensor(samples, dtype=torch.float64, device=target)
    if x.ndim != 2:
        raise ValueError(f"Expected samples x features, got shape {tuple(x.shape)}")
    n_samples, n_features = x.shape
    if n_samples < 2:
        raise ValueError("Ledoit-Wolf covariance requires at least two samples")

    x = x - x.mean(dim=0)
    x2 = x.square()
    emp_cov_trace = x2.sum(dim=0) / n_samples
    mu = emp_cov_trace.sum() / n_features
    gram = x.T @ x
    beta_sum = (x2.T @ x2).sum()
    delta_sum = gram.square().sum() / (n_samples**2)
    beta = (beta_sum / n_samples - delta_sum) / (n_features * n_samples)
    delta = (
        delta_sum
        - 2.0 * mu * emp_cov_trace.sum()
        + n_features * mu.square()
    ) / n_features
    beta = torch.minimum(beta, delta)
    shrinkage = torch.where(beta == 0, torch.zeros_like(beta), beta / delta)

    covariance = gram / n_samples
    covariance = covariance * (1.0 - shrinkage)
    covariance.diagonal().add_(shrinkage * mu)
    return covariance


def extract_ldw_corr_torch(
    data: list,
    w_size: int = 20,
    shift: int = 10,
    threshold_p: float = 0.40,
    device: str = "cuda",
):
    """GPU-accelerated equivalent of :func:`extract_ldw_corr`.

    Ledoit-Wolf covariance and covariance-to-correlation conversion run on the
    selected PyTorch device. The resulting correlation matrix is transferred to
    CPU before applying the unchanged legacy proportional-thresholding function.
    """
    import torch

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA preprocessing was requested, but no CUDA device is available. "
            "Run this command inside a GPU allocation or pass --device cpu."
        )

    overlap = w_size - shift
    n_win = [int((item.shape[0] - overlap) / shift) for item in data]
    node_feats = []
    adj_mats = []
    for subject_index, subject_data in enumerate(
        tqdm(data, desc=f"Processing subjects on {target}")
    ):
        corr_sub = []
        adj_sub = []
        for window_index in range(n_win[subject_index]):
            start = window_index * shift
            window = subject_data[start : start + w_size, :]
            covariance = ledoit_wolf_covariance_torch(window, target)
            scale = covariance.diagonal().clamp_min(0).sqrt()
            denominator = torch.outer(scale, scale)
            correlation = covariance / denominator
            correlation.diagonal().fill_(1.0)
            corr_np = correlation.cpu().numpy()
            corr_sub.append(corr_np)

            thresholded = threshold_proportional(np.abs(corr_np), threshold_p)
            np.fill_diagonal(thresholded, 1)
            if np.any(np.all(thresholded == 0, axis=1)):
                raise AssertionError("Adjacency matrix has an all-zero row")
            if np.any(np.all(thresholded == 0, axis=0)):
                raise AssertionError("Adjacency matrix has an all-zero column")
            if np.any(thresholded < 0):
                raise AssertionError("Adjacency matrix contains negative values")
            adj_sub.append(thresholded)
        node_feats.append(corr_sub)
        adj_mats.append(adj_sub)
    return node_feats, adj_mats, n_win


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 1: compute Ledoit-Wolf correlation matrices from fMRI time-series."
    )
    parser.add_argument(
        "--timeseries_dir", default="./data/timeseries",
        help="Directory containing power_asd.npy and power_td.npy.",
    )
    parser.add_argument(
        "--output_dir", default="./data/ldw_data",
        help="Directory to save LDW_abide_data.pkl and win_info.pkl.",
    )
    parser.add_argument(
        "--source",
        choices=("legacy", "abide_aal", "cobre_aal"),
        default="legacy",
        help="Input source. Use *_aal options for 116-ROI AAL derivatives.",
    )
    parser.add_argument(
        "--rois_dir",
        default="./data/nilearn/ABIDE_pcp/cpac/nofilt_noglobal",
        help="Directory containing Nilearn *_rois_aal.1D files.",
    )
    parser.add_argument(
        "--phenotypic_csv",
        default="./data/manifests/abide_rois_aal_phenotypic.csv",
        help="Phenotypic CSV with ABIDE or COBRE label columns.",
    )
    parser.add_argument(
        "--cobre_subject_column",
        default=None,
        help="Optional COBRE CSV subject label column. Defaults to the first column.",
    )
    parser.add_argument(
        "--cobre_label_column",
        default="sz",
        help="COBRE CSV diagnosis column, default sz (1=schizophrenia, 0=control).",
    )
    parser.add_argument("--w_size",      type=int,   default=20,   help="Window size.")
    parser.add_argument("--shift",       type=int,   default=10,   help="Window shift.")
    parser.add_argument("--threshold_p", type=float, default=0.40, help="Edge proportion to keep.")
    args = parser.parse_args()

    print("Loading time-series data …")
    if args.source == "abide_aal":
        data, labels = load_abide_aal_timeseries(args.rois_dir, args.phenotypic_csv)
    elif args.source == "cobre_aal":
        data, labels = load_cobre_aal_timeseries(
            args.rois_dir,
            args.phenotypic_csv,
            subject_column=args.cobre_subject_column,
            label_column=args.cobre_label_column,
        )
    else:
        data, labels = load_timeseries(args.timeseries_dir)
    data = [np.array(x) for x in data]
    print(f"  {len(data)} subjects after cleaning.")

    print(f"Extracting LDW correlations (w_size={args.w_size}, shift={args.shift}) …")
    node_feats, adj_mats, n_win = extract_ldw_corr(
        data, args.w_size, args.shift, args.threshold_p
    )

    os.makedirs(args.output_dir, exist_ok=True)

    ldw_data = {"adj_mat": adj_mats, "node_feat": node_feats, "labels": labels}
    win_info = {"wSize": args.w_size, "shift": args.shift, "nWin": n_win}

    with open(os.path.join(args.output_dir, "LDW_abide_data.pkl"), "wb") as f:
        pickle.dump(ldw_data, f, protocol=4)
    with open(os.path.join(args.output_dir, "win_info.pkl"), "wb") as f:
        pickle.dump(win_info, f, protocol=4)

    print(f"Saved to {args.output_dir}.")
