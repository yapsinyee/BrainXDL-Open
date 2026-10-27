"""ComBat harmonization helpers for ST-BrainXDL LDW connectivity features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from brainxdl.st.preprocessing import threshold_proportional


@dataclass(frozen=True)
class CombatWindowReport:
    """Summary for one dynamic window harmonized with ComBat."""

    window: int
    n_subjects: int
    n_features: int
    n_harmonized_features: int
    n_constant_or_invalid_features: int


def _import_neurocombat():
    try:
        from neuroCombat import neuroCombat
    except ImportError as exc:
        raise ImportError(
            "neuroCombat is required for ComBat harmonization. Install it with "
            "`pip install neuroCombat` in the BrainXDL environment."
        ) from exc
    return neuroCombat


def _normalise_column_list(columns: str | Iterable[str] | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        if columns.strip().lower() in {"", "none", "null"}:
            return []
        return [item.strip() for item in columns.split(",") if item.strip()]
    return [str(item).strip() for item in columns if str(item).strip()]


def infer_available_covariates(
    pheno: pd.DataFrame,
    *,
    requested_categorical: str | Iterable[str] | None = None,
    requested_continuous: str | Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve requested covariates, accepting ``auto`` for ABIDE defaults."""

    if requested_categorical == "auto":
        categorical = [col for col in ("DX_GROUP", "SEX") if col in pheno.columns]
    else:
        categorical = _normalise_column_list(requested_categorical)

    if requested_continuous == "auto":
        continuous = [col for col in ("AGE_AT_SCAN",) if col in pheno.columns]
    else:
        continuous = _normalise_column_list(requested_continuous)

    missing = [col for col in categorical + continuous if col not in pheno.columns]
    if missing:
        raise ValueError(
            "Requested ComBat covariate columns are missing from phenotype CSV: "
            + ", ".join(missing)
        )

    return categorical, continuous


def align_abide_phenotype_to_roi_order(
    rois_dir: str | Path,
    phenotypic_csv: str | Path,
    *,
    pattern: str = "*_rois_aal.1D",
    file_id_col: str = "FILE_ID",
) -> pd.DataFrame:
    """Return ABIDE phenotype rows in the same sorted ROI-file order as preprocessing."""

    rois_paths = sorted(Path(rois_dir).glob(pattern))
    if not rois_paths:
        raise FileNotFoundError(f"No ROI files matching {pattern!r} in {rois_dir}")

    pheno = pd.read_csv(phenotypic_csv)
    if file_id_col not in pheno.columns:
        raise ValueError(f"Phenotype CSV must contain {file_id_col!r}.")

    by_file_id = {
        str(row[file_id_col]): row
        for _, row in pheno.iterrows()
    }

    rows = []
    missing = []
    for path in rois_paths:
        file_id = path.name.removesuffix("_rois_aal.1D")
        row = by_file_id.get(file_id)
        if row is None:
            missing.append(file_id)
        else:
            rows.append(row)

    if missing:
        raise ValueError(
            "Missing phenotype rows for ROI files: " + ", ".join(missing[:10])
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def _validate_covars(covars: pd.DataFrame, batch_col: str, covariate_cols: list[str]) -> None:
    required = [batch_col] + covariate_cols
    missing = [col for col in required if col not in covars.columns]
    if missing:
        raise ValueError("Missing ComBat columns: " + ", ".join(missing))

    null_cols = [col for col in required if covars[col].isna().any()]
    if null_cols:
        raise ValueError(
            "ComBat covariates cannot contain missing values. Columns with missing "
            "values: " + ", ".join(null_cols)
        )

    if covars[batch_col].nunique() < 2:
        raise ValueError(
            f"ComBat batch column {batch_col!r} must contain at least two batches."
        )


def _harmonize_matrix(
    matrix: np.ndarray,
    covars: pd.DataFrame,
    *,
    batch_col: str,
    categorical_cols: list[str],
    continuous_cols: list[str],
    eb: bool,
    parametric: bool,
    mean_only: bool,
    ref_batch: str | int | None,
    verbose: bool,
) -> tuple[np.ndarray, int, int]:
    """Run neuroCombat on finite, non-constant feature rows of a feature matrix."""

    finite = np.all(np.isfinite(matrix), axis=1)
    nonconstant = np.nanstd(matrix, axis=1) > 0
    usable = finite & nonconstant

    output = matrix.copy()
    if np.any(usable):
        neuroCombat = _import_neurocombat()
        result = neuroCombat(
            dat=matrix[usable],
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            eb=eb,
            parametric=parametric,
            mean_only=mean_only,
            ref_batch=ref_batch,
        )
        output[usable] = np.asarray(result["data"], dtype=np.float64)

    return output, int(np.sum(usable)), int(np.sum(~usable))


def harmonize_ldw_connectivity(
    node_feats: list[list[np.ndarray]],
    covars: pd.DataFrame,
    *,
    batch_col: str = "SITE_ID",
    categorical_cols: str | Iterable[str] | None = "auto",
    continuous_cols: str | Iterable[str] | None = "auto",
    threshold_p: float = 0.40,
    eb: bool = True,
    parametric: bool = True,
    mean_only: bool = False,
    ref_batch: str | int | None = None,
    verbose: bool = False,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[CombatWindowReport]]:
    """Harmonize dynamic LDW correlation matrices window-by-window with ComBat.

    Each dynamic window is treated as one ComBat problem with matrix shape
    ``features x subjects``. Subjects without that window are excluded from that
    window's fit. Constant/invalid features, including the diagonal, are left
    unchanged because the Python ComBat implementation requires finite,
    non-constant feature rows.
    """

    if len(node_feats) != len(covars):
        raise ValueError(
            f"LDW subject count ({len(node_feats)}) does not match phenotype rows "
            f"({len(covars)})."
        )

    covars = covars.reset_index(drop=True).copy()
    categorical_cols, continuous_cols = infer_available_covariates(
        covars,
        requested_categorical=categorical_cols,
        requested_continuous=continuous_cols,
    )
    _validate_covars(covars, batch_col, categorical_cols + continuous_cols)

    max_windows = max(len(subject_windows) for subject_windows in node_feats)
    harmonized = [[np.asarray(window, dtype=np.float64).copy() for window in subject] for subject in node_feats]
    reports: list[CombatWindowReport] = []

    for window_idx in range(max_windows):
        subject_idx = [
            idx for idx, subject_windows in enumerate(node_feats)
            if window_idx < len(subject_windows)
        ]
        if not subject_idx:
            continue

        matrices = [np.asarray(node_feats[idx][window_idx], dtype=np.float64) for idx in subject_idx]
        first_shape = matrices[0].shape
        if any(matrix.shape != first_shape for matrix in matrices):
            raise ValueError(f"Window {window_idx} contains inconsistent matrix shapes.")
        if first_shape[0] != first_shape[1]:
            raise ValueError(f"Window {window_idx} matrices must be square.")

        feature_matrix = np.stack([matrix.reshape(-1) for matrix in matrices], axis=1)
        harmonized_features, n_usable, n_skipped = _harmonize_matrix(
            feature_matrix,
            covars.iloc[subject_idx].reset_index(drop=True),
            batch_col=batch_col,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            eb=eb,
            parametric=parametric,
            mean_only=mean_only,
            ref_batch=ref_batch,
            verbose=verbose,
        )

        for col, idx in enumerate(subject_idx):
            matrix = harmonized_features[:, col].reshape(first_shape)
            matrix = (matrix + matrix.T) / 2.0
            np.fill_diagonal(matrix, 1.0)
            harmonized[idx][window_idx] = matrix

        reports.append(
            CombatWindowReport(
                window=window_idx,
                n_subjects=len(subject_idx),
                n_features=int(feature_matrix.shape[0]),
                n_harmonized_features=n_usable,
                n_constant_or_invalid_features=n_skipped,
            )
        )

    adj_mats: list[list[np.ndarray]] = []
    for subject_windows in harmonized:
        subject_adj = []
        for matrix in subject_windows:
            corr = np.abs(matrix)
            th_corr = threshold_proportional(corr, threshold_p)
            np.fill_diagonal(th_corr, 1)
            subject_adj.append(th_corr)
        adj_mats.append(subject_adj)

    return harmonized, adj_mats, reports
