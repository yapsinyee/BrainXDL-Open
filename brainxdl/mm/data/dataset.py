"""MM data loading and preprocessing utilities.

Handles fMRI + EEG .mat file loading, connectivity matrix construction,
dataset building, and caching for the PsiConnect pipeline.

Source: extracted from legacy_code/2-MM/main.py.
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch

from brainxdl.common.connectivity import generate_connectivity_matrix
from brainxdl.mm.utils.io import get_task_list, slugify


# ---------------------------------------------------------------------------
# Sliding-window connectivity (dynamic graph mode)
# ---------------------------------------------------------------------------

def _sliding_connectivity(
    timeseries: np.ndarray,
    window_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sliding-window correlation and partial correlation stacks.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    if window_len <= 0 or stride <= 0:
        raise ValueError("window_len and stride must be positive.")
    num_timepoints, _ = timeseries.shape
    if num_timepoints < window_len:
        raise ValueError(f"time_series too short ({num_timepoints}) for window_len={window_len}")

    starts = list(range(0, num_timepoints - window_len + 1, stride))
    if not starts or starts[-1] != num_timepoints - window_len:
        # ensure last window is aligned to the tail
        starts.append(num_timepoints - window_len)

    corr_stack: List[np.ndarray] = []
    pcorr_stack: List[np.ndarray] = []
    for start in starts:
        window = timeseries[start : start + window_len]
        corr_stack.append(generate_connectivity_matrix(window, "correlation"))
        pcorr_stack.append(generate_connectivity_matrix(window, "partial correlation"))

    return np.stack(corr_stack), np.stack(pcorr_stack)


# ---------------------------------------------------------------------------
# Dataset statistics helpers
# ---------------------------------------------------------------------------

def summarize_split_counts(
    split_name: str,
    indices: np.ndarray,
    labels: np.ndarray,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Print per-class sample counts for a given split and return the histogram.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    hist = np.bincount(labels[indices].astype(int), minlength=num_classes)
    imbalance_ratio = compute_min_max_ratio(hist)
    hist_str = ", ".join(f"class {cls}: {count}" for cls, count in enumerate(hist))
    total = hist.sum()
    print(f"{split_name} class counts -> {hist_str} | total={int(total)}, min/max ratio={imbalance_ratio:.3f}")
    return hist


def compute_min_max_ratio(hist: np.ndarray) -> float:
    """Return the minority/majority ratio ignoring empty classes.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    non_zero = hist[hist > 0]
    if non_zero.size == 0:
        return float("nan")
    if non_zero.size == 1:
        return 1.0
    return float(non_zero.min()) / float(non_zero.max())


def build_weighted_sampler(
    train_labels: np.ndarray, class_hist: np.ndarray
) -> torch.utils.data.WeightedRandomSampler:
    """Create a WeightedRandomSampler using inverse-frequency class weights.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    class_hist = class_hist.astype(np.float32)
    class_weights = class_hist.sum() / (len(class_hist) * np.maximum(class_hist, 1.0))
    sample_weights = class_weights[train_labels.astype(int)]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
    )


def infer_pos_weight(train_hist: np.ndarray, default: float = 1.0) -> float:
    """Compute pos_weight = (#neg / #pos) assuming binary labels ordered [neg, pos].

    Source: extracted from legacy_code/2-MM/main.py.
    """
    if train_hist.size < 2:
        warnings.warn("Not enough classes to compute pos_weight; falling back to default=1.0.")
        return default
    neg = float(train_hist[0])
    pos = float(train_hist[1])
    if pos == 0:
        warnings.warn("Positive class count is zero; falling back to pos_weight=1.0.")
        return default
    return neg / pos


# ---------------------------------------------------------------------------
# Cache path resolution
# ---------------------------------------------------------------------------

def resolve_cache_path(args: argparse.Namespace) -> Optional[str]:
    """Resolve the .pkl cache path from args.

    Source: extracted from legacy_code/2-MM/main.py.
    """
    if getattr(args, "cache_path", None):
        cache_dir = os.path.dirname(args.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        return args.cache_path

    tasks_list = get_task_list(args)
    task_tag = "-".join(sorted(set(tasks_list))) if tasks_list else "all"
    cache_dir = args.cache_dir
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    graph_tag = slugify(getattr(args, "graph_mode", "static"))
    return os.path.join(cache_dir, f"psiconnect_cache_{task_tag}_{graph_tag}.pkl")


# ---------------------------------------------------------------------------
# Main data loader
# ---------------------------------------------------------------------------

def prepare_data(args: argparse.Namespace) -> Tuple[np.ndarray, ...]:
    """Load fMRI and EEG .mat files, compute connectivity matrices, and return arrays.

    Returns:
        (corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels, pseudo_fmri, pseudo_eeg, groups, tasks)

    Source: extracted from legacy_code/2-MM/main.py.
    """
    if args.dataset.upper() != "PSICONNECT":
        raise ValueError("Only the PsiConnect dataset is supported in this pipeline.")

    if not os.path.isdir(args.fmri_path):
        raise FileNotFoundError(f"fMRI directory not found: {args.fmri_path}")
    if not os.path.isdir(args.eeg_path):
        raise FileNotFoundError(f"EEG directory not found: {args.eeg_path}")

    fmri_files = sorted(f for f in os.listdir(args.fmri_path) if f.endswith(".mat"))
    eeg_files = sorted(f for f in os.listdir(args.eeg_path) if f.endswith(".mat"))
    if not fmri_files:
        raise FileNotFoundError(f"No fMRI .mat files found in {args.fmri_path}")
    if not eeg_files:
        raise FileNotFoundError(f"No EEG .mat files found in {args.eeg_path}")

    tasks_cli = get_task_list(args)
    if tasks_cli:
        task_keys = []
        for t in tasks_cli:
            task_keys.extend([f"task-{t}", f"task_{t}"])

        def match_any_task(name: str) -> bool:
            lname = name.lower()
            return any(k in lname for k in task_keys)

        fmri_files = [f for f in fmri_files if match_any_task(f)]
        eeg_files = [f for f in eeg_files if match_any_task(f)]
        if not fmri_files or not eeg_files:
            raise RuntimeError("No files remain after applying task filters. Check --task/--tasks.")

    def bids_key(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        stem = os.path.splitext(os.path.basename(filename))[0]
        tokens = stem.split("_")
        sub = ses = task = None
        for token in tokens:
            if token.startswith("sub-"):
                sub = token[4:]
            elif token.startswith("ses-"):
                ses = token[4:]
            elif token.startswith("task-"):
                task = token[5:]
        return sub, ses, task

    fmri_map: Dict[Tuple, str] = {}
    eeg_map: Dict[Tuple, str] = {}

    for fname in fmri_files:
        key = bids_key(fname)
        if key[0] is None or key[1] is None:
            warnings.warn(f"Skipping fMRI file without sub/ses: {fname}")
            continue
        fmri_map[key] = fname

    for fname in eeg_files:
        key = bids_key(fname)
        if key[0] is None or key[1] is None:
            warnings.warn(f"Skipping EEG file without sub/ses: {fname}")
            continue
        eeg_map[key] = fname

    common_keys = sorted(set(fmri_map.keys()) & set(eeg_map.keys()))
    if not common_keys:
        fmri_keys_relaxed = {(k[0], k[1]): v for k, v in fmri_map.items()}
        eeg_keys_relaxed = {(k[0], k[1]): v for k, v in eeg_map.items()}
        relaxed = sorted(set(fmri_keys_relaxed.keys()) & set(eeg_keys_relaxed.keys()))
        if not relaxed:
            raise RuntimeError("No overlapping subject/session pairs found between fMRI and EEG data.")
        common_keys = [(k[0], k[1], None) for k in relaxed]

    missing_fmri = sorted(set(eeg_map.keys()) - set(fmri_map.keys()))
    missing_eeg = sorted(set(fmri_map.keys()) - set(eeg_map.keys()))
    if missing_fmri:
        warnings.warn(f"Missing fMRI for {len(missing_fmri)} EEG keys (first 3): {missing_fmri[:3]}")
    if missing_eeg:
        warnings.warn(f"Missing EEG for {len(missing_eeg)} fMRI keys (first 3): {missing_eeg[:3]}")

    corr_fmri_list: List[np.ndarray] = []
    pcorr_fmri_list: List[np.ndarray] = []
    corr_eeg_list: List[np.ndarray] = []
    pcorr_eeg_list: List[np.ndarray] = []
    pseudo_fmri_list: List[np.ndarray] = []
    pseudo_eeg_list: List[np.ndarray] = []
    labels: List[int] = []
    participant_groups: List[str] = []
    task_labels: List[str] = []

    expected_fmri_nodes: Optional[int] = None
    expected_eeg_nodes: Optional[int] = None
    fmri_windows_per_subject: List[int] = []
    eeg_windows_per_subject: List[int] = []

    for key in common_keys:
        fmri_fname = fmri_map.get(key) or fmri_map.get((key[0], key[1], None))
        eeg_fname = eeg_map.get(key) or eeg_map.get((key[0], key[1], None))
        if fmri_fname is None or eeg_fname is None:
            continue

        fmri_mat = sio.loadmat(os.path.join(args.fmri_path, fmri_fname))
        if "time_series" not in fmri_mat:
            raise KeyError(f"time_series not found in fMRI file: {fmri_fname}")
        fmri_ts_raw = fmri_mat["time_series"]
        roi_names = fmri_mat.get("ROI_names")
        roi_count = None
        if isinstance(roi_names, np.ndarray):
            roi_count = int(np.prod(roi_names.shape))
        if fmri_ts_raw.ndim != 2:
            raise ValueError(f"fMRI time_series in {fmri_fname} is not 2D (shape={fmri_ts_raw.shape})")

        # Prefer the orientation whose axis matches the ROI count; fall back to the heuristic that
        # the shorter axis is nodes when ROI count is unavailable.
        if roi_count is not None:
            if fmri_ts_raw.shape[1] == roi_count:
                fmri_ts = fmri_ts_raw
            elif fmri_ts_raw.shape[0] == roi_count:
                fmri_ts = fmri_ts_raw.T
            else:
                heuristic = fmri_ts_raw if fmri_ts_raw.shape[1] <= fmri_ts_raw.shape[0] else fmri_ts_raw.T
                warnings.warn(
                    f"ROI count ({roi_count}) not found as an axis in {fmri_fname} (shape={fmri_ts_raw.shape}); "
                    f"using heuristic orientation {heuristic.shape}."
                )
                fmri_ts = heuristic
        else:
            fmri_ts = fmri_ts_raw if fmri_ts_raw.shape[1] <= fmri_ts_raw.shape[0] else fmri_ts_raw.T

        if expected_fmri_nodes is None:
            expected_fmri_nodes = fmri_ts.shape[1]
        if fmri_ts.shape[1] != expected_fmri_nodes:
            warnings.warn(
                f"Skipping {fmri_fname}: expected {expected_fmri_nodes} ROIs, found {fmri_ts.shape[1]}"
            )
            continue
        if getattr(args, "graph_mode", "static") == "dynamic":
            corr_fmri_win, pcorr_fmri_win = _sliding_connectivity(
                fmri_ts, args.window_len, args.window_stride
            )
            corr_fmri_list.append(corr_fmri_win.astype(np.float32))
            pcorr_fmri_list.append(pcorr_fmri_win.astype(np.float32))
            pseudo_fmri_list.append(
                np.repeat(np.eye(fmri_ts.shape[1], dtype=np.float32)[None, ...], corr_fmri_win.shape[0], axis=0)
            )
            fmri_windows_per_subject.append(int(corr_fmri_win.shape[0]))
        else:
            corr_fmri_list.append(generate_connectivity_matrix(fmri_ts, "correlation").astype(np.float32))
            pcorr_fmri_list.append(generate_connectivity_matrix(fmri_ts, "partial correlation").astype(np.float32))
            pseudo_fmri_list.append(np.eye(fmri_ts.shape[1], dtype=np.float32))

        eeg_mat = sio.loadmat(os.path.join(args.eeg_path, eeg_fname), squeeze_me=True, struct_as_record=False)
        if "ftData" not in eeg_mat:
            raise KeyError(f"ftData not found in EEG file: {eeg_fname}")
        ft_data = eeg_mat["ftData"]

        raw_trials = ft_data.trial
        if isinstance(raw_trials, (list, tuple)):
            trial_arrays = [np.asarray(trial, dtype=np.float32) for trial in raw_trials]
        elif isinstance(raw_trials, np.ndarray) and raw_trials.dtype == object:
            trial_arrays = [np.asarray(trial, dtype=np.float32) for trial in raw_trials.tolist()]
        elif isinstance(raw_trials, np.ndarray) and raw_trials.ndim == 3:
            trial_arrays = [np.asarray(trial, dtype=np.float32) for trial in raw_trials]
        else:
            trial_arrays = [np.asarray(raw_trials, dtype=np.float32)]

        if not trial_arrays:
            raise ValueError(f"No EEG trials found in file: {eeg_fname}")

        eeg_ts = np.concatenate(trial_arrays, axis=-1).T  # (time, channels)
        if expected_eeg_nodes is None:
            expected_eeg_nodes = eeg_ts.shape[1]
        if eeg_ts.shape[1] != expected_eeg_nodes:
            warnings.warn(
                f"Skipping {eeg_fname}: expected {expected_eeg_nodes} EEG channels, found {eeg_ts.shape[1]}"
            )
            corr_fmri_list.pop()
            pcorr_fmri_list.pop()
            pseudo_fmri_list.pop()
            if getattr(args, "graph_mode", "static") == "dynamic" and fmri_windows_per_subject:
                fmri_windows_per_subject.pop()
            continue
        if getattr(args, "graph_mode", "static") == "dynamic":
            corr_eeg_win, pcorr_eeg_win = _sliding_connectivity(
                eeg_ts, args.window_len, args.window_stride
            )
            corr_eeg_list.append(corr_eeg_win.astype(np.float32))
            pcorr_eeg_list.append(pcorr_eeg_win.astype(np.float32))
            pseudo_eeg_list.append(
                np.repeat(np.eye(eeg_ts.shape[1], dtype=np.float32)[None, ...], corr_eeg_win.shape[0], axis=0)
            )
            eeg_windows_per_subject.append(int(corr_eeg_win.shape[0]))
        else:
            corr_eeg_list.append(generate_connectivity_matrix(eeg_ts, "correlation").astype(np.float32))
            pcorr_eeg_list.append(generate_connectivity_matrix(eeg_ts, "partial correlation").astype(np.float32))
            pseudo_eeg_list.append(np.eye(eeg_ts.shape[1], dtype=np.float32))

        task_fmri = bids_key(fmri_fname)[2]
        task_eeg = bids_key(eeg_fname)[2]
        task_value = task_fmri or task_eeg or "unknown"
        task_labels.append(str(task_value).lower())

        ses = key[1]
        participant_groups.append(str(key[0]))
        ses_value = str(ses)
        if ses_value.startswith("01") or ses_value == "1":
            labels.append(1)  # Psilocybin
        elif ses_value.startswith("02") or ses_value == "2":
            labels.append(0)  # Control
        else:
            warnings.warn(f"Unrecognized session '{ses}' in key {key}; assigning label 0")
            labels.append(0)

    if not corr_fmri_list or not corr_eeg_list:
        raise RuntimeError("No matched fMRI/EEG pairs were loaded.")

    if getattr(args, "graph_mode", "static") == "dynamic":
        min_fmri_T = min(fmri_windows_per_subject) if fmri_windows_per_subject else 0
        min_eeg_T = min(eeg_windows_per_subject) if eeg_windows_per_subject else 0
        if min_fmri_T < 1 or min_eeg_T < 1:
            raise RuntimeError("Dynamic graph mode produced empty window stacks.")
        corr_fmri = np.stack([arr[:min_fmri_T] for arr in corr_fmri_list])
        pcorr_fmri = np.stack([arr[:min_fmri_T] for arr in pcorr_fmri_list])
        corr_eeg = np.stack([arr[:min_eeg_T] for arr in corr_eeg_list])
        pcorr_eeg = np.stack([arr[:min_eeg_T] for arr in pcorr_eeg_list])
        pseudo_fmri = np.stack([arr[:min_fmri_T] for arr in pseudo_fmri_list])
        pseudo_eeg = np.stack([arr[:min_eeg_T] for arr in pseudo_eeg_list])
    else:
        corr_fmri = np.stack(corr_fmri_list)
        pcorr_fmri = np.stack(pcorr_fmri_list)
        corr_eeg = np.stack(corr_eeg_list)
        pcorr_eeg = np.stack(pcorr_eeg_list)
        pseudo_fmri = np.stack(pseudo_fmri_list)
        pseudo_eeg = np.stack(pseudo_eeg_list)

    labels_arr = np.asarray(labels, dtype=np.int64)
    groups_arr = np.asarray(participant_groups, dtype=object)
    tasks_arr = np.asarray(task_labels, dtype=object)
    return corr_fmri, pcorr_fmri, corr_eeg, pcorr_eeg, labels_arr, pseudo_fmri, pseudo_eeg, groups_arr, tasks_arr
