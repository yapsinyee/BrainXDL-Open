# Reproducibility

## Status

BrainXDL-Open passed clean pip and Conda installations, its complete test suite,
three contribution-level synthetic examples, and a deterministic source/release
regression comparison. The thesis experiments have not been rerun from
BrainXDL-Open. Synthetic validation is not evidence that reported scientific
results were reproduced. Exact validation evidence is in `docs/VALIDATION.md`.

The public framework can therefore be installed and tested independently, but
it is not presented as a fully reproduced thesis archive.

## Split protocols

Two protocols intentionally remain separate:

1. Legacy ST utilities use their established nested stratified procedures and
   legacy seed behavior.
2. PsiConnect utilities use participant-grouped splits because each participant
   contributes repeated session observations.

Public repeated-participant MM workflows require groups and fail closed rather
than falling back silently to observation-level splitting. Manifest folds are
checked for participant overlap.

## ST protocol boundary

`scripts/st/st_train.py` intentionally retains legacy defaults. Without
`--strict_validation`, validation graphs are concatenated into optimizer
training. Unless `--no_test_during_training` is supplied, the test loader is
evaluated repeatedly. The default `legacy_train_bce_acc` checkpoint score uses
training quantities and minimizes `BCE + accuracy`; this historical behavior
is not suitable for new validation-controlled experiments.

The public PsiConnect nested sweep is the recommended controlled path. It
enforces strict train/validation separation, omits the outer-test loader during
inner selection, selects by validation BCE/log loss, and evaluates the outer
test only after selection. Do not relabel legacy results as nested-sweep results
or merge the two procedures.

## Seeds and folds

For new PsiConnect experiments and ablations, complete all five folds for seed
2026 first. Keep this single-seed summary separate from any later multi-seed
analysis. Legacy experiments retain their historical seeds.

## Selection, preprocessing, and evaluation

- Fit preprocessing and normalization only on training observations.
- Select checkpoints and classification thresholds using validation data.
- Evaluate the outer test fold once after selection.
- Preserve participant grouping in train, validation, and test splits.
- Report accuracy, precision, recall, F1, ROC-AUC, and log loss when available.
- Label fold means and pooled out-of-fold metrics distinctly.

The participant-grouped and refined MM workflows satisfy these rules. The
generic MM trainer uses grouped outer and validation splits when groups are
provided; its generic metrics use fixed 0.5/argmax decisions. Participant and
refined benchmarks instead select fold thresholds using validation labels only.

Sliding-window Ledoit-Wolf connectivity is fit independently within each
observation. Vector-model scalers in participant benchmarks are fit on training
rows only. By contrast, `harmonize_ldw_connectivity` fits ComBat jointly to all
subjects passed to it. Applying it before cross-validation would leak held-out
distribution information. It must not support a public CV claim until a
train-fit/held-out-transform protocol is implemented and tested.

## Expected output locations

Generated files belong under ignored `runs/`, `outputs/`, `reports/`, or
`plots/` directories. Curated public summaries must not contain participant rows.

## Known limitations

- CUDA, MPS, Linux, and Windows are untested.
- Pip and Conda resolve different tested PyTorch/PyG versions; neither is an
  assertion of support outside macOS arm64 CPU.
- Optional XAI and harmonization extras were not installed.
- Restricted-data experiments were not rerun.
- Floating-point and sparse graph operations may vary by hardware/library.
