# Validation

This page records the public software checks completed on 27 October 2026. It
does not claim that restricted-data thesis experiments were independently
reproduced.

## Tested platform and environments

Validation was performed on macOS 15.5 arm64 using CPU execution only.
BrainXDL-Open is designed for CPU and NVIDIA CUDA execution, with optional MPS
use where supported, but this design scope is broader than the completed test
evidence.

| Environment | Python | PyTorch | PyTorch Geometric | NumPy | SciPy | scikit-learn | pandas |
|---|---:|---:|---:|---:|---:|---:|---:|
| pip (`pyproject.toml`) | 3.10.22 | 2.14.1 | 2.8.1 | 1.26.4 | 1.15.3 | 1.7.2 | 2.3.3 |
| Conda (`environment.yml`) | 3.10.22 | 2.4.1 | 2.6.1 | 1.26.4 | 1.15.2 | 1.7.2 | 2.3.3 |

Both isolated installations passed `pip check`. Different PyTorch and PyTorch
Geometric versions reflect the compatible ranges resolved by the two package
managers, not a claim that every intermediate version is supported.

### Validation status by execution target

| Execution target | Design status | Independent validation status |
|---|---|---|
| CPU | Supported execution design | **Completed** in both recorded environments |
| NVIDIA CUDA GPU | Supported execution design | **Not tested**; no NVIDIA GPU was available |
| Apple MPS | Optional backend where operations are supported | **Not tested** |
| Linux | Intended cross-platform environment | **Not tested** |
| Windows | Intended cross-platform environment | **Not tested** |

An unavailable or skipped accelerator check is not a successful CUDA or MPS
validation. No GPU model, driver, or CUDA runtime result is reported because no
CUDA execution was performed.

## Checks performed

- Complete test suite: 78 passed in both the pip and Conda environments.
- Synthetic quick starts: ST-BrainXDL, MM-BrainXDL, and Applied XAI completed
  in both environments.
- Command-line interfaces: `--help` completed for all 25 executable scripts.
  The two neuroimaging visualization commands expose help without optional
  packages and require installation of the `xai` extra for actual analysis.
- Dependency integrity: no broken requirements were reported in either
  environment.
- Source/release regression: 143 deterministic arrays passed with
  `rtol=1e-6`, `atol=1e-7`, and maximum absolute difference `0.0`. The probes
  cover graph construction, metrics, ST and MM forward values and losses,
  trainable-parameter gradients and updates, and deterministic MM XAI outputs.
- Repository review: import targets, configuration paths, secret patterns,
  machine-specific paths, symlinks, large files, and sensitive-data patterns
  were checked before proposing the final inventory.

The numerical comparison harness and detailed per-array evidence are retained
as private release-compliance records rather than distributed as public
project-management material.

## Scientific scope

The regression comparison establishes that the released synthetic execution
paths preserve their source implementations within the stated tolerances. It
does not validate performance on ABIDE, COBRE, PsiConnect, or another research
cohort. Dataset access, participant-level folds, and trained checkpoints are
not distributed.

Participant-grouped public workflows reject participant overlap. Historical ST
behavior remains available for traceability, while strict validation settings
are recommended for new experiments. The legacy ST and PsiConnect procedures
remain distinct.

Whole-cohort ComBat fitting is not leakage-free cross-validation. The public
documentation does not recommend it inside cross-validation, and a split-aware
fit/transform implementation remains future work.

## Not validated

- Restricted-data thesis experiments and reported performance values.
- NVIDIA CUDA or Apple MPS execution.
- Linux or Windows.
- Optional XAI neuroimaging/3D dependencies and optional `neuroCombat` runtime.
- Cross-hardware bitwise reproducibility.
