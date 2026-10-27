# BrainXDL-Open

BrainXDL-Open is an open research framework for explainable deep learning on
brain connectivity networks. It accompanies research on personalized
prediction from fMRI and EEG connectivity while keeping reusable model code
separate from dataset-specific experiment orchestration.

The repository contains reusable implementations, experiment entry points,
synthetic examples, and tests. It does **not** contain participant-level data,
prepared folds, trained checkpoints, or restricted study artifacts. Synthetic
validation demonstrates software behavior, not reproduction of a thesis result.

## Research modules

- **ST-BrainXDL** constructs time-varying graphs from sliding-window
  Ledoit-Wolf connectivity and models them with variational graph encoders and
  graph-recurrent temporal updates.
- **MM-BrainXDL** uses separate fMRI and EEG graph encoders with adaptive or
  controlled alternative fusion mechanisms.
- **Applied XAI** provides ST attention capture, connectivity interpretation,
  temporal perturbation and fidelity analysis, latent-space analysis, and a
  project-specific optimized dense edge-mask explainer for MM models.

The compatibility class `DenseGNNExplainer` freezes the predictive model and
optimizes symmetric modality-specific edge masks. Signed saliency is computed
as optimized mask × adjacency. It is not the standard PyTorch Geometric
GNNExplainer implementation and it is not gradient-times-activation.

## Layout

```text
brainxdl/       reusable ST, MM, XAI, common, and pipeline code
configs/        versioned experiment configurations
scripts/        command-line entry points
tests/          tests for model, metric, split, and explanation behavior
docs/           installation, data, reproducibility, and thesis mapping
```

## Installation

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for optional XAI dependencies.

## Synthetic quick starts

These commands do not require research data:

```bash
python -m pytest tests/test_common_metrics.py tests/test_mm_models.py
python scripts/st/st_train.py --help
python scripts/mm/train_mm.py --help
```

Complete synthetic examples for each contribution are available under
[`examples/`](examples/README.md):

```bash
python examples/st_synthetic.py
python examples/mm_synthetic.py
python examples/xai_synthetic.py
```

They demonstrate interfaces only and are not scientific result reproductions.

## Data and grouped evaluation

Dataset files are not distributed. See [docs/DATASETS.md](docs/DATASETS.md).
PsiConnect workflows contain repeated observations per participant. Public MM
training refuses to fall back silently to observation-level splitting and
requires grouped cross-validation for repeated-participant data.

The legacy ST fold functions and participant-grouped PsiConnect fold functions
remain separate. They must not be interchanged when comparing results.

For ST training on held-out validation data, use `--strict_validation
--no_test_during_training --selection_metric val_bce`. The CLI defaults retain
the historical protocol and are documented as legacy behavior, not as the
recommended public evaluation workflow.

## Reproducibility and validation

The released implementations have been tested through isolated installation,
unit and integration tests, synthetic examples, CLI checks, and deterministic
numerical comparison with their source implementations. Thesis experiments
requiring restricted data and checkpoints have not been independently rerun.
Results and limitations are tracked in
[docs/RESULTS.md](docs/RESULTS.md) and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Validation was performed on macOS arm64 CPU with Python 3.10 using both pip and
Conda specifications. CUDA, MPS, Linux, Windows, and optional neuroimaging
extras remain untested. See [docs/VALIDATION.md](docs/VALIDATION.md) for the
tested versions and validation scope.

## Publications and citation

ST-BrainXDL-related work is described in:

> Sin-Yee Yap, Junn Yong Loo, Chee-Ming Ting, Fuad Noman, Raphaël C.-W. Phan,
> Adeel Razi, and David L. Dowe. “A Deep Probabilistic Spatiotemporal Framework
> for Dynamic Graph Representation Learning with Application to Brain Disorder
> Identification.” IJCAI 2024. DOI: `10.24963/ijcai.2024/592`.

MM-BrainXDL-related work is described in the Brain-MGF preprint, arXiv
`2511.18325`. Complete verified author metadata is available in `CITATION.cff`.
Applied XAI has no separate publication claim in this repository.

When using the software, cite BrainXDL-Open and the publication relevant to the
method used. Version 0.1.0 does not currently have a project DOI.

## Licence

BrainXDL-Open source code is licensed under the BSD 3-Clause Licence; see
[`LICENSE`](LICENSE). This software licence does not grant access to or
redistribution rights for third-party or restricted datasets, checkpoints, or
participant information. Contributions are described in
[`CONTRIBUTING.md`](CONTRIBUTING.md).
