# BrainXDL-Open

BrainXDL-Open is the open-source research framework developed as part of my PhD project, **“BrainXDL: Explainable Deep Learning for Personalized Prediction of Individual Behaviour and Neuropsychiatric Disorders from Brain Connectivity Networks,”** at Monash University (2026).

**Project website:** https://brainxdl.yapsinyee.com

The repository contains reusable implementations, experiment entry points, synthetic examples, and tests. It does **not** contain participant-level data, prepared folds, trained checkpoints, or restricted study artefacts. Synthetic validation demonstrates software behaviour, not reproduction of a thesis result.

## Research modules

- **ST-BrainXDL:** Constructs time-varying graphs from sliding-window Ledoit–Wolf connectivity and models them using variational graph encoders and graph-recurrent temporal updates.
- **MM-BrainXDL:** Uses separate fMRI and EEG graph encoders with adaptive and controlled alternative fusion mechanisms.
- **Applied XAI:** Provides ST attention capture, connectivity interpretation, temporal perturbation and fidelity analysis, latent-space analysis, and a project-specific optimised dense edge-mask explainer for MM models.

The compatibility class `DenseGNNExplainer` freezes the predictive model and optimises symmetric modality-specific edge masks. Signed saliency is computed as the optimised mask × adjacency.

## Repository structure

```text
brainxdl/       Reusable ST, MM, XAI, common, and pipeline code
configs/        Versioned experiment configurations
scripts/        Command-line experiment entry points
examples/       Synthetic demonstrations
tests/          Model, metric, split, and explanation tests
docs/           Installation, data, reproducibility, and thesis mapping
```

## Installation

Use Python 3.10 or newer. Install a CPU or CUDA build of PyTorch and PyTorch Geometric appropriate for your platform, then install BrainXDL-Open:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

See [Installation Guide](docs/INSTALLATION.md) for CPU and CUDA installation guidance and optional XAI dependencies.

## Synthetic quick starts

These commands do not require research datasets:

```bash
python -m pytest tests/test_common_metrics.py tests/test_mm_models.py
python scripts/st/st_train.py --help
python scripts/mm/train_mm.py --help
```

Synthetic examples for each research component are available under [`examples/`](examples/README.md):

```bash
python examples/st_synthetic.py
python examples/mm_synthetic.py
python examples/xai_synthetic.py
```

These examples demonstrate software interfaces and functionality; they do not reproduce thesis-reported experimental results.

## Data and grouped evaluation

Research datasets are not distributed with this repository. See [Dataset Requirements](docs/DATASETS.md) for access and preparation information.

PsiConnect workflows contain repeated observations per participant. Public MM training requires participant-grouped cross-validation and does not silently fall back to observation-level splitting.

Legacy ST fold functions and participant-grouped PsiConnect fold functions remain separate and should not be interchanged when comparing experimental results.

For new ST experiments using held-out validation data, the recommended settings are:

```bash
--strict_validation --no_test_during_training --selection_metric val_bce
```

The CLI defaults retain the historical experimental protocol for traceability and are not the recommended evaluation workflow for new experiments.

## Reproducibility and validation

The released implementations have undergone isolated installation, unit and integration testing, synthetic demonstrations, CLI verification, and deterministic numerical comparison with their source implementations.

Experiments requiring restricted datasets or trained checkpoints have not been independently rerun using this public release.

The framework is designed for CPU and NVIDIA CUDA execution. Completed release validation was performed on CPU; CUDA execution has not yet been independently tested. MPS, Linux, Windows, and optional neuroimaging dependencies also remain untested.

See [Experimental Results](docs/RESULTS.md), [Reproducibility](docs/REPRODUCIBILITY.md), and [Validation](docs/VALIDATION.md) for further details.

## Publications

The following publications contributed to the development of the BrainXDL research framework.

### ST-BrainXDL: Spatiotemporal Brain Connectivity Learning

**1. Graph Autoencoder-Based Embedded Learning (ICIP 2022)**

Noman, F., **Yap, S.-Y.**, Phan, R. C.-W., Ombao, H., & Ting, C.-M. (2022). Graph autoencoder-based embedded learning in dynamic brain networks for autism spectrum disorder identification. *2022 IEEE International Conference on Image Processing (ICIP)*, 2891–2895.

**DOI:** https://doi.org/10.1109/ICIP46576.2022.9898034

**Award:** Best Conference Paper Award 2023, IEEE SPS Malaysia Research Excellence Award.

*BrainXDL contribution: ST precursor — Graph Autoencoder.*

**2. Deep Probabilistic Spatiotemporal Framework (IJCAI 2024)**

**Yap, S.-Y.**, Loo, J. Y., Ting, C.-M., Noman, F., Phan, R. C.-W., Razi, A., & Dowe, D. L. (2024). A deep probabilistic spatiotemporal framework for dynamic graph representation learning with application to brain disorder identification. *Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI)*, 5353–5361.

**DOI:** https://doi.org/10.24963/ijcai.2024/592

*BrainXDL contribution: ST-BrainXDL.*

### MM-BrainXDL: Multimodal Brain Connectivity Learning

**3. Brain-MGF: Multimodal Graph Fusion (ISBI 2026)**

**Yap, S.-Y.**, Noman, F., Loo, J. Y., Stoliker, D., Khajehnejad, M., Phan, R. C.-W., Dowe, D. L., Razi, A., & Ting, C.-M. (2026). Brain-MGF: Multimodal graph fusion network for EEG-fMRI brain connectivity analysis under psilocybin. *2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)*, 1–5.

**DOI:** https://doi.org/10.1109/ISBI61048.2026.11515300

*BrainXDL contribution: MM-BrainXDL.*

### How to cite

If you use BrainXDL-Open in your research, please cite the publication relevant to the method used and acknowledge the BrainXDL-Open software repository.

Applied XAI is included as part of the BrainXDL research framework and does not currently have a separate publication citation.

BrainXDL-Open `v0.1.0` does not currently have a project DOI.

## Licence

BrainXDL-Open source code is licensed under the BSD 3-Clause Licence; see [`LICENSE`](LICENSE).

This software licence does not grant access to or redistribution rights for third-party or restricted datasets, checkpoints, or participant information.

Contribution guidelines are available in [`CONTRIBUTING.md`](CONTRIBUTING.md).