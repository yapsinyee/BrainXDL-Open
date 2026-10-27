# Installation

These instructions install the BrainXDL-Open software. They do not download or
authorize access to restricted research datasets.

## Validated baseline

- macOS 15.5 arm64, CPU only
- Python 3.10.22
- PyTorch 2.14.1 and PyTorch Geometric 2.8.1
- NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3

This is one of the two dependency resolutions tested on the same platform. The package metadata
allows Python 3.10 or newer, but that range is not a platform-support claim.
CUDA, MPS, Linux, Windows, and optional XAI/harmonization extras were not tested.

## Virtual environment (runtime verified)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest
```

This path passed dependency resolution, `pip check`, all 78 retained tests,
help checks for all 25 executable research scripts, and the three synthetic
examples on 2026-10-27.

## Conda (runtime verified)

```bash
conda env create -f environment.yml
conda activate brainxdl-open
python -m pytest
```

The environment was materialized on macOS arm64 and selected Python 3.10.22,
PyTorch 2.4.1, PyG 2.6.1, NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.7.2,
pandas 2.3.3, Matplotlib 3.10.9, PyYAML 6.0.3, dill 0.4.1, and pytest 9.1.1.
`pip check` and all 78 retained tests passed. Pip and Conda intentionally allow
compatible ranges and therefore resolved different PyTorch/PyG versions; both
observed resolutions passed on macOS arm64 CPU. This does not establish support
for another operating system or accelerator.

## Optional dependencies

```bash
python -m pip install -e ".[xai,dev]"
python -m pip install -e ".[harmonization,dev]"
```

The XAI extra supports optional neuroimaging/3D visualizations. Core synthetic
edge-mask XAI does not need it. PyTorch and PyG installation can depend on the
intended CUDA runtime; follow their official platform-specific instructions.
