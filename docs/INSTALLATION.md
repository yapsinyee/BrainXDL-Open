# Installation

These instructions install the BrainXDL-Open software. They do not download or
authorize access to restricted research datasets.

## Execution targets

BrainXDL-Open is designed to use standard PyTorch device execution on:

- CPU;
- NVIDIA CUDA GPUs; and
- Apple MPS where the required PyTorch and PyG operations are supported.

Only CPU execution has completed independent release validation. CUDA and MPS
execution remain untested for this release. Platform design intent must not be
read as validation of every operating-system, driver, PyTorch, CUDA, or PyG
combination.

## Recorded CPU validation environment

- macOS 15.5 arm64, CPU only
- Python 3.10.22
- PyTorch 2.14.1 and PyTorch Geometric 2.8.1
- NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3

This is one of two dependency resolutions tested on the recorded platform. The
package metadata allows Python 3.10 or newer, but that range is not a claim that
every Python or platform combination has been tested.

## CPU installation

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
to install a CPU build suitable for your operating system and package manager.
Then follow the [official PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for a PyG installation compatible with that PyTorch build. Finally, install the
project and its development dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate CPU builds of PyTorch and PyG first.
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest
```

This path passed dependency resolution, `pip check`, all 78 retained tests,
help checks for all 25 executable research scripts, and the three synthetic
examples on 2026-10-27. This is the independently verified installation path.

## NVIDIA CUDA installation

CUDA users should not copy a hard-coded CUDA package command from this
repository. Instead:

1. Confirm that the NVIDIA driver and GPU are visible on the target system.
2. Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
   to choose the supported CUDA-enabled PyTorch build appropriate to that
   system.
3. Verify the installed build with `torch.cuda.is_available()` and inspect
   `torch.version.cuda`.
4. Follow the [official PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
   for packages matching the installed PyTorch build and the CUDA runtime
   reported by PyTorch.
5. Install BrainXDL-Open with `python -m pip install -e ".[dev]"`, then run the
   test suite.

The CUDA runtime bundled with a PyTorch binary is the relevant compatibility
reference for PyG wheels. It need not be identical to a separately installed
system CUDA toolkit; a local toolkit is principally relevant when building
PyTorch or CUDA extensions from source. Driver, GPU, PyTorch, and PyG
compatibility must still be checked against their official documentation.

BrainXDL-Open does not pin a platform-specific PyTorch wheel. Its dependency
ranges do not establish that every CUDA/PyTorch/PyG combination works. CUDA
execution has not yet been independently tested for this release.

## Conda CPU environment (runtime verified)

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
for another operating system or accelerator. CUDA users should follow the
current official PyTorch and PyG guidance rather than assume this CPU-resolved
environment will select an appropriate CUDA build.

## Optional dependencies

```bash
python -m pip install -e ".[xai,dev]"
python -m pip install -e ".[harmonization,dev]"
```

The XAI extra supports optional neuroimaging/3D visualizations. Core synthetic
edge-mask XAI does not need it. PyTorch and PyG installation can depend on the
intended CUDA runtime; follow their official platform-specific instructions.
