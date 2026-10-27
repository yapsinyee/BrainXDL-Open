# Contributing to BrainXDL-Open

Contributions that improve correctness, reproducibility, documentation, and
research usability are welcome.

## Before proposing a change

- Do not include participant-level records, restricted datasets, checkpoints,
  credentials, private paths, or unpublished research artifacts.
- Preserve ST-BrainXDL, MM-BrainXDL, and Applied XAI scientific behavior unless
  a change is explicitly described and justified as a methodology change.
- Keep legacy ST and participant-grouped PsiConnect protocols distinct.
- Require participant-grouped splits whenever repeated observations could place
  one participant in more than one data partition.
- Do not describe the MM dense edge-mask method as standard PyG GNNExplainer.
- Do not add third-party implementations without documented provenance,
  attribution, and licence compatibility.

## Development checks

Create an isolated Python 3.10 environment and install the development extras:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest
python examples/st_synthetic.py
python examples/mm_synthetic.py
python examples/xai_synthetic.py
```

For methodology changes, add focused regression tests and state what numerical
or experimental behavior changes. Synthetic tests are not evidence that a
restricted-data thesis experiment has been independently reproduced.

## Reporting results

Identify the dataset, preprocessing, split protocol, seed, model configuration,
threshold rule, and aggregation method. Complete all five folds for seed 2026
before reporting a new PsiConnect experiment, and keep single-seed summaries
separate from later multi-seed analyses. Include log loss whenever available.

## Licence

By submitting a contribution, you agree that it may be distributed under the
repository's BSD 3-Clause Licence. Dataset access terms remain separate and are
not granted by the software licence.
