# Results

BrainXDL-Open publishes tested implementations and synthetic validation
evidence. It does not publish or imply independently reproduced thesis results.

## Thesis-reported results

Numerical thesis tables are not reproduced in this repository. Recreating them
requires the authorized dataset versions, prepared folds, configurations, and
checkpoints described by the corresponding study records.

## Independently reproduced results

None yet.

Syntax checks, unit tests, synthetic forward passes, and smoke training runs do
not constitute independent reproduction of a scientific experiment.

## Synthetic validation outputs

| Validation | Status | Scientific metrics |
|---|---|---|
| ST synthetic forward/backward | PASS | not applicable; generated tensors only |
| MM synthetic forward/backward | PASS | not applicable; generated tensors only |
| MM dense edge-mask XAI | PASS | deterministic mask comparison; no cohort metric |
| Private/public regression | 143/143 arrays equal within `rtol=1e-6`, `atol=1e-7`; observed maximum absolute difference 0 | synthetic only |

The metric regression fixture exercises accuracy, precision, recall/sensitivity,
specificity, F1-score, ROC-AUC, and log loss on generated labels and scores.
Those values are software fixtures and are not model performance.

## Restricted-data status

| Contribution | Thesis-reported | Independently rerun here | Release status |
|---|---:|---:|---|
| ST-BrainXDL | values not migrated | no | requires authorized data/folds and five-fold rerun |
| MM-BrainXDL | values not migrated | no | requires PsiConnect access and five-fold rerun |
| Applied XAI | values not migrated | no | requires frozen checkpoints, folds, and authorized data |

Accuracy, precision, recall, F1-score, ROC-AUC, and log loss are therefore not
available as verified research outcomes in this repository. Missing values are
left missing rather than inferred from private narratives or other metrics.

The absence of restricted-data reruns does not prevent inspection or reuse of
the tested framework. It does prevent any claim that thesis performance has
been independently reproduced from this repository.

## Reporting requirements

Future curated tables must:

- identify dataset and preprocessing version;
- identify model/configuration, fold protocol, threshold rule, and seed;
- keep seed-2026 five-fold results separate from multi-seed results;
- include accuracy, ROC-AUC, F1, and log loss whenever available;
- distinguish fold mean ± standard deviation from pooled out-of-fold metrics;
- state whether the result was thesis-reported or rerun from BrainXDL-Open.
