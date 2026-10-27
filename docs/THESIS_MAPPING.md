# Thesis-to-Code Mapping

This document maps the tested BrainXDL-Open framework to research contributions;
it is not a claim that restricted-data thesis experiments were rerun.

This map uses contribution names rather than asserting chapter, section, or
table numbers that are not encoded in the software metadata.

| Thesis contribution | Model or method | Primary code | Configuration | Entry point | Expected output |
|---|---|---|---|---|---|
| Spatiotemporal dynamic connectivity | `STBrainXDL` | `brainxdl/st/model.py` | legacy CLI; controlled PsiConnect config below | `python scripts/st/st_train.py --help` | checkpoint plus train/validation/test histories under the selected save directory |
| Dynamic graph construction | sliding-window Ledoit-Wolf correlation and proportional thresholding | `brainxdl/st/preprocessing.py`, `brainxdl/st/data_prep.py` | window, stride, density | `scripts/psiconnect/prepare_psiconnect_st.py` | prepared grouped folds under ignored `data/` |
| PsiConnect ST protocol | participant-grouped nested folds | `brainxdl/st/psiconnect.py` | `configs/psiconnect/st_nested_sweep.yaml` | `scripts/psiconnect/st_nested_sweep.py` | selection metadata and fold artifacts |
| Multimodal EEG–fMRI connectivity | `MMBrainXDL` (historical alias `BrainMGF`) | `brainxdl/mm/models/brainmgf.py` | CLI plus `configs/mm/refined/*.yaml` | `python scripts/mm/train_mm.py --help` | fold predictions, metric summaries, and optional checkpoints |
| Dynamic multimodal variant | `DynamicMMBrainXDL` | `brainxdl/mm/models/dynamic_brainmgf.py` | CLI settings | `python scripts/mm/train_mm.py --graph_mode dynamic --help` | dynamic MM evaluation artifacts |
| Controlled MM fusion variants | MM-A0–MM-A5 | `brainxdl/mm/refined_benchmark.py` | `configs/mm/refined/*.yaml` | `scripts/psiconnect/run_refined_mm_brainxdl.py` | five-fold variant summaries |
| Attention analysis | Transformer attention capture | `brainxdl/xai/st_attention.py` | extraction CLI | `scripts/st/extract_st_attention.py` | attention tensors and aggregate summaries |
| Temporal importance | block perturbation and prediction-margin analysis | `brainxdl/xai/st_temporal.py` | CLI settings | `scripts/psiconnect/run_st_temporal_importance.py` | temporal-importance and fidelity summaries |
| Connectivity attribution | attention/FC interpretation and deletion fidelity | `brainxdl/xai/st_fc_interpretation.py`, `brainxdl/xai/st_fidelity.py` | CLI settings | selected `scripts/psiconnect/` XAI commands | edge, network, and fidelity summaries |
| MM edge-mask explanation | optimized dense symmetric edge masks | `brainxdl/xai/explainer.py` | MM explanation CLI flags | `scripts/mm/train_mm.py --gnn_explain` | mask × adjacency saliency outputs |

## Dataset and command verification

- Legacy ST loaders cover the Power-atlas ASD/control format, with separate
  ABIDE AAL and COBRE AAL benchmark utilities. Which dataset corresponds to
  each final thesis table requires author confirmation.
- PsiConnect ST and MM commands require locally authorized data. Repeated
  sessions are grouped by participant in the PsiConnect workflows.
- The ST, MM, and XAI synthetic commands in `examples/` were executed in both
  validated environments. Dataset-backed commands were help-checked only.
- ST-BrainXDL is associated with the verified IJCAI 2024 publication recorded
  in `CITATION.cff`. MM-BrainXDL is associated with the verified Brain-MGF
  arXiv preprint metadata there. Applied XAI has no separate publication claim
  in this repository.
- `DenseGNNExplainer` is only a compatibility class name. The implementation is
  an optimized symmetric edge mask and signed mask-times-adjacency saliency;
  it is not standard PyG GNNExplainer and not gradient-times-activation.

No row should be treated as a reproduction claim until the command, dataset
version, five seed-2026 folds, and expected outputs have been independently
verified from this repository.
