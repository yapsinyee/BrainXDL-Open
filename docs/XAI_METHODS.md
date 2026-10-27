# Explainability Methods

These method descriptions correspond to the BrainXDL-Open implementations and
avoid substituting names of superficially similar external algorithms.

## ST graph attention

`TransformerAttentionCapture` temporarily wraps `TransformerConv` calls and
records returned edge attention without modifying checkpoint parameters. The
records can be converted into component/time series and aggregated at edge,
region, and network levels.

## ST temporal and connectivity perturbation

Temporal importance replaces selected non-overlapping time blocks and measures
changes in validation-threshold-centered prediction margins. Connectivity
fidelity masks selected undirected edges consistently across dense features,
dense adjacency, and sparse graph topology.

## MM optimized dense edge masks

The compatibility class `DenseGNNExplainer`:

1. freezes all predictive-model parameters;
2. initializes symmetric dense edge-mask logits per enabled modality;
3. optimizes the masks for the selected prediction with size and entropy
   regularization;
4. multiplies the optimized mask by the model adjacency to obtain signed
   saliency.

This project-specific method is not the standard PyG GNNExplainer algorithm.
The migrated code does not compute gradient × activation attribution.
