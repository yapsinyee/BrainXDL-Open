# Synthetic quick starts

After installing the package, run:

```bash
python examples/st_synthetic.py
python examples/mm_synthetic.py
python examples/xai_synthetic.py
```

These examples use generated matrices, do not download research data, and do
not reproduce thesis results. The XAI example uses BrainXDL's optimized dense
symmetric edge-mask method. It is neither standard PyG GNNExplainer nor
gradient-times-activation.
