#!/usr/bin/env python
"""ST-BrainXDL evaluation / smoke-test entry point.

Two modes:

1. Checkpoint analysis (default):
   Plot training curves and print final metrics from a saved .pth file.

       python scripts/st/st_evaluate.py --checkpoint path/to/model.pth

2. Fold evaluation:
   Load a checkpoint and run a forward pass on the test split of a specific
   fold to report live accuracy, BCE, and KLD.

       python scripts/st/st_evaluate.py \\
           --checkpoint path/to/model.pth \\
           --fold_eval \\
           --outer 1 --inner 1 \\
           --data_dir ./data/st_folds

Usage from repo root (brainxdl installed via pip install -e .):
    python scripts/st/st_evaluate.py --checkpoint runs/st/ST-BrainXDL_softmax_adv_fold11.pth
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from brainxdl.common.checkpointing import load_checkpoint
from brainxdl.common.device import pick_device
from brainxdl.st.dataset import STGraphDataset, load_fold_data, padseq
from brainxdl.st.model import STBrainXDL
from brainxdl.st.trainer import validate
from brainxdl.st.visualize import analyze_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained ST-BrainXDL checkpoint.")
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to .pth checkpoint.",
    )
    p.add_argument(
        "--fold_eval", action="store_true",
        help="Also run a live forward pass on the test fold.",
    )
    p.add_argument("--outer",    type=int, default=1)
    p.add_argument("--inner",    type=int, default=1)
    p.add_argument(
        "--data_dir", default="./data/st_folds",
        help="Directory with fold .pkl files (required for --fold_eval).",
    )
    p.add_argument(
        "--prefer_mps", action="store_true", default=True,
    )
    return p.parse_args()


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # --- Checkpoint analysis (always runs) ----------------------------------
    analyze_checkpoint(args.checkpoint)

    if not args.fold_eval:
        return

    # --- Live fold evaluation -----------------------------------------------
    print(f"\n[Eval] Loading fold outer={args.outer} inner={args.inner} …")
    _, test_graphs, _ = load_fold_data(
        args.outer, args.inner, data_dir=args.data_dir
    )
    test_dataset = STGraphDataset(test_graphs)
    test_loader  = DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False,
        num_workers=0, collate_fn=padseq, pin_memory=False,
    )

    DEVICE = pick_device(prefer_mps=args.prefer_mps)
    print(f"[Eval] device={DEVICE}")

    ckpt    = load_checkpoint(args.checkpoint, device=DEVICE)
    setting = ckpt["training_setting"]
    setting["device"]        = str(DEVICE)
    setting["sparse_on_cpu"] = True

    model = STBrainXDL(setting)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.apply_device_split(DEVICE, sparse_on_cpu=True)

    print("[Eval] Running test-set forward pass …")
    metrics = validate(model, setting, test_loader)

    print("\n[Eval] Test metrics:")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v.item():.4f}")


if __name__ == "__main__":
    main()
