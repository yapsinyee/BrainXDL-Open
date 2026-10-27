"""ST-BrainXDL: Checkpoint analysis and loss-curve plotting.

Run as a CLI tool:
    python -m brainxdl.st.visualize path/to/checkpoint.pth

Or import analyze_checkpoint for notebook use.

Source: extracted from legacy_code/1-ST/visualize.py.
Changes: load_checkpoint replaced by brainxdl.common.checkpointing.load_checkpoint.
"""

import argparse
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "brainxdl-matplotlib")
)
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "brainxdl-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use(os.environ["MPLBACKEND"], force=False)
import matplotlib.pyplot as plt
import torch

from brainxdl.common.checkpointing import load_checkpoint


def plot_loss_curves(losses, labels, title, xlabel="Epochs", ylabel="Loss"):
    """Plot one or more loss curves on a single figure.

    Args:
        losses: List of lists/tensors, one per curve.
        labels: Display label for each curve.
        title:  Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    plt.figure(figsize=(10, 6))
    for loss, label in zip(losses, labels):
        plt.plot(torch.stack(loss, dim=0).cpu().numpy(), label=label)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()


def analyze_checkpoint(checkpoint_path: str) -> None:
    """Load a checkpoint and plot training curves + print final metrics.

    Args:
        checkpoint_path: Path to a .pth checkpoint saved by trainer.py.
    """
    ckpt    = load_checkpoint(checkpoint_path)
    setting = ckpt["training_setting"]
    train_l = ckpt["train_losses"]
    test_l  = ckpt["test_losses"]

    plot_loss_curves(
        [train_l["z_KLD"], test_l["z_KLD"]],
        ["Train", "Test"],
        "KL Divergence Loss (z)",
    )
    plot_loss_curves(
        [train_l["a_NLL"], test_l["a_NLL"]],
        ["Train", "Test"],
        "Adjacency NLL (Graph Reconstruction)",
    )
    plot_loss_curves(
        [train_l["y_BCE"], test_l["y_BCE"]],
        ["Train", "Test"],
        "Classification BCE",
    )
    plot_loss_curves(
        [train_l["y_ACC"], test_l["y_ACC"]],
        ["Train", "Test"],
        "Classification Accuracy",
        ylabel="Accuracy",
    )

    print(f"\n--- Checkpoint: {checkpoint_path} ---")
    print("Training settings:")
    for k, v in setting.items():
        print(f"  {k}: {v}")

    if test_l["y_BCE"]:
        final_bce = test_l["y_BCE"][-1].item()
        final_acc = test_l["y_ACC"][-1].item()
        print(f"\nFinal BCE (test):      {final_bce:.4f}")
        print(f"Final Accuracy (test): {final_acc:.4f}")
        print(f"Final Error Rate:      {1 - final_acc:.4f}")
    else:
        print("\nNo test loss records in checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse a ST-BrainXDL training checkpoint."
    )
    parser.add_argument("checkpoint_path", help="Path to .pth checkpoint file.")
    args = parser.parse_args()
    analyze_checkpoint(args.checkpoint_path)
