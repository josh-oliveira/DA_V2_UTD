"""
plot_training.py
----------------
Graphs a training_log.json produced by train.py.

Usage:
    python plot_training.py --log /path/to/training_log.json

Requirements:
    pip install matplotlib
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def plot(log_path: Path):
    with open(log_path) as f:
        log = json.load(f)

    epochs      = [e["epoch"]              for e in log["epochs"]]
    train_loss  = [e["train"]["loss"]      for e in log["epochs"]]
    val_loss    = [e["val"]["loss"]        for e in log["epochs"]]
    mae         = [e["val"]["mae"]         for e in log["epochs"]]
    rmse        = [e["val"]["rmse"]        for e in log["epochs"]]
    abs_rel     = [e["val"]["abs_rel"]     for e in log["epochs"]]
    delta1      = [e["val"]["delta1"]      for e in log["epochs"]]

    losses      = log["config"]["losses"]
    train_comps = {k: [e["train"].get(k, 0) for e in log["epochs"]] for k in losses}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Training — encoder={log['config']['encoder']}  "
        f"strategy={log['config']['strategy']}  "
        f"epochs={log['config']['epochs']}",
        fontsize=12,
    )

    # ── Plot 1: Train vs Val loss ──
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train loss", linewidth=2)
    ax.plot(epochs, val_loss,   label="val loss",   linewidth=2, linestyle="--")
    ax.set_title("Total loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Loss components (train) ──
    ax = axes[0, 1]
    for name, values in train_comps.items():
        ax.plot(epochs, values, label=name, linewidth=2)
    ax.set_title("Train loss components")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Val error metrics ──
    ax = axes[1, 0]
    ax.plot(epochs, mae,     label="MAE",     linewidth=2)
    ax.plot(epochs, rmse,    label="RMSE",    linewidth=2, linestyle="--")
    ax.plot(epochs, abs_rel, label="AbsRel",  linewidth=2, linestyle=":")
    ax.set_title("Val error metrics (lower is better)")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Delta1 accuracy ──
    ax = axes[1, 1]
    ax.plot(epochs, delta1, color="green", linewidth=2, label="delta1 (%)")
    ax.set_title("Val delta1 accuracy (higher is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = log_path.parent / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to training_log.json")
    args = parser.parse_args()
    plot(Path(args.log).resolve())