import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt


def _load_points(metrics_file):
    with open(metrics_file) as f:
        data = json.load(f)
    steps = []
    losses = []
    for row in data:
        if "val_loss" in row and "step" in row:
            steps.append(row["step"])
            losses.append(row["val_loss"])
    return steps, losses


def plot_loss(metrics_file, plot_file, title, baseline_file=None):
    steps, losses = _load_points(metrics_file)
    plt.figure(figsize=(8, 5))
    if steps:
        plt.plot(steps, losses, marker="o", label="validation")
    if baseline_file:
        b_steps, b_losses = _load_points(baseline_file)
        if b_steps:
            plt.plot(b_steps, b_losses, linestyle="--", label="baseline")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
