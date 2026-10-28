"""Evaluate predeclared three-seed ensembles on the fixed held-out split."""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch

import model_sweep as sweep


SEEDS = (42, 43, 44)
SPECS = {
    "CNN + CNN + concat": ("cnn", "cnn", "concat"),
    "pretrained ViT + CNN + concat": ("pretrained_vit", "cnn", "concat"),
}


def ensemble_probs(spec, tensors, ids):
    probs = []
    for seed in SEEDS:
        model = sweep.Model(*spec).to(sweep.DEVICE)
        path = sweep.SWEEP / ("__".join(spec) + f"__seed{seed}.pt")
        model.load_state_dict(torch.load(path, map_location=sweep.DEVICE, weights_only=True))
        model.eval()
        with torch.inference_mode():
            logits = model(tensors[0][ids], tensors[1][ids], tensors[2][ids])
            probs.append(logits.softmax(1).cpu().numpy())
    return np.stack(probs).mean(0)


def main():
    tensors = sweep.load_tensors()
    ids = torch.where(tensors[4] == 2)[0]
    labels = tensors[3][ids].cpu().numpy()
    correct = {}
    report = {"device": str(sweep.DEVICE), "gpu": torch.cuda.get_device_name(0),
              "seeds": list(SEEDS), "n_test_scenes": 30, "n_test_pairs": 90,
              "ensembles": {}}
    for name, spec in SPECS.items():
        pred = ensemble_probs(spec, tensors, ids).argmax(1)
        correct[name] = (pred == labels).reshape(30, 3)
        report["ensembles"][name] = float(correct[name].mean())
    # Paired bootstrap resamples scenes, preserving all three contact sites.
    rng = np.random.default_rng(2026)
    n = 10000
    sample = rng.integers(0, 30, size=(n, 30))
    names = list(SPECS)
    diff = correct[names[1]][sample].mean((1, 2)) - correct[names[0]][sample].mean((1, 2))
    report["vit_minus_cnn"] = float(correct[names[1]].mean() - correct[names[0]].mean())
    report["paired_scene_bootstrap_95pct_interval"] = np.quantile(diff, [0.025, 0.975]).tolist()
    baseline = json.loads((sweep.OUT / "metrics.json").read_text())
    report["original_gpu_cnn_baseline"] = {
        "vision_accuracy": baseline["vision_accuracy"],
        "touch_accuracy": baseline["touch_accuracy_all_sites"],
        "fusion_accuracy": baseline["fusion_accuracy_all_sites"],
        "active_accuracy": baseline["active"]["accuracy"],
        "random_accuracy": baseline["random"]["accuracy"],
    }
    (sweep.SWEEP / "ensemble_report.json").write_text(json.dumps(report, indent=2))
    labels_plot = ["original CNN fusion\n(single seed)", "CNN ensemble", "ViT + CNN ensemble"]
    values = [baseline["fusion_accuracy_all_sites"],
              report["ensembles"][names[0]], report["ensembles"][names[1]]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels_plot, values, color=["#9c9c9c", "#4e79a7", "#f28e2b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out pair accuracy (90 pairs, 30 scenes)")
    ax.set_title("Single seed and three-seed ensembles")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.02, f"{value:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(sweep.SWEEP / "ensemble_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
