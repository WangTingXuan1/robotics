"""GPU architecture sweep on the fixed synthetic visuo-tactile dataset.

Select checkpoints by validation only. Test numbers are reported, not used to
pick an architecture. The existing transformer_experiment.py is untouched.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F

from experiment import CLASSES, OUT, SIZE, SITES

DEVICE = torch.device("cuda")
SWEEP = OUT / "sweep"
VIT_MODEL = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"


def get_vit_features(data):
    path = SWEEP / "vit_tiny_features_gpu.npy"
    n = len(data["label"])
    if path.exists():
        cached = np.load(path)
        if cached.shape == (n, 192):
            return cached
    print(f"Extracting {VIT_MODEL} features on {DEVICE}...", flush=True)
    vit = timm.create_model(VIT_MODEL, pretrained=True, num_classes=0).eval().to(DEVICE)
    for parameter in vit.parameters():
        parameter.requires_grad_(False)
    images = data["vision"]
    features = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    with torch.inference_mode():
        for start in range(0, n, 16):
            batch = torch.from_numpy(images[start:start+16].copy()).permute(0, 3, 1, 2).float().to(DEVICE)/255
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
            features.append(vit((batch-mean)/std).cpu().numpy())
    del vit
    torch.cuda.empty_cache()
    features = np.concatenate(features).astype(np.float32)
    if features.shape != (n, 192):
        raise ValueError(f"Unexpected ViT feature shape: {features.shape}")
    np.save(path, features)
    return features


class CNNEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class PatchTransformer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.patch = nn.Conv2d(channels, 64, kernel_size=8, stride=8)
        self.pos = nn.Parameter(torch.zeros(1, 36, 64))
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        z = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        return self.norm(self.encoder(z).mean(1))


class Model(nn.Module):
    def __init__(self, visual, tactile, fusion):
        super().__init__()
        self.visual_kind, self.tactile_kind, self.fusion_kind = visual, tactile, fusion
        if visual == "cnn":
            self.visual = CNNEncoder(3)
        elif visual == "patch_transformer":
            self.visual = PatchTransformer(3)
        elif visual == "pretrained_vit":
            self.visual = nn.Sequential(nn.Linear(192, 64), nn.LayerNorm(64), nn.GELU())
        if tactile == "cnn":
            self.tactile = CNNEncoder(4)
        elif tactile == "patch_transformer":
            self.tactile = PatchTransformer(4)
        if visual != "none" and tactile != "none":
            if fusion == "concat":
                self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                          nn.Dropout(0.1), nn.Linear(64, len(CLASSES)))
            else:
                self.cls = nn.Parameter(torch.zeros(1, 1, 64))
                self.type_embed = nn.Parameter(torch.zeros(1, 3, 64))
                layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1,
                                                   batch_first=True, norm_first=True)
                self.fuser = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
                self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, len(CLASSES)))
        else:
            self.head = nn.Linear(64, len(CLASSES))

    def forward(self, vision, touch, vit):
        v = None
        t = None
        if self.visual_kind == "pretrained_vit":
            v = self.visual(vit)
        elif self.visual_kind != "none":
            v = self.visual(vision)
        if self.tactile_kind != "none":
            t = self.tactile(touch)
        if v is None:
            return self.head(t)
        if t is None:
            return self.head(v)
        if self.fusion_kind == "concat":
            return self.head(torch.cat((v, t), dim=1))
        tokens = torch.stack((v, t), 1)
        cls = self.cls.expand(len(tokens), -1, -1)
        tokens = torch.cat((cls, tokens), 1) + self.type_embed
        return self.head(self.fuser(tokens)[:, 0])


def load_tensors():
    data = np.load(OUT / "dataset.npz")
    n, sites = len(data["label"]), len(SITES)
    idx = np.repeat(np.arange(n), sites)
    site = np.tile(np.arange(sites), n)
    vision = torch.from_numpy(data["vision"][idx].copy()).permute(0, 3, 1, 2).float().to(DEVICE)/255
    touch = torch.from_numpy(data["touch"][idx, site].copy()).permute(0, 3, 1, 2).float().to(DEVICE)/255
    plane = torch.tensor(site, device=DEVICE).float().view(-1, 1, 1, 1).expand(-1, 1, SIZE, SIZE)/(sites-1)
    touch = torch.cat((touch, plane), dim=1)
    features = get_vit_features(data)
    vit = torch.from_numpy(features[idx].copy()).float().to(DEVICE)
    labels = torch.from_numpy(data["label"][idx].copy()).long().to(DEVICE)
    split = torch.from_numpy(data["split"][idx].copy()).to(DEVICE)
    return vision, touch, vit, labels, split


def variants():
    rows = []
    for visual in ("cnn", "patch_transformer", "pretrained_vit"):
        rows.append((visual, "none", "none"))
    for tactile in ("cnn", "patch_transformer"):
        rows.append(("none", tactile, "none"))
    for visual in ("cnn", "patch_transformer", "pretrained_vit"):
        for tactile in ("cnn", "patch_transformer"):
            for fusion in ("concat", "attention"):
                rows.append((visual, tactile, fusion))
    return rows


def score(model, tensors, ids):
    model.eval()
    v, t, f, y, _ = tensors
    logits = []
    with torch.inference_mode():
        for batch in ids.split(128):
            logits.append(model(v[batch], t[batch], f[batch]))
        logits = torch.cat(logits)
        accuracy = (logits.argmax(1) == y[ids]).float().mean().item()
        loss = nn.functional.cross_entropy(logits, y[ids]).item()
    return accuracy, loss


def train_variant(spec, tensors, seed, max_epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(*spec).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    train_ids = torch.where(tensors[4] == 0)[0]
    val_ids = torch.where(tensors[4] == 1)[0]
    test_ids = torch.where(tensors[4] == 2)[0]
    best_acc, best_loss, best_epoch, no_gain = -1.0, float("inf"), 0, 0
    name = "__".join(spec)
    path = SWEEP / f"{name}__seed{seed}.pt"
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = train_ids[torch.randperm(len(train_ids), device=DEVICE)]
        for batch in order.split(64):
            opt.zero_grad(set_to_none=True)
            logits = model(tensors[0][batch], tensors[1][batch], tensors[2][batch])
            loss = nn.functional.cross_entropy(logits, tensors[3][batch])
            loss.backward()
            opt.step()
        val_acc, val_loss = score(model, tensors, val_ids)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, best_epoch, no_gain = val_acc, val_loss, epoch, 0
            torch.save(model.state_dict(), path)
        else:
            no_gain += 1
        if no_gain >= patience:
            break
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    test_acc, test_loss = score(model, tensors, test_ids)
    torch.cuda.synchronize()
    return {
        "visual": spec[0], "tactile": spec[1], "fusion": spec[2],
        "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
        "validation_accuracy": best_acc, "validation_loss": best_loss,
        "test_accuracy": test_acc, "test_loss": test_loss,
        "train_seconds": round(time.perf_counter() - started, 2),
        "checkpoint": str(path.relative_to(OUT)),
    }


def summarize(runs):
    result = []
    for spec in variants():
        chosen = [r for r in runs if (r["visual"], r["tactile"], r["fusion"]) == spec]
        result.append({
            "visual": spec[0], "tactile": spec[1], "fusion": spec[2],
            "validation_mean": float(np.mean([r["validation_accuracy"] for r in chosen])),
            "validation_std": float(np.std([r["validation_accuracy"] for r in chosen], ddof=1)),
            "test_mean": float(np.mean([r["test_accuracy"] for r in chosen])),
            "test_std": float(np.std([r["test_accuracy"] for r in chosen], ddof=1)),
            "train_seconds_mean": float(np.mean([r["train_seconds"] for r in chosen])),
        })
    # Architecture selection uses validation performance only.
    result.sort(key=lambda r: (-r["validation_mean"], r["validation_std"], r["visual"], r["tactile"], r["fusion"]))
    return result


def write_outputs(runs, summary, seeds, epochs, patience):
    selected = summary[0]
    report = {
        "dataset": "results/dataset.npz", "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "n_train_scenes": 90, "n_validation_scenes": 30, "n_test_scenes": 30,
        "n_test_pairs": 90, "seeds": seeds, "max_epochs": epochs,
        "early_stop_patience": patience,
        "selection_rule": "Highest mean validation pair accuracy across seeds; test is not used for selection",
        "selected": selected, "summary": summary, "runs": runs,
    }
    (SWEEP / "report.json").write_text(json.dumps(report, indent=2))
    with (SWEEP / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    plot_rows = summary[:10]
    labels = [f"{r['visual']} + {r['tactile']} + {r['fusion']}" for r in plot_rows]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(plot_rows))
    ax.barh(y, [r["test_mean"] for r in plot_rows], xerr=[r["test_std"] for r in plot_rows],
            color="#4e79a7", capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test pair accuracy, mean ± SD across seeds")
    ax.set_title("Top 10 architectures ranked by validation accuracy")
    fig.tight_layout()
    fig.savefig(SWEEP / "comparison.png", dpi=170)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this sweep")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    SWEEP.mkdir(parents=True, exist_ok=True)
    tensors = load_tensors()
    print("Training on", torch.cuda.get_device_name(0), flush=True)
    print(f"{len(variants())} variants × {len(args.seeds)} seeds", flush=True)
    runs = []
    for spec in variants():
        for seed in args.seeds:
            row = train_variant(spec, tensors, seed, args.epochs, args.patience)
            runs.append(row)
            print(f"{' / '.join(spec)} seed={seed}: val={row['validation_accuracy']:.3f}, "
                  f"test={row['test_accuracy']:.3f}, epoch={row['best_epoch']}", flush=True)
        summary = summarize(runs) if len(runs) == len(variants())*len(args.seeds) else None
    report = write_outputs(runs, summary, args.seeds, args.epochs, args.patience)
    print("SELECTED BY VALIDATION:", report["selected"], flush=True)


if __name__ == "__main__":
    main()
