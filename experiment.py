"""Course experiment 1: simulated visuo-tactile active recognition.

PyBullet supplies the RGB scene; TACTO renders procedural contact depth maps.
The contact maps are controlled surface models, not force-based collision results.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import pybullet as p
import tacto
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CLASSES = ["smooth", "ridges", "bumps"]
SITES = ["plain", "ridge_probe", "bump_probe"]
SIZE = 48
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def tactile_depth(label, site, rng):
    """Make a local depth field for a controlled contact at one surface site."""
    y, x = np.mgrid[-1:1:48j, -1:1:48j]
    dx, dy = rng.normal(0, 0.055, 2)
    x, y = x - dx, y - dy
    base = 0.42 * np.exp(-((x / 0.76) ** 4 + (y / 0.76) ** 4))
    pattern = np.zeros_like(base)
    if label == 1 and site == 1:
        pattern = 0.18 * (0.5 + 0.5 * np.cos(17 * x + rng.uniform(-0.5, 0.5)))
    elif label == 2 and site == 2:
        for cx in (-0.48, 0, 0.48):
            for cy in (-0.48, 0, 0.48):
                pattern += 0.21 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.012)
    depth = base + pattern * (base > 0.1)
    depth += rng.normal(0, 0.012, depth.shape)
    return np.clip(depth, 0, 1).astype(np.float32)


def visual_image(cid, label, rng):
    """Render a common-looking block in PyBullet, with a weak color cue."""
    p.resetSimulation(physicsClientId=cid)
    color = np.clip(0.54 + 0.025 * label + rng.normal(0, 0.055), 0.3, 0.78)
    half = [0.30, 0.30, 0.06]
    shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                rgbaColor=[color, color, color, 1], physicsClientId=cid)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                      basePosition=[0, 0, 0], physicsClientId=cid)
    eye_x, eye_y = rng.normal(0, 0.06, 2)
    view = p.computeViewMatrix([eye_x, eye_y, 1.4], [0, 0, 0], [0, 1, 0])
    proj = p.computeProjectionMatrixFOV(45, 1, 0.1, 4)
    _, _, rgba, _, _ = p.getCameraImage(SIZE, SIZE, view, proj,
                                        renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
    return np.asarray(rgba, dtype=np.uint8).reshape(SIZE, SIZE, 4)[:, :, :3]


def generate(n_per_class=50):
    set_seed()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    cid = p.connect(p.DIRECT)
    renderer = tacto.Renderer(width=SIZE, height=SIZE, background=None,
                              config_path=tacto.get_digit_config_path())
    images, touches, labels, splits = [], [], [], []
    try:
        for label in range(len(CLASSES)):
            for j in range(n_per_class):
                images.append(visual_image(cid, label, rng))
                row = []
                for site in range(len(SITES)):
                    depth = tactile_depth(label, site, rng)
                    color, _ = renderer.render_from_depth(depth, scale=0.01)
                    row.append(np.asarray(color, dtype=np.uint8))
                touches.append(row)
                labels.append(label)
                # Disjoint objects/scenes: 60% train, 20% validation, 20% test.
                splits.append(0 if j < 0.6*n_per_class else 1 if j < 0.8*n_per_class else 2)
                if (j + 1) % 10 == 0:
                    print(f"generated {CLASSES[label]} {j+1}/{n_per_class}", flush=True)
    finally:
        renderer.r.delete()
        p.disconnect(cid)
    path = OUT / "dataset.npz"
    np.savez_compressed(path, vision=np.stack(images), touch=np.stack(touches),
                        label=np.array(labels), split=np.array(splits))
    print(f"saved {path}: {len(labels)} scenes, {len(labels)*len(SITES)} tactile samples")


class Pairs(Dataset):
    def __init__(self, data, split, mode):
        self.v = data["vision"]
        self.t = data["touch"]
        self.y = data["label"]
        self.ids = np.where(data["split"] == split)[0]
        self.mode = mode
        self.length = len(self.ids) if mode == "vision" else len(self.ids)*len(SITES)

    def __len__(self):
        return self.length

    def __getitem__(self, k):
        i = self.ids[k] if self.mode == "vision" else self.ids[k//len(SITES)]
        site = 0 if self.mode == "vision" else k % len(SITES)
        v = torch.from_numpy(self.v[i].copy()).permute(2, 0, 1).float()/255
        t = torch.from_numpy(self.t[i, site].copy()).permute(2, 0, 1).float()/255
        site_plane = torch.full((1, SIZE, SIZE), site / (len(SITES)-1))
        return v, torch.cat((t, site_plane)), int(self.y[i])


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten())

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode in ("vision", "fusion"):
            self.vision = Encoder(3)
        if mode in ("touch", "fusion"):
            self.touch = Encoder(4)
        self.head = nn.Linear(96 if mode == "fusion" else 48, len(CLASSES))

    def forward(self, vision, touch):
        if self.mode == "vision":
            features = self.vision(vision)
        elif self.mode == "touch":
            features = self.touch(touch)
        else:
            features = torch.cat((self.vision(vision), self.touch(touch)), dim=1)
        return self.head(features)


def train(epochs=8):
    set_seed()
    OUT.mkdir(exist_ok=True)
    data = np.load(OUT / "dataset.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training baseline models on {device}", flush=True)
    summary = {}
    for mode in ("vision", "touch", "fusion"):
        model = Classifier(mode).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.002)
        train_loader = DataLoader(Pairs(data, 0, mode), batch_size=32, shuffle=True)
        val_loader = DataLoader(Pairs(data, 1, mode), batch_size=64)
        best = -1
        for epoch in range(epochs):
            model.train()
            for v, t, y in train_loader:
                v, t, y = v.to(device), t.to(device), y.to(device)
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(v, t), y)
                loss.backward()
                opt.step()
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for v, t, y in val_loader:
                    v, t, y = v.to(device), t.to(device), y.to(device)
                    pred = model(v, t).argmax(1)
                    correct += int((pred == y).sum())
                    total += len(y)
            acc = correct / total
            if acc > best:
                best = acc
                torch.save(model.state_dict(), OUT / f"{mode}.pt")
            print(f"{mode} epoch {epoch+1}/{epochs}: val accuracy {acc:.3f}", flush=True)
        summary[mode] = best
    (OUT / "training.json").write_text(json.dumps(summary, indent=2))


def predict_all(data):
    probs = {}
    for mode in ("vision", "touch", "fusion"):
        model = Classifier(mode)
        model.load_state_dict(torch.load(OUT / f"{mode}.pt", map_location="cpu", weights_only=True))
        model.eval()
        pred = []
        loader = DataLoader(Pairs(data, 2, mode), batch_size=64)
        with torch.no_grad():
            for v, t, _ in loader:
                pred.append(model(v, t).softmax(1).numpy())
        pred = np.concatenate(pred)
        if mode == "vision":
            probs[mode] = pred
        else:
            probs[mode] = pred.reshape(-1, len(SITES), len(CLASSES))
    return probs


def site_scores(data):
    """Class-pair separability at each site, estimated on training data only."""
    touch = data["touch"][data["split"] == 0].astype(np.float32) / 255
    labels = data["label"][data["split"] == 0]
    means = np.stack([touch[labels == c].mean(0) for c in range(len(CLASSES))])
    scores = np.zeros((len(SITES), len(CLASSES), len(CLASSES)))
    for s in range(len(SITES)):
        for a in range(len(CLASSES)):
            for b in range(len(CLASSES)):
                scores[s, a, b] = np.mean((means[a, s]-means[b, s])**2)
    return scores


def explore(vprob, tprob, scores, random=False, seed=0, threshold=0.85):
    rng = np.random.default_rng(seed)
    posterior = vprob.copy()
    remaining = list(range(len(SITES)))
    history = []
    for _ in range(2):
        if random:
            site = int(rng.choice(remaining))
        else:
            values = [sum(posterior[a]*posterior[b]*scores[s, a, b]
                          for a in range(len(CLASSES)) for b in range(a+1, len(CLASSES)))
                      for s in remaining]
            site = remaining[int(np.argmax(values))]
        remaining.remove(site)
        # Visual prior plus one observation. Softmax probabilities are approximate
        # likelihoods; they are not a calibrated Bayesian sensor model.
        posterior = posterior * np.maximum(tprob[site], 1e-5)
        posterior /= posterior.sum()
        history.append((site, posterior.copy()))
        if posterior.max() >= threshold:
            break
    return history


def evaluate():
    set_seed()
    data = np.load(OUT / "dataset.npz")
    ids = np.where(data["split"] == 2)[0]
    labels = data["label"][ids]
    probs = predict_all(data)
    scores = site_scores(data)
    metrics = {
        "n_test": len(ids),
        "vision_accuracy": float(np.mean(probs["vision"].argmax(1) == labels)),
        "touch_accuracy_all_sites": float(np.mean(probs["touch"].argmax(2) == labels[:, None])),
        "fusion_accuracy_all_sites": float(np.mean(probs["fusion"].argmax(2) == labels[:, None])),
    }
    runs = {}
    for strategy in ("active", "random"):
        histories = [explore(probs["vision"][i], probs["touch"][i], scores,
                             random=strategy == "random", seed=SEED+i) for i in range(len(ids))]
        runs[strategy] = histories
        metrics[strategy] = {
            "accuracy": float(np.mean([h[-1][1].argmax() == labels[i] for i, h in enumerate(histories)])),
            "mean_contacts": float(np.mean([len(h) for h in histories])),
            "accuracy_after_first_contact": float(np.mean([h[0][1].argmax() == labels[i] for i, h in enumerate(histories)])),
        }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ["Vision", "Touch", "Fusion"]
    vals = [metrics["vision_accuracy"], metrics["touch_accuracy_all_sites"], metrics["fusion_accuracy_all_sites"]]
    axes[0].bar(names, vals, color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Modality comparison")
    for strategy in ("random", "active"):
        h = runs[strategy]
        after_one = np.mean([x[0][1].argmax() == labels[i] for i, x in enumerate(h)])
        after_two = np.mean([x[-1][1].argmax() == labels[i] for i, x in enumerate(h)])
        axes[1].plot([0, 1, 2], [metrics["vision_accuracy"], after_one, after_two],
                     marker="o", label=strategy)
    axes[1].set_xticks([0, 1, 2]); axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Maximum contacts"); axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Exploration comparison"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=160); plt.close(fig)
    demo(data, ids[0], probs["vision"][0], runs["active"][0])
    print(json.dumps(metrics, indent=2))


def demo(data, sample_id, visual_prob, history):
    """Save a self-contained animated GIF showing one complete decision."""
    frames = []
    steps = [(None, visual_prob)] + history
    for step, (site, prob) in enumerate(steps):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].imshow(data["vision"][sample_id]); axes[0].set_title("Visual input")
        axes[0].axis("off")
        if site is None:
            axes[1].text(0.5, 0.5, "No contact yet", ha="center", va="center")
        else:
            axes[1].imshow(data["touch"][sample_id, site]); axes[1].set_title(f"Contact: {SITES[site]}")
        axes[1].axis("off")
        axes[2].bar(CLASSES, prob, color="#4e79a7"); axes[2].set_ylim(0, 1)
        axes[2].set_title(f"Step {step}: predicts {CLASSES[int(prob.argmax())]}")
        axes[2].tick_params(axis="x", rotation=25)
        fig.suptitle(f"True class: {CLASSES[int(data['label'][sample_id])]}")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    frames[0].save(OUT / "demo.gif", save_all=True, append_images=frames[1:],
                   duration=1400, loop=0)
    imageio.mimsave(OUT / "demo.mp4", [np.asarray(frame) for frame in frames],
                   fps=1, macro_block_size=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["generate", "train", "evaluate", "all"])
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.stage in ("generate", "all"):
        generate(args.samples_per_class)
    if args.stage in ("train", "all"):
        train(args.epochs)
    if args.stage in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
