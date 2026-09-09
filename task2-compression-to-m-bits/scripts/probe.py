#!/usr/bin/env python
"""Semantic evaluation: how much of the digit survives the m-bit bottleneck.

Distortion numbers say how many spike cells moved. They do not say whether the
recording is still recognisable, which is the question that matters for the
distributed-detection extension of task 3. Two probes, both following the
protocol of Skatchkovsky et al. (2021), who score naturalised spike trains by
feeding them to a classifier trained on clean data:

* `code probe`  logistic regression from the m-bit code to the digit label.
                Measures the information the encoder chose to keep.
* `recon probe` a small CNN trained on *original* time-summed N-MNIST frames,
                then evaluated on *reconstructed* ones. It never sees a
                reconstruction during training, so the accuracy drop is a clean
                measure of how much semantic content the codec destroyed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import NMNISTSpikes, default_data_root
from src.metrics import accumulated_frame
from scripts.visualize import load


class FrameCNN(nn.Module):
    """Small CNN on the 34x34 time-summed frame."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def norm_frame(x):
    """(B,T,2,H,W) spikes -> (B,1,H,W) frame, per-sample max-normalised."""
    f = accumulated_frame(x)[:, None]
    return f / f.flatten(1).max(1).values.clamp(min=1.0)[:, None, None, None]


def train_reference_cnn(root, n_bins, device, epochs=4):
    tr = NMNISTSpikes(root, n_bins, train=True)
    net = FrameCNN().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    dl = DataLoader(tr, 256, shuffle=True, num_workers=6, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs * len(dl))
    for ep in range(epochs):
        net.train()
        for x, y in dl:
            loss = nn.functional.cross_entropy(net(norm_frame(x.to(device))), y.to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        print(f"  reference CNN epoch {ep + 1}/{epochs}", flush=True)
    return net.eval()


@torch.no_grad()
def accuracy(net, loader, device, model=None, threshold=None):
    """Accuracy on originals (model=None) or on reconstructions."""
    correct = total = 0
    for x, y in loader:
        x = x.to(device)
        if model is not None:
            logits, _ = model(x)
            x = (torch.sigmoid(logits) > threshold).float()
        pred = net(norm_frame(x)).argmax(1).cpu()
        correct += (pred == y).sum().item(); total += len(y)
    return correct / total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--cnn-cache", type=Path, default=Path("runs/reference-cnn.pt"))
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = a.data_root or default_data_root()

    model, cfg, th, _ = load(a.ckpt, device)
    test = NMNISTSpikes(root, cfg.n_bins, train=False)
    test_dl = DataLoader(test, 256, num_workers=6)

    # --- code probe -------------------------------------------------------
    codes, labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            codes.append(model.encode(x.to(device)).cpu()); labels.append(y)
    codes = torch.cat(codes).numpy(); labels = torch.cat(labels).numpy()
    # the test set is stored class-ordered, so shuffle before splitting
    perm = np.random.default_rng(0).permutation(len(labels))
    codes, labels = codes[perm], labels[perm]
    from sklearn.linear_model import LogisticRegression
    split = int(0.7 * len(labels))
    probe = LogisticRegression(max_iter=3000).fit(codes[:split], labels[:split])
    code_acc = probe.score(codes[split:], labels[split:])

    # --- reconstruction probe --------------------------------------------
    if a.cnn_cache.exists():
        net = FrameCNN().to(device)
        net.load_state_dict(torch.load(a.cnn_cache, map_location=device))
        net.eval()
        print("loaded cached reference CNN")
    else:
        print("training reference CNN on original N-MNIST frames")
        net = train_reference_cnn(root, cfg.n_bins, device)
        a.cnn_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), a.cnn_cache)

    acc_orig = accuracy(net, test_dl, device)
    acc_recon = accuracy(net, test_dl, device, model, th)

    out = {"n_bits": cfg.n_bits, "code_probe_acc": code_acc,
           "cnn_acc_original": acc_orig, "cnn_acc_reconstructed": acc_recon,
           "semantic_retention": acc_recon / acc_orig}
    print(json.dumps(out, indent=2))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
