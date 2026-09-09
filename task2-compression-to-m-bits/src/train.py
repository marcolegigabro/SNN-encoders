"""Training and evaluation for the m-bit spike-train compressor."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .data import NMNISTSpikes, default_data_root
from .losses import SpikeReconstructionLoss
from .metrics import (
    code_usage,
    image_metrics,
    pick_threshold,
    spike_agreement,
    van_rossum,
)
from .model import SpikeCompressor


@dataclass
class Config:
    n_bits: int = 128
    n_bins: int = 16
    width: int = 32
    hidden: int = 256
    beta: float = 0.9
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 1.0
    w_bce: float = 1.0
    w_dice: float = 1.0
    w_psp: float = 20.0
    w_frame: float = 0.3
    tau: float = 3.0
    train_subset: int = 0  # 0 = full 60k
    seed: int = 0
    workers: int = 6


def make_loaders(cfg: Config, data_root: Path):
    train = NMNISTSpikes(data_root, cfg.n_bins, train=True)
    test = NMNISTSpikes(data_root, cfg.n_bins, train=False)
    if cfg.train_subset:
        g = torch.Generator().manual_seed(cfg.seed)
        idx = torch.randperm(len(train), generator=g)[: cfg.train_subset]
        train = Subset(train, idx.tolist())
    common = dict(num_workers=cfg.workers, pin_memory=True,
                  persistent_workers=cfg.workers > 0)
    return (
        DataLoader(train, cfg.batch_size, shuffle=True, drop_last=True, **common),
        DataLoader(test, cfg.batch_size, shuffle=False, **common),
    )


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, threshold: float,
             max_batches: int | None = None, seed: int = 0):
    """Evaluate the three ways of reading out the decoder.

    The decoder does not emit a spike train, it emits a Bernoulli intensity
    p(spike) per cell. How that is turned back into spikes changes the numbers
    more than most architecture choices do, so all three readouts are reported:

    * `soft`   use p itself, i.e. reconstruct a spike *rate* rather than a
               spike train. Not an admissible answer to task 2, but it is the
               decoder's actual output and the bound the others sit under.
    * `sample` x_hat ~ Bernoulli(p). A genuine binary spike train whose
               first-order statistics match the model's belief. This is the
               headline reconstruction.
    * `hard`   threshold p at the value that maximises F1. Best cell-wise
               agreement by construction, but it over-predicts badly (a diffuse
               intensity thresholded low becomes a blob), which shows up as a
               van Rossum distance far worse than the other two.

    Metrics are accumulated batch by batch and averaged with sample weights
    rather than materialising the whole test set: 10k samples of probabilities
    at T=16 is 1.5 GB, and the PSP filter would need two more copies of that.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    acc, n = {}, 0
    code_all = []
    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        logits, b = model(x)
        loss, _ = loss_fn(logits, x)
        p = torch.sigmoid(logits)

        readouts = {
            "soft": p,
            "sample": torch.bernoulli(p, generator=gen),
            "hard": (p > threshold).float(),
        }
        batch = {"loss": loss.item(), "true_occupancy": x.mean().item()}
        for tag, pred in readouts.items():
            m = spike_agreement(pred, x)
            m["van_rossum"] = van_rossum(pred, x, loss_fn.tau)
            m.update(image_metrics(pred, x))
            m["occupancy"] = pred.mean().item()
            batch.update({f"{tag}_{k}": v for k, v in m.items()})

        bs = x.shape[0]
        for k, v in batch.items():
            acc[k] = acc.get(k, 0.0) + v * bs
        n += bs
        code_all.append(b.cpu())

    out = {k: v / n for k, v in acc.items()}
    out["threshold"] = threshold
    out.update(code_usage(torch.cat(code_all)))
    return out


def train(cfg: Config, out_dir: Path, data_root: Path | None = None,
          device: str = "cuda"):
    torch.manual_seed(cfg.seed)
    data_root = data_root or default_data_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = make_loaders(cfg, data_root)
    model = SpikeCompressor(cfg.n_bits, cfg.n_bins, cfg.width, cfg.hidden,
                            cfg.beta).to(device)
    loss_fn = SpikeReconstructionLoss(cfg.pos_weight, cfg.w_bce, cfg.w_dice,
                                      cfg.w_psp, cfg.w_frame, cfg.tau).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(train_loader),
        pct_start=0.15,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"m = {cfg.n_bits} bits | block = {cfg.n_bins * 2 * 34 * 34} bits "
          f"| ratio = {model.compression_ratio():.0f}x | {n_params / 1e6:.2f}M params",
          flush=True)
    print(f"train batches/epoch = {len(train_loader)}", flush=True)

    history = []
    best_score = -1.0
    # Selection metric, chosen twice over.
    #
    # Not F1: a zero-bit baseline that always emits the dataset-average spike
    # train already scores F1 = 0.380 on this data (scripts/baselines.py), so
    # F1 barely separates a real codec from a constant one, and it rewards
    # over-prediction on top of that.
    #
    # Not the *soft* readout either. Measured over a run, soft_frame_corr peaks
    # at epoch 1 (0.911) and then declines to 0.877 while every metric of the
    # sampled spike train keeps improving (van Rossum 1.096 -> 0.875, F1
    # 0.167 -> 0.262): the model is trading a diffuse intensity that correlates
    # well in aggregate for a sharp one that places individual spikes well.
    # Selecting on the soft readout would therefore save an untrained model.
    # The deliverable of task 2 is a spike train, so we select on the sampled
    # one.
    SELECT = "sample_frame_corr"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        run = {}
        for step, (x, _) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            logits, _ = model(x)
            loss, parts = loss_fn(logits, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in ({"total": loss.item()} | parts).items():
                run[k] = run.get(k, 0.0) + v
        run = {k: v / len(train_loader) for k, v in run.items()}

        # threshold fitted on training batches, then applied to the test set
        th, _ = _fit_threshold(model, train_loader, device, n_batches=8)
        stats = evaluate(model, test_loader, loss_fn, device, threshold=th)
        stats["epoch"] = epoch
        stats["train_loss"] = run["total"]
        stats["secs"] = time.time() - t0
        history.append(stats)

        print(
            f"ep {epoch:3d} | train {run['total']:.4f} "
            f"(bce {run['bce']:.3f} dice {run['dice']:.3f} psp {run['psp']:.4f}) "
            f"| corr soft {stats['soft_frame_corr']:.4f} samp {stats['sample_frame_corr']:.4f} "
            f"| vR soft {stats['soft_van_rossum']:.3f} samp {stats['sample_van_rossum']:.3f} "
            f"| F1 samp {stats['sample_f1']:.4f} hard {stats['hard_f1']:.4f} "
            f"| bits {stats['code_bits_used']}/{cfg.n_bits} | {stats['secs']:.0f}s",
            flush=True,
        )

        if stats[SELECT] > best_score:
            best_score = stats[SELECT]
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                        "threshold": th, "stats": stats},
                       out_dir / "best.pt")
        # also keep the final epoch: the selection metric is one opinion among
        # several here and the last model is the one the schedule converged to
        torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                    "threshold": th, "stats": stats}, out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return model, history


@torch.no_grad()
def _fit_threshold(model, loader, device, n_batches: int = 8):
    model.eval()
    probs, targets = [], []
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device, non_blocking=True)
        logits, _ = model(x)
        probs.append(torch.sigmoid(logits).cpu())
        targets.append(x.cpu())
    return pick_threshold(torch.cat(probs), torch.cat(targets))


def _fmt(stats: dict) -> str:
    return " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in stats.items())
