"""Training and evaluation of one (bottleneck, sigma, K, distortion) point.

One model per point of the rate-distortion curve, and one model per distortion:
a curve is the locus of *optimal* systems at each rate, so a single model tuned
for one distortion and then scored on the other two would produce three curves
that are all wrong except one. It costs three times as much and there is no way
around it.

The source is a generator, so there is no epoch structure and no train/test
split: every batch is fresh, the evaluation batch is drawn from a separately
seeded stream, and overfitting is not a possible confound.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import distortions as dist
from . import theory as th
from .model import JSCCSystem, SystemConfig
from .source import PoissonSource


@dataclass
class Config:
    # source
    n_neurons: int = 16
    n_bins: int = 12
    rate: float = 0.15
    source: str = "iid"            # "iid" | "latent"
    # system
    n_code: int = 8
    hidden: int = 256
    bottleneck: str = "ttfs"       # "ttfs" | "multispike"
    sigma: float = 1.0
    beta: float = 0.9
    # objective
    distortion: str = "hamming"    # hamming | count_mse | van_rossum
    tau: float = 3.0
    # optimisation
    steps: int = 3000
    batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "cpu"
    eval_batch: int = 4096
    eval_reps: int = 4
    log_every: int = 250

    def system(self) -> SystemConfig:
        return SystemConfig(
            n_neurons=self.n_neurons, n_bins=self.n_bins, n_code=self.n_code,
            hidden=self.hidden, bottleneck=self.bottleneck, sigma=self.sigma,
            beta=self.beta, source_rate=self.rate,
        )


def _thresholds():
    return torch.linspace(0.02, 0.98, 49)


@torch.no_grad()
def fit_threshold(model, source, d: dist.Distortion, cfg: Config,
                  n_batch: int = 4) -> float:
    """Pick the decision threshold that minimises this distortion on a genuinely
    binary reconstruction.

    Fitted for every distortion, not only the ones scored on spikes: Hamming
    needs it as its primary measurement, and the other two need it for the
    secondary, binary-reproduction number reported next to the
    `reproduction="binary"` bound. For Hamming a calibrated model would put it
    at 1/2 and this only corrects miscalibration; under a squared-error
    distortion the optimum is genuinely elsewhere, because the cost of a missing
    spike and of a spurious one are not equal.
    """
    # eval mode, like `evaluate`: a threshold fitted while the batch norms are
    # still using per-batch statistics is fitted to a different network.
    model.eval()
    grid = _thresholds().to(cfg.device)
    totals = torch.zeros_like(grid)
    for _ in range(n_batch):
        x = source.sample(cfg.batch)
        probs = torch.sigmoid(model(x)[0])
        for i, th_ in enumerate(grid):
            totals[i] += d.measure((probs > th_).float(), x)
    model.train()
    return float(grid[int(totals.argmin())])


@torch.no_grad()
def evaluate(model, source, cfg: Config, threshold: float) -> dict:
    """All three distortions on a fresh stream, plus code statistics.

    Every distortion is reported for every model even though only one of them
    was trained for: the off-diagonal numbers are what show that the three
    distortions really do disagree, which is the reason the subject asks for
    three of them.
    """
    model.eval()
    ds = {n: dist.build(n, cfg.tau, cfg.n_bins) for n in dist.NAMES}
    acc = {n: 0.0 for n in dist.NAMES}
    acc.update({f"{n}_soft": 0.0 for n in dist.NAMES})
    acc.update({f"{n}_hard": 0.0 for n in dist.NAMES})
    code_stat = {"code_mean": 0.0, "code_std": 0.0, "code_rate": 0.0}
    reps = cfg.eval_reps
    for _ in range(reps):
        x = source.sample(cfg.eval_batch)
        logits, code, _ = model(x)
        probs = torch.sigmoid(logits)
        hard = (probs > threshold).float()
        for n, d in ds.items():
            # `n` is the primary pairing (binary for Hamming, real-valued for the
            # two squared-error distortions); the other two columns let a reader
            # see what the choice of reproduction alphabet is worth.
            acc[n] += d.measure(hard if d.on_hard else probs, x) / reps
            acc[f"{n}_soft"] += d.measure(probs, x) / reps
            acc[f"{n}_hard"] += d.measure(hard, x) / reps
        code_stat["code_mean"] += float(code.mean()) / reps
        code_stat["code_std"] += float(code.std()) / reps
        code_stat["code_rate"] += float((code > 0.5).float().mean()) / reps
    model.train()
    out = {**acc, **code_stat, "threshold": threshold,
           "target_rate": float(x.mean())}
    return out


def train(cfg: Config, out_dir: Path, cache_path: Path | None = None,
          verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    train_src = PoissonSource(cfg.n_neurons, cfg.n_bins, cfg.rate, cfg.source,
                              device=cfg.device, seed=cfg.seed)
    eval_src = PoissonSource(cfg.n_neurons, cfg.n_bins, cfg.rate, cfg.source,
                             device=cfg.device, seed=10_000 + cfg.seed)
    eval_src.gain = train_src.gain  # the tuning is the source's, not the split's

    model = JSCCSystem(cfg.system()).to(cfg.device)
    d = dist.build(cfg.distortion, cfg.tau, cfg.n_bins)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1)

    history = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        x = train_src.sample(cfg.batch)
        logits, _, _ = model(x)
        loss = d.loss(logits, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step == cfg.steps:
            history.append({"step": step, "loss": float(loss.detach())})
            if verbose:
                print(f"  step {step:5d}/{cfg.steps}  loss {float(loss.detach()):.5f}"
                      f"  {time.time() - t0:.0f}s", flush=True)

    threshold = fit_threshold(model, eval_src, d, cfg)
    result = evaluate(model, eval_src, cfg, threshold)

    # rate: channel uses per block priced at the channel's own capacity, then
    # normalised per source bin so it can share an axis with R(D).
    capacity = th.channel_capacity(cfg.bottleneck, cfg.n_bins, cfg.sigma,
                                   cache_path)
    block_bits = cfg.n_bins * cfg.n_neurons
    result.update({
        "capacity_bits_per_use": capacity,
        "rate_bits_per_block": cfg.n_code * capacity,
        "rate_bits_per_bin": cfg.n_code * capacity / block_bits,
        "distortion_trained": cfg.distortion,
        "primary": result[cfg.distortion],
        "train_seconds": time.time() - t0,
    })

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    torch.save({"cfg": asdict(cfg), "state": model.state_dict()},
               out_dir / "model.pt")
    return result
