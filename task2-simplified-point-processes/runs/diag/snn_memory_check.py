"""SNN temporal-memory check: can longer hidden time constants or a scaled (instead of
detached) recurrent gradient let the SNN encoder learn precise event timing?

Same encoder-alone test as encoder_capacity_grid.py: SNN width 64, 20000 steps,
regress the hand-built 8x64 event-time symbols, decode the rounded predictions by hand
(perfect symbols: D 0.0378). References from that grid: current SNN D 0.1786,
parameter-matched GRU D 0.0523. Variants are patched in here; src/ is untouched.
"""
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache" / "torchinductor"))
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from src import snn as snn_mod
from src.distortion import counting_l1
from src.encoders import SNNEncoder
from src.sources import Events, eval_set, homogeneous_poisson

dev, rate, L, A, M, WIDTH, STEPS = "cuda", 5.0, 8, 64, 200, 64, 20000
# (label, hidden LIF beta at init, recurrent gradient scale; 0 = detached as in src/)
VARIANTS = [("beta0.97", 0.97, 0.0), ("beta0.99", 0.99, 0.0),
            ("gamma0.5", 0.9, 0.5), ("beta0.97+gamma0.5", 0.97, 0.5)]


def make_forward(gamma):
    def forward(self, x, state):
        drive = self.syn(x)
        if self.rec is not None and state.s is not None:
            s = state.s
            # forward value unchanged; gradient through the recurrence scaled by gamma
            s_rec = s.detach() if gamma == 0 else s * gamma + s.detach() * (1.0 - gamma)
            drive = drive + self.rec(s_rec)
        return self.lif(drive, state)
    return forward


def code(ev):
    t = ev.padded_to(L).times[:, :L]
    cells = A - 1
    return torch.where(t < ev.T, (t / ev.T * cells).floor().clamp(max=cells - 1).long(),
                       torch.full_like(t, cells, dtype=torch.long))


def handbuilt(sym, T=1.0):
    cells = A - 1
    t_hat = torch.where(sym < cells, (sym.float() + 0.5) * T / cells,
                        torch.full_like(sym, T, dtype=torch.float))
    return Events(t_hat, (sym < cells).sum(1), T)


test = eval_set(rate, 10_000).to(dev)
target_test = code(test)
print("references: current SNN (beta 0.9, detached) D 0.1786 | GRU D 0.0523 | perfect 0.0378", flush=True)
for label, beta, gamma in VARIANTS:
    snn_mod.SpikingLinear.forward = make_forward(gamma)
    torch.manual_seed(0)
    enc = SNNEncoder(WIDTH, L).to(dev)
    with torch.no_grad():
        for layer in (enc.l1, enc.l2):
            layer.lif.beta_logit.fill_(math.log(beta / (1 - beta)))
    fwd = torch.compile(enc)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.1)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0, skipped, loss = time.time(), 0, None
    enc.train()
    for step in range(1, STEPS + 1):
        ev = homogeneous_poisson(512, rate, generator=gen, device=dev)
        v = (A - 1) * torch.sigmoid(fwd(ev.binned(M)))
        loss = F.smooth_l1_loss(v, code(ev).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        if not torch.isfinite(norm):
            skipped += 1
            continue
        opt.step()
        sched.step()
        if step % 5000 == 0:
            print(f"  [{label}] step {step} train huber {loss.item():.3f} skipped {skipped} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    enc.eval()
    with torch.no_grad():
        v = torch.cat([(A - 1) * torch.sigmoid(enc(test[i:i + 1000].binned(M)))
                       for i in range(0, len(test), 1000)])
        betas = [torch.sigmoid(layer.lif.beta_logit).mean().item() for layer in (enc.l1, enc.l2)]
    pred = v.round().long().clamp(0, A - 1)
    exact = (pred == target_test).float().mean().item()
    within1 = ((pred - target_test).abs() <= 1).float().mean().item()
    D = counting_l1(test, handbuilt(pred)).mean().item()
    print(f"{label:18s} | final huber {loss.item():.3f} skipped {skipped} | learned mean beta l1 {betas[0]:.3f} "
          f"l2 {betas[1]:.3f} | exact {exact:.3f} within+-1 {within1:.3f} | D {D:.4f} | "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
