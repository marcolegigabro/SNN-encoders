"""Encoder capacity grid: how much do width and training length help an encoder
represent a fine event-time code on its own?

Each encoder is trained alone (no quantizer, no decoder) to regress the hand-built
8x64 event-time symbols (k-th event time on a 63-cell grid, level 63 = no event),
then its rounded predictions are decoded by hand (perfect symbols give D 0.0378).
The GRU is always parameter-matched to the SNN of the same hidden width.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache" / "torchinductor"))
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from src.distortion import counting_l1
from src.encoders import SNNEncoder, matched_ann, n_params
from src.sources import Events, eval_set, homogeneous_poisson

dev, rate, L, A, M = "cuda", 5.0, 8, 64, 200
GRID = [("ann", 64, 6000), ("ann", 128, 6000), ("ann", 256, 6000), ("ann", 64, 20000),
        ("snn", 128, 6000), ("snn", 256, 6000), ("snn", 64, 20000)]


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
print(f"perfect symbols: D {counting_l1(test, handbuilt(target_test)).mean().item():.4f}", flush=True)
print("reference (6000 steps, width 64): ann exact 0.538 D 0.1919 | snn exact 0.366 D 0.2913", flush=True)
for name, hidden, steps in GRID:
    torch.manual_seed(0)
    snn = SNNEncoder(hidden, L)
    enc = (snn if name == "snn" else matched_ann(snn, L)).to(dev)
    fwd = torch.compile(enc) if name == "snn" else enc
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    enc.train()
    for step in range(1, steps + 1):
        ev = homogeneous_poisson(512, rate, generator=gen, device=dev)
        v = (A - 1) * torch.sigmoid(fwd(ev.binned(M)))
        loss = F.smooth_l1_loss(v, code(ev).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        opt.step()
        sched.step()
    enc.eval()
    with torch.no_grad():
        v = torch.cat([(A - 1) * torch.sigmoid(enc(test[i:i + 1000].binned(M)))
                       for i in range(0, len(test), 1000)])
    pred = v.round().long().clamp(0, A - 1)
    exact = (pred == target_test).float().mean().item()
    within1 = ((pred - target_test).abs() <= 1).float().mean().item()
    D = counting_l1(test, handbuilt(pred)).mean().item()
    print(f"{name} width {hidden:<4d} ({n_params(enc):>6d} params) steps {steps:<6d} | final train huber "
          f"{loss.item():.3f} | exact {exact:.3f} within+-1 {within1:.3f} | D {D:.4f} | "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
