# SNN-encoders

Filière Métiers de la Recherche, sujet 12: **Spike-Train Compression for
Distributed Neuromorphic Intelligence**.
Supervisor: Michèle Wigger. Team 12: Ouail El Khattabi, Mouad Leachouri,
Fergal Miskdjian, Marc-Antoine Wilk.

## Layout

One folder per task from the subject sheet. **The tasks are independent: keep
your work inside your task folder** so we do not collide.

```
task1-compression-with-channel/   end-to-end compression through a channel
task2-compression-to-m-bits/      end-to-end compression to a fixed m-bit code
data/                             datasets, gitignored, see below
```

There is deliberately no `shared/` package yet. The two tasks do share
primitives (LIF cells, surrogate gradients, spike sources, distortion
metrics), so some duplication is expected for now; factoring them out is worth
doing once both tasks have stabilised, not before.

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install tonic h5py torch
```

Python is pinned to 3.12 rather than a newer interpreter, because the SNN
stack (numba, torch) does not ship wheels for 3.14 yet.

## Data

```bash
python task2-compression-to-m-bits/scripts/download_datasets.py
```

Downloads N-MNIST (~1 GB) and SHD (~400 MB) into `data/`, which is gitignored.
Override the location with `--data-root` or `$SNN_DATA_ROOT`. Add `-d all` for
DVS128 Gesture (~3 GB).

Note the script keeps the extracted data but truncates the source archives to
0 bytes to save space. Do not delete those empty `.zip` files: tonic checks
that the archive exists and will re-download otherwise.
