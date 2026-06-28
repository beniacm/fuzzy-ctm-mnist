# Fuzzy Convolutional Tsetlin Machine — MNIST (99.2%)

A compact, dependency-light **Fuzzy Convolutional Tsetlin Machine (FCTM)** in pure Julia that
trains on MNIST to **99.2% test accuracy** — competitive with the Convolutional Tsetlin Machine
literature (~99.4%) and well above a plain Tsetlin Machine (~98.2%).

A Tsetlin Machine learns **interpretable conjunctive clauses** over boolean features using
Tsetlin automata and integer reinforcement — no gradients, no floating point, just bitwise
ops and counters. This implementation adds two ingredients that lift it to ~99%:

- **Convolution (weight sharing):** each clause is evaluated over every sliding 10×10 patch and
  votes from its best-matching patch — a translation-tolerant, memory-cheap inductive bias.
- **Fuzzy voting:** a clause casts a *graded* vote `max(0, LF − failed_literals)` instead of a
  hard all-or-nothing fire — the [Fuzzy-Pattern Tsetlin Machine](https://arxiv.org/abs/2508.08350)
  mechanism (Hnilov 2025), where a clause's surviving literals still contribute a proportionally
  reduced score rather than being disqualified by a single failed literal.

Both act as **regularizers**, which is why this model generalizes well (see the train/test gap below).

## Results

| model | MNIST **test** acc | **train** acc | gap |
|---|---|---|---|
| plain Tsetlin Machine | ~98.2% | ~99.9% | 1.7 pts |
| **this (fuzzy conv TM)** | **99.2%** | **99.5%** | **0.33 pts** |

The small train/test gap (5× tighter than a plain TM) shows the convolution + fuzziness are
regularizing rather than memorizing.

## Performance (CPU vs GPU, stride 1 vs 2)

Measured on one machine — a 16-thread CPU and an AMD Radeon 8060S iGPU (ROCm) — at the default
`CPC=320, lf=16`, full 60k-train / 10k-test, steady-state epoch:

| backend | stride | patches/img | test acc | train throughput | train / epoch | full 40-ep run |
|---|---|---|---|---|---|---|
| CPU (16 threads) | 1 | 361 | **0.9921** | 3.3k samp/s | 18.3 s | ~14 min |
| **GPU (ROCm)**   | 1 | 361 | **0.9921** | **11.7k samp/s** | **5.1 s** | **~4 min** |
| CPU (16 threads) | 2 | 100 | 0.9901 | 6.2k samp/s | 9.7 s | ~7 min |
| **GPU (ROCm)**   | 2 | 100 | 0.9901 | **14.6k samp/s** | **4.1 s** | **~3 min** |

- **GPU vs CPU** — ~3.5× faster training at stride 1 (one GPU work-item per clause vs CPU threads
  over clauses). Both backends are **bit-exact**: identical per-epoch accuracy to 4 dp at either
  stride, so the GPU is purely an accelerator, not a different model.
- **stride 2 vs 1** — a coarser patch grid (10×10 = 100 origins instead of 19×19 = 361) trains
  ~1.9× faster on CPU for a **0.20 pt** accuracy cost (0.9901 vs 0.9921). Evaluation scales even
  better with stride (the per-clause include-build is a fixed cost that only training pays). Use
  `--stride 2` when you want most of the accuracy at roughly half the training time.

```bash
julia --project -t auto train.jl --backend gpu                 # 0.9921 in ~4 min
julia --project -t auto train.jl --backend gpu --stride 2      # 0.9901 in ~3 min
julia --project -t auto train.jl --stride 2                    # CPU, 0.9901 in ~7 min
```

## Quick start

```bash
# install deps (CPU-only users may drop AMDGPU from Project.toml)
julia --project -e 'using Pkg; Pkg.instantiate()'

# CPU — portable, reaches ~99.2% (~22 s/epoch, full 40-epoch run ~15 min, multi-threaded)
julia --project -t auto train.jl

# quicker preview (~99.0%, a few minutes)
julia --project -t auto train.jl --cpc 160 --epochs 15

# AMD ROCm acceleration (~3× faster, ~7 s/epoch; needs an AMD GPU + AMDGPU.jl)
julia --project -t auto train.jl --backend gpu
```

MNIST is downloaded automatically on first run (or point `--data DIR` at local IDX files).

## How it works (one sample)

1. **Booleanize** the 28×28 image with adaptive-Gaussian thresholding (local mean − C). This
   matters: a single global threshold caps ~1 pt lower.
2. **Patch literals:** for each of the 19×19 sliding 10×10 patches, build a 320-bit literal
   vector `{features, ¬features}` (100 pixels + 36 x/y position-thermometer bits, padded to 160,
   then negated), bit-packed into 5 × `UInt64`.
3. **Evaluate** every clause: its vote is `max(0, LF − min over patches of failed literals)`
   (the best-matching patch). Class score = signed sum of its clauses' votes; predict argmax.
4. **Feedback** (Tsetlin Type I / Ib / II) reinforces the true class `y` and penalizes one random
   rival class `q ≠ y`, clamped by a vote target `T`.

Training is online (sample *i* updates the automata read by *i+1*) but every sample's work is
**parallel over the `10 × CPC` clauses** — the CPU backend threads over clauses, the GPU backend
runs one work-item per clause.

## Configuration

| flag | default | meaning |
|---|---|---|
| `--cpc` | 320 | clauses per class (half +, half − polarity) |
| `--lf` | 16 | fuzziness (graded-vote width) |
| `--stride` | 1 | patch step in pixels; `2` scans a coarser grid (~3.6× fewer patches, faster, ~0.2pt cost) |
| `--epochs` | 40 | training passes |
| `--backend` | cpu | `cpu` or `gpu` (AMD ROCm) |
| `--no-adaptive` | off | use a global threshold instead of adaptive-Gaussian |
| `--train-n` / `--eval-n` | all | subset sizes |

Other knobs (`L` include cap, `s` Type-Ib draws, `T` vote target) are in `FCTM(...)`.

## Files

```
src/FuzzyCTM.jl      core algorithm (CPU, multi-threaded, pure Julia)
src/FuzzyCTMGPU.jl   optional AMD ROCm / AMDGPU.jl backend (bit-exact, ~3–10× faster)
src/MNISTData.jl     MNIST download + adaptive-Gaussian booleanization (pure Julia)
train.jl             full pipeline: load → binarize → train → test
```

The library API is small:

```julia
using .FuzzyCTM, .MNISTData
Xtr, Ytr, Xte, Yte = load_mnist("data")          # 784×N Bool, labels 0:9
m = FCTM(; cpc=320, lf=16)
fit!(m, Xtr, Ytr; epochs=40, evalX=Xte, evalY=Yte)
accuracy(m, Xte, Yte)
```

## Provenance & references

Extracted from an FPGA Tsetlin-Machine accelerator project (the same algorithm runs on an
Arria 10 at 9 W). The CPU and GPU backends are bit-exact to each other.

- O.-C. Granmo, *The Tsetlin Machine* (2018), arXiv:1804.01508
- O.-C. Granmo et al., *The Convolutional Tsetlin Machine* (2019), arXiv:1905.09688
- A. Hnilov, *Fuzzy-Pattern Tsetlin Machine* (2025), arXiv:2508.08350 — the graded-vote
  ("fuzzy") clause evaluation used here ([Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl))

## License

MIT — see [LICENSE](LICENSE).
