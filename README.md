# Time-Scale Coupling Between States and Parameters in Recurrent Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2508.12121-b31b1b.svg)](https://arxiv.org/abs/2508.12121)

Code accompanying the paper:

**Lorenzo Livi**
*Time-Scale Coupling Between States and Parameters in Recurrent Neural Networks*
Paper: https://arxiv.org/abs/2508.12121

---

## Overview

This repository contains the code used to generate the experiments reported in the paper.
The work shows that gating mechanisms in RNNs induce lag-dependent and
direction-dependent **effective learning rates**, even when training uses a fixed,
global step size. This behavior arises from a coupling between state-space
time-scales (parametrized by the gates) and parameter-space dynamics during
gradient descent. By deriving exact Jacobians and applying a first-order
expansion, we make explicit how constant, scalar, and multi-dimensional gates
reshape gradient propagation, modulate effective step sizes, and introduce
anisotropy in parameter updates.

---

## Repository structure

```
.
├── s1_effective_lr.py       # Single-seed: effective LR profiles (Section VII-A)
├── s2_joint_anisotropy.py   # Single-seed: Jacobian & gradient-covariance anisotropy (Section VII-B)
├── appendixA_check.py       # Single-seed: first-order Frechet expansion validation (Appendix B)
├── s1_robust.py             # Multi-seed wrapper for S1 (aggregation + plotting)
├── s2_robust.py             # Multi-seed wrapper for S2 (aggregation + plotting)
├── appendixA_robust.py      # Multi-seed wrapper for Appendix A (aggregation + plotting)
├── run_all_robust.py        # Master runner: executes all three experiments
├── requirements.txt
└── README.md
```

Each experiment has two scripts:

- **Core script** (`s1_effective_lr.py`, `s2_joint_anisotropy.py`, `appendixA_check.py`):
  implements the single-seed training, probing, and analysis logic.
- **Robust wrapper** (`s1_robust.py`, `s2_robust.py`, `appendixA_robust.py`):
  runs the core script across multiple random seeds and produces aggregated
  plots with uncertainty bands (mean +/- std or median + IQR).

---

## Requirements

```
pip install -r requirements.txt
```

The code requires Python 3.9+ and depends on PyTorch, NumPy, and Matplotlib.

---

## Reproducing the paper results

### Full replication (20 seeds, as in the paper)

```bash
python run_all_robust.py --num-seeds 20 --device cpu
```

This runs all three experiments sequentially. On a modern CPU, expect roughly
4-6 hours for the full 20-seed run. To use a GPU:

```bash
python run_all_robust.py --num-seeds 20 --device cuda
```

### Quick verification (fewer seeds)

```bash
python run_all_robust.py --num-seeds 3 --s1-steps 200 --s2-iters 400 --device cpu
```

### Running individual experiments

```bash
# S1: Effective learning rate profiles
python s1_robust.py --seeds 1 2 3 4 5 --tasks adding --models leaky scalar multigate --device cpu

# S2: Joint anisotropy (all 5 tasks)
python s2_robust.py --seeds 1 2 3 4 5 --tasks adding ar2 movingavg delaysum narma10 --device cpu

# Appendix A: First-order expansion validation
python appendixA_robust.py --seeds 0 1 2 3 4 --device cpu
```

### Outputs

Each experiment produces:
- **PNG figures** suitable for inclusion in the paper
- **JSON summaries** with per-seed and aggregated statistics

Outputs are saved in directories specified by the `--out` flag (defaults:
`s1_robust/`, `s2_robust/`, `appendixA_robust/`).

---

## Experiment details

### S1 -- Effective learning rate profiles

Trains leaky-integrator, scalar-gated, and multi-gated RNNs on the adding task
with plain SGD. At selected checkpoints, measures the lag-conditioned sensitivity
$S_{t,k} = \| \prod_{j=k+1}^{t} J_j \|_2$ and compares the empirical profile
against zeroth-order gate-product predictors. Reports the fitted log-log slope
$s(\ell)$ across training.

### S2 -- Joint anisotropy analysis

Compares propagation anisotropy (from Jacobian products) with update anisotropy
(from gradient covariance) across five tasks: adding, AR(2), moving-average,
delay-sum, and NARMA-10. Three models are trained: plain RNN + Adam,
scalar-gated RNN + SGD, and multi-gated RNN + SGD.

### Appendix A -- First-order expansion validation

Verifies that the truncation error of the first-order Frechet derivative of
gated Jacobian products scales as $O(\varepsilon^2)$, for both scalar-gated and
multi-gated architectures.

---

## Citation

```bibtex
@article{livi2025timescale,
  title={Time-Scale Coupling Between States and Parameters in Recurrent Neural Networks},
  author={Livi, Lorenzo},
  journal={arXiv preprint arXiv:2508.12121},
  year={2025}
}
```

## License

This code is released for academic and research use. See the paper for details.
