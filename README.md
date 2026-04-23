# Risk Parity Portfolio Optimization with Weight Constraints

Implements constrained risk parity portfolio optimization in Python, with two solvers: a fast CCD baseline and a constrained SCA solver supporting per-asset weight bounds.

---

## What problem it solves

Modern portfolios require each asset to contribute equally to total risk — a property called *risk parity*. In practice, weights must also satisfy bounds like w_i ≤ 10%. This package implements both an unconstrained solver (CCD, Choi & Chen 2022) and a constrained solver (SCA, Feng & Palomar 2025), and benchmarks their computational efficiency across portfolio sizes.

---

## Installation

```bash
pip install risk-parity-constrained
```

Or clone and install locally:

```bash
git clone https://github.com/mercuryyujia/MATH5030_group_project.git
cd MATH5030_group_project
pip install -e .
```

---

## Quick start

```python
import numpy as np
from riskparity import CCDSolver, SCASolver

Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])

# Unconstrained
w = CCDSolver(Sigma).solve()

# Constrained (max 60% per asset)
w_constrained = SCASolver(Sigma, w_max=0.6).solve()
```

---

## API reference

### `CCDSolver(Sigma, tol=1e-8, max_iter=1000)`
| Parameter | Type | Description |
|-----------|------|-------------|
| `Sigma` | `np.ndarray` (n×n) | Asset covariance matrix |
| `tol` | `float` | Convergence tolerance |
| `max_iter` | `int` | Maximum iterations |

**`.solve()`** → `np.ndarray` (n,): portfolio weights summing to 1

---

### `SCASolver(Sigma, w_max=1.0, tol=1e-6, max_iter=200)`
| Parameter | Type | Description |
|-----------|------|-------------|
| `Sigma` | `np.ndarray` (n×n) | Asset covariance matrix |
| `w_max` | `float` | Per-asset weight upper bound |
| `tol` | `float` | Convergence tolerance |
| `max_iter` | `int` | Maximum outer SCA iterations |

**`.solve()`** → `np.ndarray` (n,): portfolio weights summing to 1

---

## License

MIT

---

## Demo notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USER/YOUR_REPO/blob/main/notebooks/demo.ipynb)