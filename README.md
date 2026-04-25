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
For tests:

```bash
pip install -e .[test] 
pytest
```
---

## Quick start

```python
import numpy as np
from riskparity import CCDSolver, SCASolver, relative_risk_contributions

Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])

w = CCDSolver(Sigma).solve()
print(w)
print(relative_risk_contributions(Sigma, w))

w_constrained = SCASolver(Sigma, w_max=0.6).solve()
print(w_constrained)
print(relative_risk_contributions(Sigma, w_constrained))
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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mercuryyujia/MATH5030_group_project/blob/main/notebooks/demo.ipynb)


## Project structure
```
.
├── LICENSE
├── notebooks
│   ├── demo.html
│   ├── demo.ipynb
│   └── demo.pdf
├── Numerical_Methods_Proposal.pdf
├── Project Prototype _ MATHGR5030 [2026S] - NUMERICAL METHODS IN FINANCE.pdf
├── Project Suggestion_ Risk Parity_ MATHGR5030 [2026S] - NUMERICAL METHODS IN FINANCE.pdf
├── pyproject.toml
├── README.md
├── risk_parity_constrained.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── riskparity
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   └── _core.cpython-313.pyc
│   └── _core.py
└── tests
    ├── __pycache__
    │   ├── test_riskparity.cpython-313-pytest-8.4.2.pyc
    │   └── test_riskparity.cpython-313-pytest-9.0.3.pyc
    └── test_riskparity.py

7 directories, 21 files
```