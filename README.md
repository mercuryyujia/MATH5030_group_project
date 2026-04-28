# Risk Parity Portfolio Optimization with Weight Constraints

Python package for long-only risk parity portfolio construction with two solvers:

- `CCDSolver` for the unconstrained baseline problem
- `SCASolver` for risk parity with per-asset upper bounds

The project is designed as a compact, testable implementation for numerical finance coursework and lightweight experimentation.

## Features

- Unconstrained risk parity via cyclical coordinate descent (CCD)
- Constrained risk parity via successive convex approximation (SCA)
- Utility functions for absolute and relative risk contributions
- NumPy-only core dependency
- Pytest test suite and GitHub Actions CI

## Installation

Install from PyPI:

```bash
pip install risk-parity-constrained
```

Or install locally from source:

```bash
git clone https://github.com/mercuryyujia/MATH5030_group_project.git
cd MATH5030_group_project
pip install -e .
```

Install test dependencies:

```bash
pip install -e .[test]
```

Install notebook/demo dependencies:

```bash
pip install -e .[demo]
```

## Quick Start

```python
import numpy as np
from riskparity import (
    CCDSolver,
    SCASolver,
    relative_risk_contributions,
    risk_contribution_gap,
)

Sigma = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.02],
    [0.00, 0.02, 0.16],
])

w_ccd = CCDSolver(Sigma).solve()
print("CCD weights:", w_ccd)
print("CCD relative RC:", relative_risk_contributions(Sigma, w_ccd))
print("CCD gap:", risk_contribution_gap(Sigma, w_ccd))

w_sca = SCASolver(Sigma, w_max=0.5).solve()
print("SCA weights:", w_sca)
print("SCA relative RC:", relative_risk_contributions(Sigma, w_sca))
print("SCA gap:", risk_contribution_gap(Sigma, w_sca))
```

## What the Solvers Do

Risk parity aims to allocate weights so that each asset contributes equally to total portfolio risk. In practice, portfolio construction often requires additional constraints such as:

- long-only weights
- weights summing to one
- per-asset caps like `w_i <= 10%`

This package provides:

- `CCDSolver`: an unconstrained long-only baseline solver
- `SCASolver`: a constrained solver with box constraints of the form `0 <= w_i <= w_max`

## API Reference

### Solvers

#### `CCDSolver(Sigma, tol=1e-8, max_iter=1000)`

Computes an unconstrained long-only risk parity solution and normalizes weights to sum to 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `Sigma` | `np.ndarray` | Symmetric covariance matrix of shape `(n, n)` |
| `tol` | `float` | Convergence tolerance |
| `max_iter` | `int` | Maximum CCD iterations |

Key attributes after `.solve()`:

- `n_iter_`
- `converged_`
- `objective_`
- `risk_contribution_gap_`

#### `SCASolver(Sigma, w_max=1.0, tol=1e-6, max_iter=200)`

Computes a constrained risk parity portfolio under:

- `sum(w) = 1`
- `0 <= w_i <= w_max`

| Parameter | Type | Description |
|-----------|------|-------------|
| `Sigma` | `np.ndarray` | Symmetric covariance matrix of shape `(n, n)` |
| `w_max` | `float` | Per-asset upper bound in `(0, 1]` |
| `tol` | `float` | Convergence tolerance |
| `max_iter` | `int` | Maximum outer SCA iterations |

Key attributes after `.solve()`:

- `n_iter_`
- `converged_`
- `objective_`
- `risk_contribution_gap_`

### Utility Functions

#### `risk_contributions(Sigma, w)`

Returns per-asset risk contributions:

```python
w * (Sigma @ w)
```

#### `relative_risk_contributions(Sigma, w)`

Normalizes risk contributions to sum to 1.

#### `risk_contribution_gap(Sigma, w)`

Returns the maximum absolute deviation from equal risk contributions.

## Testing

The test suite is organized around correctness, validation, and robustness:

- **Correct implementation and validation**: covariance input validation, CCD solver checks, diagonal covariance closed-form cases, two-asset sanity checks, long-only weights that sum to one, and equal risk contribution checks.
- **Robustness testing**: constrained solver feasibility, infeasible `w_max` errors, randomized covariance matrices, stability across different portfolio sizes, and boundary parameter cases.

Run the test suite with:

```bash
pytest
```

CI currently tests Python `3.10`, `3.11`, and `3.12`.

## Demo Notebook

The repository includes a demonstration notebook in [`notebooks/demo.ipynb`](./notebooks/demo.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mercuryyujia/MATH5030_group_project/blob/main/notebooks/demo.ipynb)

## Project Structure

```text
.
├── .github/workflows/
│   ├── ci.yml
│   └── publish.yml
├── notebooks/
│   ├── demo.html
│   ├── demo.ipynb
│   └── demo.pdf
├── riskparity/
│   ├── __init__.py
│   └── _core.py
├── tests/
│   └── test_riskparity.py
├── LICENSE
├── pyproject.toml
└── README.md
```

## References

- Feng, Y., and Palomar, D. P. (2015). SCRIP: Successive Convex Optimization Methods for Risk Parity Portfolio Design. *IEEE Transactions on Signal Processing*. https://ieeexplore.ieee.org/document/7145485
- Choi, J., and Chen, R. (2022). Improved iterative methods for solving risk parity portfolio. *Journal of Derivatives and Quantitative Studies*, 30(2), 114-124. https://doi.org/10.1108/JDQS-12-2021-0031
- `riskParityPortfolio`: R package for risk parity portfolio design, used for one fixed reference validation example. https://github.com/dppalomar/riskParityPortfolio

## License

MIT License. See [`LICENSE`](./LICENSE).
