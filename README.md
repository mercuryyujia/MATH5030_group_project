# Risk Parity Portfolio Optimization with Weight Constraints

Python package for long-only risk parity portfolio construction with two solvers:

- `CCDSolver` for the unconstrained baseline problem
- `SCASolver` for risk parity with per-asset upper bounds

The project is designed as a compact, testable implementation for numerical finance coursework and lightweight experimentation.

## Features

- Unconstrained risk parity via PyFENG's cyclical coordinate descent (CCD)
- Constrained risk parity via successive convex approximation (SCA)
- Utility functions for absolute and relative risk contributions
- NumPy and PyFENG core dependencies
- Pytest test suite and GitHub Actions CI

## How We Prompted AI

During the project, we used AI as a supporting tool in our discussion, coding,
debugging, and documentation process. We first chose the project direction and
worked through the main idea of risk parity portfolio optimization ourselves:
constructing portfolio weights so that assets make balanced contributions to
total portfolio risk. After that, we used AI to help us organize the project
more clearly and check whether our explanations and code structure were easy
to follow.

Our prompts were usually specific questions that came up while developing the
package. For example, we asked about input validation for covariance matrices,
ways to compare risk contributions numerically, useful tests for portfolio
weights and solver outputs, and clearer ways to explain the difference between
unconstrained and constrained risk parity. We reviewed the suggestions against
our own code, the course material, and the numerical behavior we observed
before deciding what to keep.

AI was also useful when revising the README and demo notebook, especially for
making the project description more organized and identifying places where our
writing was vague. The final choices about the project scope, implementation,
tests, and interpretation were made by our group after checking the code and
results ourselves.

## What We Added Beyond the Reference Papers

The references provide the numerical foundations (CCD and SCA-style ideas), but this project contributes an end-to-end, reproducible implementation layer for constrained portfolio construction in practice.

- **Constrained extension on top of the baseline CCD workflow**: we keep PyFENG's CCD as the unconstrained baseline and add `SCASolver` for feasible long-only portfolios with per-asset caps (`sum(w)=1`, `0 <= w_i <= w_max`), which addresses concentration control not handled by unconstrained risk parity outputs.
- **A complete feasibility mechanism**: we implement simplex-box projection (`_project_simplex_box`) and explicit infeasibility checks (`w_max < 1/n`) so constraints are enforced numerically, not only stated theoretically.
- **Production-style solver diagnostics**: both solvers expose convergence and quality indicators (`n_iter_`, `converged_`, `objective_`, `risk_contribution_gap_`) for transparent evaluation and model debugging.
- **Robust input and output validation**: covariance, tolerance, and iteration guards are built in; returned weights are checked for finiteness, budget feasibility, non-negativity, and cap compliance.
- **Substantial verification suite**: 102 pytest cases cover analytical sanity checks (identity/diagonal/two-asset), fixed-matrix regression, constrained feasibility, randomized SPD robustness, and PyFENG edge-case behavior.
- **Reproducible research-to-package pipeline**: we provide a clean Python API, notebook demo, CI across Python 3.10-3.12, and packaging/release workflow (`pyproject.toml` + PyPI publishing), turning paper ideas into a reusable artifact.

## How Much Existing Code We Reused

We reused existing numerical tools where they were already reliable, and focused our own work on the constrained solver layer, validation, diagnostics, tests, notebook, and packaging rather than re-implementing mature solvers from scratch.

- **High reuse for the unconstrained CCD baseline**: `CCDSolver` delegates the actual optimization to PyFENG's `RiskParity.weight()`, which implements the improved CCD method of Choi and Chen (2022). Our wrapper adds input validation, normalization to `sum(w)=1`, error handling, and diagnostics around PyFENG's returned weights.
- **NumPy for core linear algebra**: covariance checks, risk contributions (`w * (Sigma @ w)`), objective values, gradients, clipping, projections, and randomized SPD test matrices are built with NumPy primitives instead of custom matrix or loop code.
- **No copied implementation from the papers**: Feng and Palomar (2015) and Choi and Chen (2022) guide the mathematical design, but we do not vendor the authors' code or translate the R `riskParityPortfolio` package into Python.
- **Local implementation for constrained SCA behavior**: `SCASolver` borrows the successive-approximation idea from the SCRIP reference, but the projected-gradient style iteration, backtracking step control, simplex-box projection, feasibility checks, and diagnostics are implemented in this repository.
- **No direct SciPy optimizer in the solver path**: although a generic `scipy.optimize.minimize` approach would be possible, the current implementation avoids a black-box optimizer dependency so the constrained updates and feasibility enforcement remain transparent.
- **External code used mainly as trusted infrastructure**: PyFENG handles the established CCD baseline, while pytest, GitHub Actions, packaging metadata, and notebook tooling support reproducibility and verification rather than replacing our project-specific solver logic.

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

Computes an unconstrained long-only risk parity solution using PyFENG's `RiskParity.weight()` implementation and normalizes weights to sum to 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `Sigma` | `np.ndarray` | Symmetric covariance matrix of shape `(n, n)` |
| `tol` | `float` | Convergence tolerance |
| `max_iter` | `int` | Backward-compatible parameter; PyFENG controls CCD iterations |

Key attributes after `.solve()`:

- `n_iter_`
- `converged_`
- `objective_`
- `risk_contribution_gap_`

#### `SCASolver(sigma=None, cor=None, cov=None, w_max=1.0, ret=None, budget=None, longshort=1, tol=1e-6, max_iter=200)`

Computes a constrained risk parity portfolio under:

- `sum(w) = 1`
- `0 <= w_i <= w_max`

| Parameter | Type | Description |
|-----------|------|-------------|
| `sigma` | `np.ndarray` | Asset volatilities, used with `cor` when `cov` is not supplied |
| `cor` | `np.ndarray` or `float` | Correlation matrix or scalar constant correlation |
| `cov` | `np.ndarray` | Symmetric covariance matrix of shape `(n, n)`; preferred when supplied |
| `w_max` | `float` | Per-asset upper bound in `(0, 1]` |
| `ret` | `np.ndarray` | Expected returns, stored for API compatibility and not used in optimization |
| `budget` | `np.ndarray` | Placeholder for PyFENG compatibility; only equal budgets are currently supported |
| `longshort` | `np.ndarray` or `int` | Placeholder for PyFENG compatibility; only long-only portfolios are currently supported |
| `tol` | `float` | Convergence tolerance |
| `max_iter` | `int` | Maximum outer SCA iterations |

Use `.weight(tol=None)` for the PyFENG-style API. The original project API
`.solve()` remains available, and `SCASolver(Sigma, ...)` is still accepted as
a shorthand for `SCASolver(cov=Sigma, ...)`.

Key attributes after `.solve()`:

- `n_iter_`
- `converged_`
- `objective_`
- `risk_contribution_gap_`
- `_result`

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
- Choi, J. PyFENG: Python Financial Engineering. https://github.com/PyFE/PyFENG
- `riskParityPortfolio`: Design of Risk Parity Portfolios. https://github.com/dppalomar/riskParityPortfolio

## License

MIT License. See [`LICENSE`](./LICENSE).
