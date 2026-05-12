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