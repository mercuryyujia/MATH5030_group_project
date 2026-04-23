"""Core solvers for risk parity portfolio optimization.

This module provides two solvers for the risk parity problem:

* :class:`CCDSolver` — a fast Cyclical Coordinate Descent baseline for the
  unconstrained long-only problem (Choi & Chen, 2022).
* :class:`SCASolver` — a Successive Convex Approximation solver that supports
  per-asset upper bounds ``w_i <= w_max`` (Feng & Palomar, 2025).

Both solvers return weights ``w`` with ``w_i >= 0`` and ``sum(w) == 1``.
"""

from __future__ import annotations

import numpy as np


def _validate_covariance(Sigma: np.ndarray) -> np.ndarray:
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square 2-D array.")
    if not np.allclose(Sigma, Sigma.T, atol=1e-10):
        raise ValueError("Sigma must be symmetric.")
    return Sigma


class CCDSolver:
    """Cyclical Coordinate Descent solver for the unconstrained risk parity problem.

    Solves the convex surrogate

        min_{w > 0}  0.5 * w^T Sigma w  -  (1/n) * sum_i log(w_i)

    and rescales so that ``sum(w) = 1``. At the optimum every asset contributes
    equally to total portfolio risk.
    """

    def __init__(self, Sigma: np.ndarray, tol: float = 1e-8, max_iter: int = 1000):
        self.Sigma = _validate_covariance(Sigma)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.n_iter_: int | None = None
        self.converged_: bool = False

    def solve(self) -> np.ndarray:
        Sigma = self.Sigma
        n = Sigma.shape[0]
        c = 1.0 / n

        w = np.full(n, 1.0 / n)
        Sw = Sigma @ w

        for it in range(1, self.max_iter + 1):
            w_old = w.copy()
            for i in range(n):
                sigma_ii = Sigma[i, i]
                b_i = Sw[i] - sigma_ii * w[i]
                disc = b_i * b_i + 4.0 * sigma_ii * c
                w_new = (-b_i + np.sqrt(disc)) / (2.0 * sigma_ii)
                delta = w_new - w[i]
                if delta != 0.0:
                    Sw += delta * Sigma[:, i]
                    w[i] = w_new

            if np.linalg.norm(w - w_old, ord=np.inf) < self.tol:
                self.n_iter_ = it
                self.converged_ = True
                break
        else:
            self.n_iter_ = self.max_iter
            self.converged_ = False

        return w / w.sum()
