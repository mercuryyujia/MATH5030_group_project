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


class SCASolver:
    """Successive Convex Approximation solver with per-asset upper bounds.

    Handles the constrained risk parity problem

        min_w   sum_i ( w_i * (Sigma w)_i  -  r_bar )^2
        s.t.    sum_i w_i = 1,   0 <= w_i <= w_max

    by iteratively linearising the non-convex objective around the current
    iterate and solving the resulting convex quadratic subproblem.
    """

    def __init__(
        self,
        Sigma: np.ndarray,
        w_max: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 200,
    ):
        self.Sigma = _validate_covariance(Sigma)
        n = self.Sigma.shape[0]
        if w_max <= 0 or w_max > 1:
            raise ValueError("w_max must lie in (0, 1].")
        if w_max * n < 1.0:
            raise ValueError(
                f"w_max={w_max} is infeasible for n={n}: need w_max >= 1/n."
            )
        self.w_max = float(w_max)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.n_iter_: int | None = None
        self.converged_: bool = False

    def solve(self) -> np.ndarray:
        Sigma = self.Sigma
        n = Sigma.shape[0]

        w = CCDSolver(Sigma, tol=1e-8, max_iter=500).solve()
        w = np.clip(w, 0.0, self.w_max)
        w = self._project_simplex_box(w, self.w_max)

        step = 1.0
        for it in range(1, self.max_iter + 1):
            w_old = w.copy()
            Sw = Sigma @ w
            rc = w * Sw
            r_bar = rc.mean()
            resid = rc - r_bar

            grad = 2.0 * (Sw * resid + w * (Sigma @ resid))

            while True:
                w_new = self._project_simplex_box(w - step * grad, self.w_max)
                if self._objective(w_new) <= self._objective(w) - 1e-12 * step * np.dot(grad, w - w_new) or step < 1e-12:
                    break
                step *= 0.5

            w = w_new
            step = min(step * 1.5, 1.0)

            if np.linalg.norm(w - w_old, ord=np.inf) < self.tol:
                self.n_iter_ = it
                self.converged_ = True
                break
        else:
            self.n_iter_ = self.max_iter
            self.converged_ = False

        return w

    def _objective(self, w: np.ndarray) -> float:
        rc = w * (self.Sigma @ w)
        return float(np.sum((rc - rc.mean()) ** 2))

    @staticmethod
    def _project_simplex_box(v: np.ndarray, u: float) -> np.ndarray:
        """Project ``v`` onto { w : sum(w)=1, 0 <= w <= u }."""
        n = v.size
        if u * n < 1.0 - 1e-12:
            raise ValueError("Infeasible box for simplex projection.")
        lo, hi = v.min() - 1.0, v.max()
        for _ in range(100):
            tau = 0.5 * (lo + hi)
            w = np.clip(v - tau, 0.0, u)
            s = w.sum()
            if abs(s - 1.0) < 1e-12:
                return w
            if s > 1.0:
                lo = tau
            else:
                hi = tau
        return np.clip(v - 0.5 * (lo + hi), 0.0, u)
