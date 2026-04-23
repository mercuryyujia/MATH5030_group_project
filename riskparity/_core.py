"""Core solvers for risk parity portfolio optimization.

This module provides two solvers for the risk parity problem:

* :class:`CCDSolver` - a Cyclical Coordinate Descent baseline for the
  unconstrained long-only problem.
* :class:`SCASolver` - a constrained solver supporting per-asset upper
  bounds ``w_i <= w_max``.

Both solvers return weights ``w`` with ``w_i >= 0`` and ``sum(w) == 1``.
"""

from __future__ import annotations

import numpy as np


def _validate_covariance(Sigma: np.ndarray) -> np.ndarray:
    """Validate and return a covariance matrix."""
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square 2-D array.")
    if not np.all(np.isfinite(Sigma)):
        raise ValueError("Sigma must contain only finite values.")
    if not np.allclose(Sigma, Sigma.T, atol=1e-10):
        raise ValueError("Sigma must be symmetric.")
    if np.any(np.diag(Sigma) <= 0.0):
        raise ValueError("Sigma must have strictly positive diagonal entries.")
    return Sigma


def _validate_tol(tol: float) -> float:
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite float.")
    return tol


def _validate_max_iter(max_iter: int) -> int:
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    return max_iter


def risk_contributions(Sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return per-asset risk contributions ``w_i * (Sigma w)_i``."""
    Sigma = _validate_covariance(Sigma)
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or w.size != Sigma.shape[0]:
        raise ValueError("w must be a 1-D array with the same length as Sigma.")
    if not np.all(np.isfinite(w)):
        raise ValueError("w must contain only finite values.")
    return w * (Sigma @ w)


def relative_risk_contributions(Sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return risk contributions normalised to sum to one."""
    rc = risk_contributions(Sigma, w)
    total = rc.sum()
    if total <= 0.0:
        raise ValueError("Total risk contribution must be positive.")
    return rc / total


def risk_contribution_gap(Sigma: np.ndarray, w: np.ndarray) -> float:
    """Return the max absolute deviation from equal risk contributions."""
    rc = risk_contributions(Sigma, w)
    return float(np.max(np.abs(rc - rc.mean())))


class CCDSolver:
    """Cyclical Coordinate Descent solver for unconstrained risk parity.

    The algorithm solves the convex surrogate

        min_{w > 0}  0.5 * w^T Sigma w  -  (1/n) * sum_i log(w_i)

    and rescales the result so that ``sum(w) = 1``.
    """

    def __init__(self, Sigma: np.ndarray, tol: float = 1e-8, max_iter: int = 1000):
        self.Sigma = _validate_covariance(Sigma)
        self.tol = _validate_tol(tol)
        self.max_iter = _validate_max_iter(max_iter)
        self.n_iter_: int | None = None
        self.converged_: bool = False
        self.objective_: float | None = None
        self.risk_contribution_gap_: float | None = None

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

        w = self._normalise_weights(w)
        self.objective_ = self._objective(w)
        self.risk_contribution_gap_ = risk_contribution_gap(Sigma, w)
        return w

    def _objective(self, w: np.ndarray) -> float:
        n = w.size
        return float(0.5 * w @ self.Sigma @ w - np.log(w).sum() / n)

    @staticmethod
    def _normalise_weights(w: np.ndarray) -> np.ndarray:
        total = float(w.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise FloatingPointError("Computed weights must have a positive finite sum.")
        w = w / total
        if np.any(w <= 0.0):
            raise FloatingPointError("CCD produced non-positive weights.")
        return w


class SCASolver:
    """Successive Convex Approximation solver with per-asset upper bounds.

    The solver minimises the dispersion of risk contributions under

        sum_i w_i = 1,   0 <= w_i <= w_max.
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
        self.w_max = float(w_max)
        if not np.isfinite(self.w_max) or self.w_max <= 0.0 or self.w_max > 1.0:
            raise ValueError("w_max must lie in (0, 1].")
        if self.w_max * n < 1.0 - 1e-12:
            raise ValueError(
                f"w_max={self.w_max} is infeasible for n={n}: need w_max >= 1/n."
            )
        self.tol = _validate_tol(tol)
        self.max_iter = _validate_max_iter(max_iter)
        self.n_iter_: int | None = None
        self.converged_: bool = False
        self.objective_: float | None = None
        self.risk_contribution_gap_: float | None = None

    def solve(self) -> np.ndarray:
        Sigma = self.Sigma

        w = CCDSolver(Sigma, tol=min(self.tol, 1e-8), max_iter=500).solve()
        w = self._project_simplex_box(np.clip(w, 0.0, self.w_max), self.w_max)

        step = 1.0
        current_obj = self._objective(w)

        for it in range(1, self.max_iter + 1):
            w_old = w.copy()
            Sw = Sigma @ w
            rc = w * Sw
            resid = rc - rc.mean()
            grad = 2.0 * (Sw * resid + w * (Sigma @ resid))

            local_step = step
            while True:
                candidate = self._project_simplex_box(
                    w - local_step * grad,
                    self.w_max,
                )
                candidate_obj = self._objective(candidate)
                sufficient_decrease = current_obj - candidate_obj
                if sufficient_decrease >= 1e-12 or local_step <= 1e-12:
                    break
                local_step *= 0.5

            w = candidate
            current_obj = candidate_obj
            step = min(local_step * 1.5, 1.0)

            if np.linalg.norm(w - w_old, ord=np.inf) < self.tol:
                self.n_iter_ = it
                self.converged_ = True
                break
        else:
            self.n_iter_ = self.max_iter
            self.converged_ = False

        self._check_solution(w)
        self.objective_ = current_obj
        self.risk_contribution_gap_ = risk_contribution_gap(Sigma, w)
        return w

    def _objective(self, w: np.ndarray) -> float:
        rc = risk_contributions(self.Sigma, w)
        return float(np.sum((rc - rc.mean()) ** 2))

    def _check_solution(self, w: np.ndarray) -> None:
        if not np.all(np.isfinite(w)):
            raise FloatingPointError("Solver returned non-finite weights.")
        if not np.isclose(w.sum(), 1.0, atol=1e-8):
            raise FloatingPointError("Solver returned weights that do not sum to 1.")
        if np.any(w < -1e-10):
            raise FloatingPointError("Solver returned negative weights.")
        if np.any(w > self.w_max + 1e-10):
            raise FloatingPointError("Solver violated the upper-bound constraint.")

    @staticmethod
    def _project_simplex_box(v: np.ndarray, u: float) -> np.ndarray:
        """Project ``v`` onto ``{w : sum(w)=1, 0 <= w <= u}``."""
        v = np.asarray(v, dtype=float)
        if v.ndim != 1:
            raise ValueError("v must be a 1-D array.")
        if not np.all(np.isfinite(v)):
            raise ValueError("v must contain only finite values.")
        if not np.isfinite(u) or u <= 0.0:
            raise ValueError("u must be a positive finite float.")
        n = v.size
        if u * n < 1.0 - 1e-12:
            raise ValueError("Infeasible box for simplex projection.")
        lo, hi = v.min() - u, v.max()
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
