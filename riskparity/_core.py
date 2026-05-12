"""Core solvers for risk parity portfolio optimization.

This module provides two solvers for the risk parity problem:

* :class:`CCDSolver` - a wrapper around PyFENG's Cyclical Coordinate
  Descent implementation for the unconstrained long-only problem.
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


def _validate_positive_definite_covariance(Sigma: np.ndarray) -> np.ndarray:
    """Validate and return a positive definite covariance matrix."""
    Sigma = _validate_covariance(Sigma)
    if np.min(np.linalg.eigvalsh(Sigma)) <= 0.0:
        raise ValueError("Sigma must be positive definite.")
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


def _make_pyfeng_risk_parity_model(pf, Sigma: np.ndarray):
    """Instantiate PyFENG RiskParity across versions with/without ``cov``."""
    try:
        return pf.RiskParity(cov=Sigma)
    except TypeError as exc:
        if "cov" not in str(exc):
            raise

    sigma = np.sqrt(np.diag(Sigma))
    cor = Sigma / np.outer(sigma, sigma)
    try:
        return pf.RiskParity(sigma=sigma, cor=cor)
    except TypeError:
        return pf.RiskParity(sigma, cor)


class CCDSolver:
    """Cyclical Coordinate Descent solver for unconstrained risk parity.

    This class preserves the package's ``CCDSolver(...).solve()`` API while
    delegating the numerical solve to :class:`pyfeng.RiskParity`, which
    implements the improved CCD method of Choi and Chen (2022).

    PyFENG's current API accepts ``tol`` but does not expose ``max_iter``;
    the argument is retained here for backward compatibility with existing
    project code.
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
        try:
            import pyfeng as pf
        except ImportError as exc:
            raise ImportError(
                "CCDSolver requires PyFENG. Install this package with its "
                "project dependencies, or run `pip install pyfeng`."
            ) from exc

        model = _make_pyfeng_risk_parity_model(pf, self.Sigma)
        w = model.weight(tol=self.tol)
        result = getattr(model, "_result", {})
        self.n_iter_ = int(result.get("n_iter", 0)) or None
        err = result.get("err")
        self.converged_ = bool(w is not None and err is not None and err < self.tol)
        if w is None:
            raise FloatingPointError("PyFENG RiskParity failed to converge.")

        w = self._normalise_weights(np.asarray(w, dtype=float))
        self.objective_ = self._objective(w)
        self.risk_contribution_gap_ = risk_contribution_gap(self.Sigma, w)
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

    Parameters
    ----------
    sigma : np.ndarray, optional
        Asset standard deviations. For backward compatibility, a square
        2-D array passed as the first positional argument is treated as
        ``cov``.
    cor : np.ndarray or float, optional
        Correlation matrix, or a scalar constant correlation.
    cov : np.ndarray, optional
        Covariance matrix. This takes precedence over ``sigma`` and ``cor``.
    w_max : float, optional
        Upper bound on each asset weight. Must satisfy ``w_max >= 1/n``.
    ret : np.ndarray, optional
        Expected returns. Stored for API compatibility; not used by SCA.
    budget : np.ndarray, optional
        Risk budgets. Currently only equal budgets are supported.
    longshort : np.ndarray or int, optional
        Long/short flags. Currently only long-only portfolios are supported.
    tol : float, optional
        Convergence tolerance for maximum weight change.
    max_iter : int, optional
        Maximum number of SCA iterations.

    Attributes
    ----------
    _result : dict
        Diagnostics for the last call to :meth:`weight` or :meth:`solve`.
    """

    def __init__(
        self,
        sigma: np.ndarray | None = None,
        cor: np.ndarray | float | None = None,
        cov: np.ndarray | None = None,
        w_max: float = 1.0,
        ret: np.ndarray | None = None,
        budget: np.ndarray | None = None,
        longshort: np.ndarray | int = 1,
        tol: float = 1e-6,
        max_iter: int = 200,
    ):
        sigma_arr = None if sigma is None else np.asarray(sigma)
        if (
            cov is None
            and sigma_arr is not None
            and sigma_arr.ndim == 2
            and sigma_arr.shape[0] == sigma_arr.shape[1]
            and cor is not None
            and np.isscalar(cor)
            and w_max == 1.0
        ):
            w_max = cor
            cor = None

        self.Sigma = self._covariance_from_inputs(sigma=sigma, cor=cor, cov=cov)
        self.cov = self.Sigma
        self.cov_m = self.Sigma
        self.sigma = np.sqrt(np.diag(self.Sigma))
        self.cor_m = self.Sigma / np.outer(self.sigma, self.sigma)
        n = self.Sigma.shape[0]
        self.n_asset = n
        self.w_max = float(w_max)
        if not np.isfinite(self.w_max) or self.w_max <= 0.0 or self.w_max > 1.0:
            raise ValueError("w_max must lie in (0, 1].")
        if self.w_max * n < 1.0 - 1e-12:
            raise ValueError(
                f"w_max={self.w_max} is infeasible for n={n}: need w_max >= 1/n."
            )
        if budget is not None:
            budget = np.asarray(budget, dtype=float)
            if budget.shape != (n,) or not np.allclose(budget, 1.0 / n):
                raise ValueError("budget is not yet supported except equal budgets.")
        if longshort is None:
            longshort_arr = np.ones(n, dtype=np.int8)
        elif np.isscalar(longshort):
            longshort_arr = np.full(n, np.sign(longshort), dtype=np.int8)
        else:
            longshort_arr = np.sign(longshort).astype(np.int8)
        if longshort_arr.shape != (n,) or np.any(longshort_arr != 1):
            raise ValueError("longshort is not yet supported except long-only.")
        self.ret = None if ret is None else np.asarray(ret, dtype=float)
        if self.ret is not None and self.ret.shape not in {(n,), ()}:
            raise ValueError("ret must be scalar or have one value per asset.")
        self.budget = np.full(n, 1.0 / n)
        self.longshort = longshort_arr
        self.tol = _validate_tol(tol)
        self.max_iter = _validate_max_iter(max_iter)
        self.n_iter_: int | None = None
        self.converged_: bool = False
        self.objective_: float | None = None
        self.risk_contribution_gap_: float | None = None
        self._result: dict[str, float | int | bool] = {}

    @staticmethod
    def _covariance_from_inputs(
        sigma: np.ndarray | None = None,
        cor: np.ndarray | float | None = None,
        cov: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a covariance matrix from PyFENG-style constructor inputs."""
        if cov is not None:
            return _validate_positive_definite_covariance(cov)

        if sigma is None:
            raise ValueError("Either cov or sigma must be provided.")

        sigma_arr = np.asarray(sigma, dtype=float)
        if (
            sigma_arr.ndim == 2
            and sigma_arr.shape[0] == sigma_arr.shape[1]
            and cor is None
        ):
            return _validate_positive_definite_covariance(sigma_arr)

        sigma_arr = np.atleast_1d(sigma_arr)
        if sigma_arr.ndim != 1 or sigma_arr.size < 2:
            raise ValueError("sigma must be a 1-D array with at least two assets.")
        if not np.all(np.isfinite(sigma_arr)) or np.any(sigma_arr <= 0.0):
            raise ValueError("sigma must contain only positive finite values.")

        n = sigma_arr.size
        if cor is None:
            cor_m = np.eye(n)
        elif np.isscalar(cor):
            rho = float(cor)
            if not np.isfinite(rho) or rho <= -1.0 / (n - 1) or rho >= 1.0:
                raise ValueError(
                    "scalar cor must define a positive definite correlation matrix."
                )
            cor_m = rho * np.ones((n, n)) + (1.0 - rho) * np.eye(n)
        else:
            cor_m = np.asarray(cor, dtype=float)
            if cor_m.shape != (n, n):
                raise ValueError("cor must be a square matrix matching sigma.")
            if not np.allclose(np.diag(cor_m), 1.0, atol=1e-10):
                raise ValueError("cor must have ones on the diagonal.")

        return _validate_positive_definite_covariance(
            np.outer(sigma_arr, sigma_arr) * cor_m
        )

    def weight(self, tol: float | None = None) -> np.ndarray:
        """Compute constrained risk parity weights.

        Parameters
        ----------
        tol : float, optional
            Override the instance convergence tolerance for this call.

        Returns
        -------
        np.ndarray
            Portfolio weights satisfying ``sum(w)=1`` and
            ``0 <= w_i <= w_max``.

        Raises
        ------
        FloatingPointError
            If the SCA loop does not converge within ``max_iter``.
        """
        return self._solve(tol=tol, raise_on_nonconvergence=True)

    def solve(self) -> np.ndarray:
        """Return constrained risk parity weights.

        This method keeps the original project API. Use :meth:`weight` for
        the PyFENG-style API, which raises if the SCA loop does not converge.
        """
        return self._solve(tol=None, raise_on_nonconvergence=False)

    def _solve(
        self,
        tol: float | None = None,
        raise_on_nonconvergence: bool = False,
    ) -> np.ndarray:
        Sigma = self.Sigma
        tol_eff = self.tol if tol is None else _validate_tol(tol)

        w = CCDSolver(Sigma, tol=min(tol_eff, 1e-8), max_iter=500).solve()
        w = self._project_simplex_box(np.clip(w, 0.0, self.w_max), self.w_max)

        step = 1.0
        current_obj = self._objective(w)
        err = np.inf

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

            err = float(np.linalg.norm(w - w_old, ord=np.inf))
            if err < tol_eff:
                self.n_iter_ = it
                self.converged_ = True
                break
        else:
            self.n_iter_ = self.max_iter
            self.converged_ = False

        self._check_solution(w)
        self.objective_ = current_obj
        self.risk_contribution_gap_ = risk_contribution_gap(Sigma, w)
        self._result = {
            "n_iter": int(self.n_iter_),
            "err": float(err),
            "objective": float(self.objective_),
            "gap": float(self.risk_contribution_gap_),
            "converged": bool(self.converged_),
        }
        if raise_on_nonconvergence and not self.converged_:
            raise FloatingPointError("SCASolver failed to converge.")
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
