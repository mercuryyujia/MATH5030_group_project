"""Tests for ``riskparity._core``: validation, CCD/SCA solvers, and robustness."""

import os
import sys

sys.path.append(os.path.abspath("."))

import numpy as np
import pytest

from riskparity._core import (
    CCDSolver,
    SCASolver,
    _validate_covariance,
    _validate_max_iter,
    _validate_tol,
    risk_contributions,
    relative_risk_contributions,
    risk_contribution_gap,
)

# Public API

def test_public_api_imports():
    """Check that the main solvers are exposed through the package API."""
    import riskparity
    assert hasattr(riskparity, "CCDSolver")
    assert hasattr(riskparity, "SCASolver")
    assert hasattr(riskparity, "risk_contributions")


def _random_spd_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw a symmetric positive definite matrix (well conditioned)."""
    a = rng.standard_normal((n, n))
    q, _ = np.linalg.qr(a)
    d = 0.5 + rng.random(n)
    return q @ np.diag(d) @ q.T


def _equicorrelation_covariance(n: int, rho: float, vol: np.ndarray) -> np.ndarray:
    """Constant-correlation covariance with per-asset volatilities on the diagonal."""
    c = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    return np.outer(vol, vol) * c


# Reusable SPD covariance matrices for tests

COV_2 = np.array([[0.04, 0.01], [0.01, 0.09]])
COV_3 = np.array(
    [
        [0.04, 0.01, 0.00],
        [0.01, 0.09, 0.02],
        [0.00, 0.02, 0.16],
    ]
)


# Covariance validation

def test_validate_covariance_accepts_valid_matrix():
    Sigma = COV_2.copy()
    out = _validate_covariance(Sigma)
    assert out.shape == (2, 2)
    assert out.dtype == np.float64
    assert np.allclose(out, Sigma)
    assert np.array_equal(out, Sigma.astype(float))


def test_validate_covariance_accepts_integer_dtype():
    Sigma = np.array([[4, 1], [1, 9]], dtype=np.int64)
    out = _validate_covariance(Sigma)
    assert out.dtype == np.float64
    assert np.allclose(out, Sigma.astype(float))


def test_validate_covariance_rejects_non_square_matrix():
    Sigma = np.ones((2, 3))
    with pytest.raises(ValueError, match=r"Sigma must be a square 2-D array\."):
        _validate_covariance(Sigma)


def test_validate_covariance_rejects_one_dimensional():
    Sigma = np.ones(3)
    with pytest.raises(ValueError, match=r"Sigma must be a square 2-D array\."):
        _validate_covariance(Sigma)


def test_validate_covariance_rejects_three_dimensional():
    Sigma = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match=r"Sigma must be a square 2-D array\."):
        _validate_covariance(Sigma)


def test_validate_covariance_rejects_non_symmetric_matrix():
    Sigma = np.array([[1.0, 0.2], [0.1, 1.0]])
    with pytest.raises(ValueError, match=r"Sigma must be symmetric\."):
        _validate_covariance(Sigma)


def test_validate_covariance_accepts_near_symmetric_within_atol():
    """Accept tiny off-diagonal asymmetry within the symmetry tolerance."""
    Sigma = np.array([[1.0, 0.0], [1e-11, 1.0]])
    out = _validate_covariance(Sigma)
    assert out.shape == (2, 2)
    assert np.allclose(out, out.T, atol=1e-10)


def test_validate_covariance_rejects_asymmetric_beyond_atol():
    Sigma = np.array([[1.0, 1e-9], [0.0, 1.0]])
    with pytest.raises(ValueError, match=r"Sigma must be symmetric\."):
        _validate_covariance(Sigma)


def test_validate_covariance_rejects_nonpositive_diagonal():
    Sigma = np.array([[0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(
        ValueError, match=r"Sigma must have strictly positive diagonal entries\."
    ):
        _validate_covariance(Sigma)


def test_validate_covariance_rejects_negative_diagonal():
    Sigma = np.array([[-0.01, 0.0], [0.0, 0.04]])
    with pytest.raises(
        ValueError, match=r"Sigma must have strictly positive diagonal entries\."
    ):
        _validate_covariance(Sigma)


def test_validate_covariance_rejects_nonfinite_entries():
    Sigma = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError, match=r"Sigma must contain only finite values\."):
        _validate_covariance(Sigma)


def test_validate_covariance_rejects_infinite_entries():
    Sigma = np.array([[1.0, np.inf], [np.inf, 1.0]])
    with pytest.raises(ValueError, match=r"Sigma must contain only finite values\."):
        _validate_covariance(Sigma)


def test_validate_covariance_rejects_neg_infinite_entries():
    Sigma = np.array([[1.0, -np.inf], [-np.inf, 1.0]])
    with pytest.raises(ValueError, match=r"Sigma must contain only finite values\."):
        _validate_covariance(Sigma)


def test_validate_tol_rejects_nonpositive():
    pat = r"tol must be a positive finite float\."
    with pytest.raises(ValueError, match=pat):
        _validate_tol(0.0)
    with pytest.raises(ValueError, match=pat):
        _validate_tol(-1e-6)


def test_validate_max_iter_rejects_nonpositive():
    with pytest.raises(ValueError, match=r"max_iter must be at least 1\."):
        _validate_max_iter(0)


# CCDSolver (correctness)

def test_ccd_sets_internal_attributes_after_solve():
    """Check that CCDSolver stores diagnostics after solve()."""
    Sigma = np.eye(3)
    solver = CCDSolver(Sigma)
    w = solver.solve()
    assert np.isclose(w.sum(), 1.0)
    assert solver.n_iter_ is not None
    assert isinstance(solver.converged_, bool)
    assert solver.objective_ is not None
    assert solver.risk_contribution_gap_ is not None


def test_ccd_returns_valid_weights():
    Sigma = COV_3.copy()
    w = CCDSolver(Sigma).solve()
    assert w.shape == (3,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-10)
    assert np.all(w > 0.0)


def test_ccd_identity_covariance_gives_equal_weights():
    Sigma = np.eye(4)
    w = CCDSolver(Sigma).solve()
    expected = np.full(4, 0.25)
    assert np.allclose(w, expected, atol=1e-6)


def test_ccd_diagonal_covariance_matches_inverse_vol_weights():
    variances = np.array([0.04, 0.09, 0.16, 0.25])
    Sigma = np.diag(variances)
    w = CCDSolver(Sigma).solve()
    inv_vol = 1.0 / np.sqrt(variances)
    expected = inv_vol / inv_vol.sum()
    assert np.allclose(w, expected, atol=1e-6)


def test_ccd_two_asset_sanity_check_diagonal_case():
    Sigma = np.diag([0.04, 0.09])
    w = CCDSolver(Sigma).solve()
    expected = np.array([3.0, 2.0]) / 5.0
    assert np.allclose(w, expected, atol=1e-6)


# Analytical and regression tests

def test_ccd_fixed_covariance_regression_reference_weights():
    """Reference regression on a fixed SPD matrix to catch implementation drift."""
    Sigma = COV_3.copy()
    w = CCDSolver(Sigma, tol=1e-12, max_iter=8000).solve()
    expected = np.array(
        [0.4731662854646679, 0.2902505718029981, 0.2365831427323339],
        dtype=float,
    )
    assert np.allclose(w, expected, atol=1e-7)


def test_ccd_matches_riskparityportfolio_readme_reference_example():
    """Validate against a reference example from the riskParityPortfolio README (R)."""
    # Source example:
    # set.seed(42); N <- 5; V <- matrix(rnorm(N^2), ncol = N); Sigma <- cov(V)
    # The expected weights are the README output from riskParityPortfolio(Sigma).
    Sigma = np.array(
        [
            [
                0.4801188807777429,
                -0.30572104311802423,
                -0.25679598847701546,
                0.14211478575036879,
                0.42071591100057448,
            ],
            [
                -0.30572104311802423,
                1.0622380648411025,
                0.4313019214811612,
                -0.75419580442894918,
                -0.21652431205803163,
            ],
            [
                -0.25679598847701546,
                0.4313019214811612,
                2.0786635079669096,
                1.4000578790801681,
                -1.2825690301607684,
            ],
            [
                0.14211478575036879,
                -0.75419580442894918,
                1.4000578790801681,
                3.2235720789574756,
                0.22194856932813867,
            ],
            [
                0.42071591100057448,
                -0.21652431205803163,
                -1.2825690301607684,
                0.22194856932813867,
                2.0548335344840334,
            ],
        ]
    )
    expected = np.array([0.32715962, 0.27110678, 0.14480081, 0.09766356, 0.15926922])
    w = CCDSolver(Sigma, tol=1e-12, max_iter=10000).solve()
    assert np.allclose(w, expected, atol=1e-7)


def test_ccd_two_asset_correlated_case_matches_closed_form_inverse_vol():
    """For 2 assets, equal-risk solution has inverse-vol form even with correlation."""
    Sigma = np.array([[0.04, 0.018], [0.018, 0.09]])
    w = CCDSolver(Sigma, tol=1e-12, max_iter=5000).solve()
    vol = np.sqrt(np.diag(Sigma))
    expected = (1.0 / vol) / np.sum(1.0 / vol)
    assert np.allclose(w, expected, atol=1e-6)
    rc = risk_contributions(Sigma, w)
    assert np.allclose(rc[0], rc[1], atol=1e-8)


@pytest.mark.parametrize(
    "Sigma",
    [
        COV_2.copy(),
        np.array([[0.04, 0.018], [0.018, 0.09]], dtype=float),
    ],
)
def test_sca_matches_ccd_when_w_max_is_non_binding_on_representative_cases(Sigma):
    """With non-binding cap, constrained and unconstrained solvers should agree."""
    w_ccd = CCDSolver(Sigma, tol=1e-12, max_iter=5000).solve()
    w_sca = SCASolver(Sigma, w_max=1.0, tol=1e-10, max_iter=2000).solve()
    assert np.allclose(w_sca, w_ccd, atol=1e-5)


# Risk contributions

def test_risk_contributions_sum_to_portfolio_variance():
    Sigma = COV_2.copy()
    w = np.array([0.6, 0.4])
    rc = risk_contributions(Sigma, w)
    portfolio_variance = w @ Sigma @ w
    assert np.isclose(rc.sum(), portfolio_variance)


def test_relative_risk_contributions_sum_to_one():
    Sigma = COV_2.copy()
    w = np.array([0.6, 0.4])
    rrc = relative_risk_contributions(Sigma, w)
    assert np.isclose(rrc.sum(), 1.0)
    assert np.all(rrc >= 0.0)


def test_ccd_risk_contributions_are_close_to_equal():
    Sigma = COV_3.copy()
    w = CCDSolver(Sigma, tol=1e-10, max_iter=2000).solve()
    rc = risk_contributions(Sigma, w)
    assert np.allclose(rc, np.full(3, rc.mean()), atol=1e-6)


def test_ccd_risk_contribution_gap_is_small():
    Sigma = COV_3.copy()
    w = CCDSolver(Sigma, tol=1e-10, max_iter=2000).solve()
    gap = risk_contribution_gap(Sigma, w)
    assert gap < 1e-6


def test_risk_contributions_rejects_weight_length_mismatch():
    Sigma = np.eye(3)
    w = np.array([0.5, 0.5])
    with pytest.raises(
        ValueError, match=r"w must be a 1-D array with the same length as Sigma\."
    ):
        risk_contributions(Sigma, w)


# CCD stress (scale / conditioning)

def test_ccd_large_random_spd_portfolio():
    rng = np.random.default_rng(0)
    n = 64
    Sigma = _random_spd_matrix(rng, n)
    solver = CCDSolver(Sigma, tol=1e-9, max_iter=5000)
    w = solver.solve()
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-9)
    assert np.all(w > 0.0)
    assert solver.converged_ is True
    gap = risk_contribution_gap(Sigma, w)
    assert gap < 1e-5


def test_ccd_highly_correlated_equicorrelation():
    n = 40
    rho = 0.92
    vol = np.sqrt(np.linspace(0.01, 0.05, n))
    Sigma = _equicorrelation_covariance(n, rho, vol)
    w = CCDSolver(Sigma, tol=1e-9, max_iter=8000).solve()
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-9)
    rc = risk_contributions(Sigma, w)
    assert np.allclose(rc, np.full(n, rc.mean()), rtol=0, atol=5e-5)


def test_ccd_near_singular_low_rank_plus_jitter():
    rng = np.random.default_rng(1)
    n = 35
    rank = 4
    b = rng.standard_normal((n, rank))
    Sigma = b @ b.T + 1e-5 * np.eye(n)
    Sigma = 0.5 * (Sigma + Sigma.T)
    w = CCDSolver(Sigma, tol=1e-9, max_iter=10000).solve()
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-8)
    gap = risk_contribution_gap(Sigma, w)
    assert gap < 1e-4


# Robustness: random SPD (CCD + SCA)

@pytest.mark.parametrize(
    "seed,n",
    [
        (0, 18),
        (1, 24),
        (2, 30),
        (3, 22),
        (4, 36),
        (5, 20),
    ],
)
def test_ccd_random_spd_robust_across_seeds_and_sizes(seed, n):
    rng = np.random.default_rng(seed)
    Sigma = _random_spd_matrix(rng, n)
    solver = CCDSolver(Sigma, tol=1e-8, max_iter=12000)
    w = solver.solve()
    assert w.shape == (n,)
    assert w.dtype == np.float64
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-10)
    assert np.all(w > 0.0)
    assert solver.converged_ is True
    gap = risk_contribution_gap(Sigma, w)
    assert gap < 5e-5


def test_ccd_random_spd_many_consecutive_draws_same_stream():
    """Many independent SPD draws from one RNG stream (stability smoke)."""
    rng = np.random.default_rng(2026)
    n = 16
    for _ in range(40):
        Sigma = _random_spd_matrix(rng, n)
        w = CCDSolver(Sigma, tol=1e-8, max_iter=8000).solve()
        assert np.all(np.isfinite(w))
        assert np.isclose(w.sum(), 1.0, atol=1e-9)
        assert np.all(w > 0.0)
        assert risk_contribution_gap(Sigma, w) < 1e-4


@pytest.mark.parametrize(
    "seed,n,w_max",
    [
        (10, 14, 0.5),
        (11, 25, 0.55),
        (12, 30, 0.6),
        (13, 18, 1.0),
        (14, 22, 0.48),
    ],
)
def test_sca_random_spd_feasible_box_and_simplex(seed, n, w_max):
    assert w_max * n >= 1.0 - 1e-12
    rng = np.random.default_rng(seed)
    Sigma = _random_spd_matrix(rng, n)
    w = SCASolver(Sigma, w_max=w_max, tol=1e-7, max_iter=2000).solve()
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= -1e-8)
    assert np.all(w <= w_max + 1e-7)


@pytest.mark.parametrize("seed", [0, 3, 7, 11, 15])
def test_ccd_vs_sca_random_spd_non_binding_w_max(seed):
    """With w_max=1, SCA should stay close to CCD on generic SPD draws."""
    rng = np.random.default_rng(5000 + seed)
    n = 12
    Sigma = _random_spd_matrix(rng, n)
    w_ccd = CCDSolver(Sigma, tol=1e-9, max_iter=8000).solve()
    w_sca = SCASolver(Sigma, w_max=1.0, tol=1e-8, max_iter=1200).solve()
    assert np.allclose(w_sca, w_ccd, atol=5e-3)


# Robustness: cross-dimensional stability

@pytest.mark.parametrize("n", [2, 3, 6, 12, 24, 48, 72, 96])
def test_ccd_identity_covariance_stable_across_dimensions(n):
    Sigma = np.eye(n)
    w = CCDSolver(Sigma, tol=1e-10, max_iter=min(6000, max(1500, 40 * n))).solve()
    assert w.shape == (n,)
    assert np.allclose(w, np.full(n, 1.0 / n), rtol=0.0, atol=1e-8)
    assert np.isclose(w.sum(), 1.0, atol=1e-12)


@pytest.mark.parametrize("n", [3, 8, 16, 24, 32, 48, 64])
def test_ccd_random_spd_stable_across_dimensions(n):
    rng = np.random.default_rng(9000 + n)
    Sigma = _random_spd_matrix(rng, n)
    tol = 1e-8 if n <= 40 else 5e-8
    max_iter = min(10000, max(4000, 80 * n))
    solver = CCDSolver(Sigma, tol=tol, max_iter=max_iter)
    w = solver.solve()
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-9)
    assert np.all(w > 0.0)
    assert solver.converged_ is True
    gap = risk_contribution_gap(Sigma, w)
    gap_cap = max(8e-5, 5e-7 * n)
    assert gap < gap_cap


@pytest.mark.parametrize("n", [10, 20, 32, 44, 56])
def test_ccd_equicorrelation_stable_across_dimensions(n):
    rho = 0.84
    vol = np.sqrt(np.linspace(0.015, 0.055, n))
    Sigma = _equicorrelation_covariance(n, rho, vol)
    max_iter = min(12000, max(6000, 100 * n))
    solver = CCDSolver(Sigma, tol=1e-8, max_iter=max_iter)
    w = solver.solve()
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-8)
    assert np.all(w > 0.0)
    assert solver.converged_ is True
    rc = risk_contributions(Sigma, w)
    atol_rc = 8e-4 if n >= 44 else 3e-4
    assert np.allclose(rc, np.full(n, rc.mean()), rtol=0.0, atol=atol_rc)


@pytest.mark.parametrize("n", [8, 16, 28, 40, 52])
def test_sca_random_spd_stable_across_dimensions(n):
    rng = np.random.default_rng(11000 + n)
    Sigma = _random_spd_matrix(rng, n)
    max_iter = min(2500, max(1200, 20 * n))
    w = SCASolver(Sigma, w_max=1.0, tol=1e-7, max_iter=max_iter).solve()
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= -1e-8)
    assert np.all(w <= 1.0 + 1e-7)


@pytest.mark.parametrize("n,rank", [(20, 3), (32, 4), (44, 5), (36, 4)])
def test_ccd_low_rank_spd_jitter_stable_across_dimensions(n, rank):
    rng = np.random.default_rng(13000 + n + rank)
    b = rng.standard_normal((n, rank))
    Sigma = b @ b.T + 1e-5 * np.eye(n)
    Sigma = 0.5 * (Sigma + Sigma.T)
    max_iter = min(20000, max(10000, 150 * n))
    solver = CCDSolver(Sigma, tol=5e-8, max_iter=max_iter)
    w = solver.solve()
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-7)
    assert np.all(w > 0.0)
    gap = risk_contribution_gap(Sigma, w)
    assert gap < 2e-3
    assert solver.converged_ or gap < 1e-3


# Robustness: boundary parameters (tol, max_iter, w_max)

def test_ccd_max_iter_minimum_still_returns_valid_weights():
    Sigma = np.eye(3)
    solver = CCDSolver(Sigma, tol=1e-6, max_iter=1)
    w = solver.solve()
    assert w.shape == (3,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-12)
    assert np.all(w > 0.0)


def test_ccd_very_loose_tol_converges_immediately_on_small_problem():
    Sigma = COV_2.copy()
    solver = CCDSolver(Sigma, tol=0.2, max_iter=500)
    w = solver.solve()
    assert solver.converged_ is True
    assert solver.n_iter_ is not None and solver.n_iter_ <= 3
    assert np.isclose(w.sum(), 1.0, atol=1e-10)
    assert np.all(w > 0.0)


def test_ccd_extremely_tight_tol_small_system():
    Sigma = np.diag([0.04, 0.09, 0.16])
    solver = CCDSolver(Sigma, tol=1e-14, max_iter=20000)
    w = solver.solve()
    assert solver.converged_ is True
    assert np.isclose(w.sum(), 1.0, atol=1e-12)
    assert risk_contribution_gap(Sigma, w) < 1e-10


def test_ccd_accepts_min_positive_tol_float():
    Sigma = np.eye(2)
    tol = np.finfo(float).tiny
    w = CCDSolver(Sigma, tol=float(tol), max_iter=3000).solve()
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-10)


@pytest.mark.parametrize("n", [3, 6, 10])
def test_sca_w_max_just_feasible_above_inverse_n(n):
    """Slightly above 1/n must be accepted; solution stays on simplex + box."""
    w_max = (1.0 - 9e-13) / n + 2e-12
    assert w_max <= 1.0
    Sigma = np.eye(n)
    w = SCASolver(Sigma, w_max=w_max, tol=1e-8, max_iter=600).solve()
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w <= w_max + 1e-7)
    assert np.all(w >= -1e-9)


def test_sca_max_iter_one_returns_feasible_point():
    Sigma = COV_3.copy()
    w = SCASolver(Sigma, w_max=0.5, tol=1e-6, max_iter=1).solve()
    assert w.shape == (3,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-8)
    assert np.all(w >= -1e-8)
    assert np.all(w <= 0.5 + 1e-7)


def test_sca_very_loose_tol_finishes_within_few_iterations():
    Sigma = np.diag([0.04, 0.09, 0.16])
    solver = SCASolver(Sigma, w_max=1.0, tol=0.05, max_iter=50)
    w = solver.solve()
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert solver.n_iter_ is not None and solver.n_iter_ <= 10


def test_sca_w_max_at_one_with_tiny_tol():
    rng = np.random.default_rng(77)
    Sigma = _random_spd_matrix(rng, 8)
    w = SCASolver(Sigma, w_max=1.0, tol=1e-12, max_iter=3000).solve()
    assert np.isclose(w.sum(), 1.0, atol=1e-7)
    assert np.all(w <= 1.0 + 1e-8)


def test_ccd_max_iter_large_does_not_break_small_identity():
    Sigma = np.eye(4)
    w = CCDSolver(Sigma, tol=1e-10, max_iter=50_000).solve()
    assert np.allclose(w, 0.25, rtol=0.0, atol=1e-8)


# Robustness tests: SCA box + simplex constraints

def test_sca_returns_feasible_weights():
    Sigma = COV_3.copy()
    w = SCASolver(Sigma, w_max=0.5, tol=1e-8, max_iter=500).solve()
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-7)
    assert np.all(w >= -1e-10)
    assert np.all(w <= 0.5 + 1e-8)


def test_sca_tight_cap_still_satisfies_simplex_and_bounds():
    Sigma = np.diag([0.04, 0.09, 0.16])
    w = SCASolver(Sigma, w_max=1.0 / 3.0 + 1e-12, tol=1e-8, max_iter=500).solve()
    assert np.isclose(w.sum(), 1.0, atol=1e-7)
    assert np.all(w <= 1.0 / 3.0 + 1e-7)


def test_sca_rejects_infeasible_w_max():
    Sigma = np.eye(5)
    with pytest.raises(ValueError, match=r"w_max=.*infeasible for n=5"):
        SCASolver(Sigma, w_max=0.15)


@pytest.mark.parametrize(
    "n,w_max",
    [
        (4, 0.24),
        (8, 0.124),
        (10, 0.099),
    ],
)
def test_sca_rejects_w_max_below_simplex_box_feasibility(n, w_max):
    Sigma = np.eye(n)
    with pytest.raises(ValueError, match=rf"w_max=.*infeasible for n={n}"):
        SCASolver(Sigma, w_max=w_max)


def test_sca_accepts_w_max_at_inverse_n_boundary():
    """At w_max = 1/n, SCA should return the equal-weight feasible solution."""
    n = 7
    Sigma = np.eye(n)
    w_max = 1.0 / n
    w = SCASolver(Sigma, w_max=w_max, tol=1e-9, max_iter=800).solve()
    assert np.isclose(w.sum(), 1.0, atol=1e-7)
    assert np.all(w <= w_max + 1e-7)
    assert np.all(w >= -1e-10)
    assert np.allclose(w, np.full(n, w_max), atol=1e-5)


def test_sca_rejects_invalid_w_max_domain():
    Sigma = np.eye(3)
    pat = r"w_max must lie in \(0, 1\]\."
    with pytest.raises(ValueError, match=pat):
        SCASolver(Sigma, w_max=0.0)
    with pytest.raises(ValueError, match=pat):
        SCASolver(Sigma, w_max=-0.05)
    with pytest.raises(ValueError, match=pat):
        SCASolver(Sigma, w_max=1.0 + 1e-6)
    with pytest.raises(ValueError, match=pat):
        SCASolver(Sigma, w_max=np.nan)
    with pytest.raises(ValueError, match=pat):
        SCASolver(Sigma, w_max=np.inf)


def test_sca_respects_explicit_w_max_grid():
    Sigma = COV_3.copy()
    for w_max in (1.0 / 3.0 + 1e-12, 0.4, 0.55, 0.8, 1.0):
        w = SCASolver(Sigma, w_max=w_max, tol=1e-8, max_iter=800).solve()
        assert w.shape == (3,)
        assert np.all(np.isfinite(w))
        assert np.isclose(w.sum(), 1.0, atol=1e-6), w_max
        assert np.all(w >= -1e-9), w_max
        assert np.all(w <= w_max + 1e-7), w_max


def test_sca_binding_upper_bound_touches_cap_on_heterogeneous_diagonal():
    """Tight w_max: at least one weight should sit on the upper bound (within tol)."""
    Sigma = np.diag([0.01, 0.64, 0.09])
    w_max = 1.0 / 3.0 + 1e-12
    w = SCASolver(Sigma, w_max=w_max, tol=1e-8, max_iter=1000).solve()
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w <= w_max + 1e-7)
    assert np.max(w) >= w_max - 1e-6


def test_sca_constructor_propagates_tol_and_max_iter_validation():
    Sigma = np.eye(2)
    with pytest.raises(ValueError, match=r"tol must be a positive finite float\."):
        SCASolver(Sigma, w_max=1.0, tol=0.0)
    with pytest.raises(ValueError, match=r"max_iter must be at least 1\."):
        SCASolver(Sigma, w_max=1.0, max_iter=0)
