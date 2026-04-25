import sys
import os

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

# Covariance validation tests (aligned with riskparity._core._validate_covariance)
def test_validate_covariance_accepts_valid_matrix():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
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
    """np.allclose(Sigma, Sigma.T, atol=1e-10) should accept tiny off-diagonal asymmetry."""
    Sigma = np.array([[1.0, 0.0], [1e-11, 1.0]])
    out = _validate_covariance(Sigma)
    assert out.shape == (2, 2)
    assert np.allclose(out, out.T, atol=1e-10)


def test_validate_covariance_rejects_asymmetric_beyond_atol():
    Sigma = np.array([[1.0, 1e-9], [0.0, 1.0]])
    with pytest.raises(ValueError, match=r"Sigma must be symmetric\."):
        _validate_covariance(Sigma)


# Edge case tests for covariance validation
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
    with pytest.raises(ValueError):
        _validate_tol(0.0)
    with pytest.raises(ValueError):
        _validate_tol(-1e-6)


def test_validate_max_iter_rejects_nonpositive():
    with pytest.raises(ValueError):
        _validate_max_iter(0)

# CCD solver basic correctness tests
def test_ccd_returns_valid_weights():
    Sigma = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ]
    )
    w = CCDSolver(Sigma).solve()
    assert w.shape == (3,)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-10)
    assert np.all(w > 0.0)

# Analytical validation (diagonal covariance cases)
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

# Risk contribution validation tests
def test_risk_contributions_sum_to_portfolio_variance():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.6, 0.4])
    rc = risk_contributions(Sigma, w)
    portfolio_variance = w @ Sigma @ w
    assert np.isclose(rc.sum(), portfolio_variance)

def test_relative_risk_contributions_sum_to_one():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.6, 0.4])
    rrc = relative_risk_contributions(Sigma, w)
    assert np.isclose(rrc.sum(), 1.0)
    assert np.all(rrc >= 0.0)

def test_ccd_risk_contributions_are_close_to_equal():
    Sigma = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ]
    )
    w = CCDSolver(Sigma, tol=1e-10, max_iter=2000).solve()
    rc = risk_contributions(Sigma, w)
    assert np.allclose(rc, np.full(3, rc.mean()), atol=1e-6)

def test_ccd_risk_contribution_gap_is_small():
    Sigma = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ]
    )
    w = CCDSolver(Sigma, tol=1e-10, max_iter=2000).solve()
    gap = risk_contribution_gap(Sigma, w)
    assert gap < 1e-6


def test_risk_contributions_rejects_weight_length_mismatch():
    Sigma = np.eye(3)
    w = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        risk_contributions(Sigma, w)


# Scale and numerical-stability checks (larger n, stiff correlations)
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


# --- Robustness (Person 3): constrained solver — box & simplex constraints ---


def test_sca_returns_feasible_weights():
    Sigma = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ]
    )
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
    with pytest.raises(ValueError, match=r"w_max=0\.15 is infeasible for n=5"):
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
    with pytest.raises(ValueError, match=rf"w_max={w_max} is infeasible for n={n}"):
        SCASolver(Sigma, w_max=w_max)


def test_sca_accepts_w_max_at_inverse_n_boundary():
    """Feasibility uses strict inequality w_max * n < 1 - 1e-12; equality w_max = 1/n is OK."""
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
    Sigma = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ]
    )
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


def test_sca_matches_ccd_when_w_max_is_non_binding():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w_ccd = CCDSolver(Sigma, tol=1e-10, max_iter=2000).solve()
    w_sca = SCASolver(Sigma, w_max=1.0, tol=1e-10, max_iter=500).solve()
    assert np.allclose(w_sca, w_ccd, atol=1e-4)