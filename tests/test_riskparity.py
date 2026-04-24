import sys
import os

sys.path.append(os.path.abspath("."))

import numpy as np
import pytest

from riskparity._core import (
    CCDSolver, 
    _validate_covariance,
    risk_contributions,
    relative_risk_contributions,
    risk_contribution_gap,
)

# Covariance validation tests
def test_validate_covariance_accepts_valid_matrix():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    out = _validate_covariance(Sigma)
    assert out.shape == (2, 2)
    assert np.allclose(out, Sigma)

def test_validate_covariance_rejects_non_square_matrix():
    Sigma = np.ones((2, 3))
    with pytest.raises(ValueError):
        _validate_covariance(Sigma)

def test_validate_covariance_rejects_non_symmetric_matrix():
    Sigma = np.array([[1.0, 0.2], [0.1, 1.0]])
    with pytest.raises(ValueError):
        _validate_covariance(Sigma)

# Edge case tests for covariance validation
def test_validate_covariance_rejects_nonpositive_diagonal():
    Sigma = np.array([[0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        _validate_covariance(Sigma)

def test_validate_covariance_rejects_nonfinite_entries():
    Sigma = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError):
        _validate_covariance(Sigma)

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