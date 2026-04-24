import sys
import os
sys.path.append(os.path.abspath("."))
import numpy as np
import pytest

from riskparity._core import CCDSolver, _validate_covariance

def test_debug():
    assert True

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

def test_validate_covariance_rejects_nonpositive_diagonal():
    Sigma = np.array([[0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        _validate_covariance(Sigma)

def test_validate_covariance_rejects_nonfinite_entries():
    Sigma = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError):
        _validate_covariance(Sigma)

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