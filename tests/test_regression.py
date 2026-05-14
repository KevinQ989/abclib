import numpy as np
import pytest
from abclib.samplers.rejection import RejectionABC
from abclib.postprocessing.regression import RegressionAdjustment


PRIOR_BOUNDS = [(-1, 1), (-1, 1)]


def test_smoke(ma2_components):
    """Does it run without error?"""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=2000, q=0.05)
    adj = RegressionAdjustment(PRIOR_BOUNDS)
    adj.fit(result, s_obs)
    adjusted = adj.adjust(result, s_obs)
    assert adjusted is not None


def test_shape_preserved(ma2_components):
    """Adjustment should not change the number of samples."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=2000, q=0.05)
    adj = RegressionAdjustment(PRIOR_BOUNDS)
    adj.fit(result, s_obs)
    adjusted = adj.adjust(result, s_obs)
    assert len(adjusted.samples) == len(result.samples)


def test_adjusted_samples_within_bounds(ma2_components):
    """All adjusted samples should lie within prior bounds after reflection."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=2000, q=0.05)
    adj = RegressionAdjustment(PRIOR_BOUNDS)
    adj.fit(result, s_obs)
    adjusted = adj.adjust(result, s_obs)
    for j, (lo, hi) in enumerate(PRIOR_BOUNDS):
        assert np.all(adjusted.samples[:, j] >= lo)
        assert np.all(adjusted.samples[:, j] <= hi)


def test_adjust_called_before_fit_raises(ma2_components):
    """Calling adjust before fit should raise a RuntimeError."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=2000, q=0.05)
    adj = RegressionAdjustment(PRIOR_BOUNDS)
    with pytest.raises(RuntimeError):
        adj.adjust(result, s_obs)