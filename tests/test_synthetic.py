import numpy as np
import pytest
from abclib.synthetic_likelihood import SyntheticLikelihood


def _prior_pdf(theta):
    t1, t2 = theta
    if (t1 + t2 < 1) and (t2 - t1 < 1) and (-1 < t2 < 1):
        return 1.0
    return 0.0


def test_smoke(ma2_components):
    """Does it run without error?"""
    prior, simulator, stat, distance = ma2_components
    sl = SyntheticLikelihood(prior, simulator, stat, _prior_pdf, proposal_std=0.1)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sl.sample(s_obs, n_simulations=50, M=20)
    assert result is not None


def test_output_shapes(ma2_components):
    """Are samples, likelihoods the right shapes?"""
    prior, simulator, stat, distance = ma2_components
    sl = SyntheticLikelihood(prior, simulator, stat, _prior_pdf, proposal_std=0.1)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sl.sample(s_obs, n_simulations=50, M=20)
    assert result.samples.shape == (50, 2)
    assert result.likelihoods.shape == (50,)


def test_n_simulations_correct(ma2_components):
    """Total simulator calls should be n_simulations * M."""
    prior, simulator, stat, distance = ma2_components
    sl = SyntheticLikelihood(prior, simulator, stat, _prior_pdf, proposal_std=0.1)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sl.sample(s_obs, n_simulations=50, M=20)
    assert result.n_simulations == 50 * 20


def test_log_likelihoods_finite(ma2_components):
    """Log likelihoods should be finite for a well-initialised chain."""
    prior, simulator, stat, distance = ma2_components
    sl = SyntheticLikelihood(prior, simulator, stat, _prior_pdf, proposal_std=0.1)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sl.sample(s_obs, n_simulations=50, M=20)
    assert np.all(np.isfinite(result.log_likelihoods))


def test_larger_M_reduces_log_likelihood_variance(ma2_components):
    """Larger M should produce more stable likelihood estimates."""
    prior, simulator, stat, distance = ma2_components
    sl = SyntheticLikelihood(prior, simulator, stat, _prior_pdf, proposal_std=0.1)
    theta_fixed = np.array([0.6, 0.2])
    y_obs = simulator(theta_fixed)
    s_obs = stat.transform(y_obs)
    repeats = 30
    liks_small = [sl._log_likelihood(theta_fixed, M=5, s_obs=s_obs) for _ in range(repeats)]
    liks_large = [sl._log_likelihood(theta_fixed, M=100, s_obs=s_obs) for _ in range(repeats)]
    assert np.var(liks_large) < np.var(liks_small)


def test_unfitted_stat_raises():
    """Passing an unfitted summary statistic should raise ValueError."""
    from examples.ma2.model import prior, simulator, SUMMARY_FUNCTIONS
    from abclib.statistics.handcrafted import HandCraftedSummary
    stat = HandCraftedSummary(SUMMARY_FUNCTIONS)
    with pytest.raises(ValueError):
        SyntheticLikelihood(prior, simulator, stat, _prior_pdf, proposal_std=0.1)