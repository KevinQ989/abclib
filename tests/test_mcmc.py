import numpy as np
import pytest
from abclib.samplers.mcmc import MCMCABC


N_SIMS = 2000
EPSILON = 0.3


def _prior_pdf(theta):
    t1, t2 = theta
    if (t1 + t2 < 1) and (t2 - t1 < 1) and (-1 < t2 < 1):
        return 1.0
    return 0.0


def test_smoke(ma2_components):
    """Does it run without error?"""
    prior, simulator, stat, distance = ma2_components
    sampler = MCMCABC(prior, simulator, stat, distance, _prior_pdf, proposal_std=0.1)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_samples=100, epsilon=EPSILON)
    assert result is not None


def test_output_shapes(ma2_components):
    """Are samples, distances, summaries the right shapes?"""
    prior, simulator, stat, distance = ma2_components
    sampler = MCMCABC(prior, simulator, stat, distance, _prior_pdf, proposal_std=0.1)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_samples=100, epsilon=EPSILON)
    n = len(result.samples)
    assert result.samples.shape == (n, 2)
    assert result.distances.shape == (n,)
    assert result.summaries.shape == (n, 2)


def test_acceptance_rate_reasonable(ma2_components):
    """Is the acceptance rate strictly between 0 and 1?"""
    prior, simulator, stat, distance = ma2_components
    sampler = MCMCABC(prior, simulator, stat, distance, _prior_pdf, proposal_std=0.1)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_samples=200, epsilon=EPSILON)
    assert 0 < result.acceptance_rate < 1


def test_all_distances_within_epsilon(ma2_components):
    """Every stored sample must have passed the epsilon gate."""
    prior, simulator, stat, distance = ma2_components
    sampler = MCMCABC(prior, simulator, stat, distance, _prior_pdf, proposal_std=0.1)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_samples=100, epsilon=EPSILON)
    assert np.all(result.distances <= EPSILON)


def test_tight_proposal_reduces_moves(ma2_components):
    """A very small proposal std should produce a chain with near-zero variance."""
    prior, simulator, stat, distance = ma2_components
    sampler = MCMCABC(prior, simulator, stat, distance, _prior_pdf, proposal_std=1e-6)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_samples=200, epsilon=EPSILON)
    assert result.samples[:, 0].std() < 0.01
    assert result.samples[:, 1].std() < 0.01


@pytest.mark.slow
def test_posterior_mean_reasonable(ma2_components):
    """Does the posterior mean recover theta* roughly?"""
    prior, simulator, stat, distance = ma2_components
    sampler = MCMCABC(prior, simulator, stat, distance, _prior_pdf, proposal_std=0.1)
    true_theta = np.array([0.6, 0.2])
    y_obs = simulator(true_theta)
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_samples=1000, epsilon=0.1)
    mean = result.posterior_mean()
    assert np.abs(mean[0] - true_theta[0]) < 0.3
    assert np.abs(mean[1] - true_theta[1]) < 0.3