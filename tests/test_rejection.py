import numpy as np
import pytest
from abclib.samplers.rejection import RejectionABC


EPSILON = 0.3
N_SIMS = 2000


def test_smoke(ma2_components):
    """Does it run without error?"""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=N_SIMS, q=0.05)
    assert result is not None


def test_output_shapes(ma2_components):
    """Are samples, distances, summaries the right shapes?"""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=N_SIMS, q=0.05)
    n = len(result.samples)
    assert result.samples.shape == (n, 2)
    assert result.distances.shape == (n,)
    assert result.summaries.shape == (n, 2)


def test_epsilon_respected(ma2_components):
    """Are all returned distances <= epsilon?"""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=N_SIMS, q=0.05)
    assert np.all(result.distances <= result.epsilon)


def test_acceptance_rate_reasonable(ma2_components):
    """Is the acceptance rate strictly between 0 and 1?"""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=N_SIMS, q=0.05)
    assert 0 < result.acceptance_rate < 1


def test_posterior_mean_reasonable(ma2_components):
    """Does the posterior mean recover theta* roughly?"""
    prior, simulator, stat, distance = ma2_components
    true_theta = np.array([0.6, 0.2])
    y_obs = simulator(true_theta)
    s_obs = stat.transform(y_obs)
    sampler = RejectionABC(prior, simulator, stat, distance)
    result = sampler.sample(s_obs, n_simulations=5000, q=0.01)
    mean = result.posterior_mean()
    assert np.abs(mean[0] - true_theta[0]) < 0.3
    assert np.abs(mean[1] - true_theta[1]) < 0.3