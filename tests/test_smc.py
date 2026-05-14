import numpy as np
import pytest
from abclib.samplers.smc import SMCABC


def _prior_density(thetas):
    """Batch prior density for SMC. Returns 1 inside invertibility region, 0 outside."""
    densities = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        t1, t2 = theta
        if (t1 + t2 < 1) and (t2 - t1 < 1) and (-1 < t2 < 1):
            densities[i] = 1.0
    return densities


def test_smoke(ma2_components):
    """Does it run without error?"""
    prior, simulator, stat, distance = ma2_components
    sampler = SMCABC(prior, simulator, stat, distance, _prior_density)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, M=100, T=3, q=0.5)
    assert result is not None


def test_output_shapes(ma2_components):
    """Are samples, distances, summaries the right shapes?"""
    prior, simulator, stat, distance = ma2_components
    sampler = SMCABC(prior, simulator, stat, distance, _prior_density)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, M=100, T=3, q=0.5)
    assert result.samples.shape == (100, 2)
    assert result.distances.shape == (100,)
    assert result.summaries.shape == (100, 2)


def test_tolerance_sequence_decreasing(ma2_components):
    """Tolerances should decrease monotonically across SMC stages."""
    prior, simulator, stat, distance = ma2_components
    sampler = SMCABC(prior, simulator, stat, distance, _prior_density)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, M=100, T=5, q=0.5)
    assert np.all(np.diff(result.epsilons) < 0)


def test_acceptance_rate_reasonable(ma2_components):
    """Is the acceptance rate strictly between 0 and 1?"""
    prior, simulator, stat, distance = ma2_components
    sampler = SMCABC(prior, simulator, stat, distance, _prior_density)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, M=100, T=3, q=0.5)
    assert 0 < result.acceptance_rate < 1


@pytest.mark.slow
def test_posterior_mean_reasonable(ma2_components):
    """Does the posterior mean recover theta* roughly?"""
    prior, simulator, stat, distance = ma2_components
    sampler = SMCABC(prior, simulator, stat, distance, _prior_density)
    true_theta = np.array([0.6, 0.2])
    y_obs = simulator(true_theta)
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, M=500, T=6, q=0.3)
    mean = result.posterior_mean()
    assert np.abs(mean[0] - true_theta[0]) < 0.3
    assert np.abs(mean[1] - true_theta[1]) < 0.3