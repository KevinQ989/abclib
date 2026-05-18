import numpy as np
import pytest
from abclib.samplers.rejection import RejectionABC
from abclib.diagnostics import run_sbc, run_ppc, run_str

# --- PPC tests ---

def test_ppc_smoke(ma2_components):
    """Does it run without error?"""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=2000, q=0.05)
    ppc = run_ppc(result, simulator, y_obs, np.mean, n_samples=100)
    assert "t_rep" in ppc
    assert "t_obs" in ppc
    assert "p_value" in ppc


def test_ppc_p_value_bounds(ma2_components):
    """p_value must always be in [0, 1]."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=2000, q=0.05)
    ppc = run_ppc(result, simulator, y_obs, np.mean, n_samples=200)
    assert 0.0 <= ppc["p_value"] <= 1.0


def test_ppc_t_rep_shape(ma2_components):
    """t_rep should have shape (n_samples,)."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=2000, q=0.05)
    ppc = run_ppc(result, simulator, y_obs, np.mean, n_samples=150)
    assert ppc["t_rep"].shape == (150,)


def test_ppc_extreme_statistic(ma2_components):
    """A statistic the model cannot reproduce should give extreme p-value."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    y_obs = simulator(np.array([0.6, 0.2]))
    s_obs = stat.transform(y_obs)
    result = sampler.sample(s_obs, n_simulations=2000, q=0.05)
    # constant statistic much larger than any simulation will produce
    extreme_stat = lambda y: 1e10 if np.array_equal(y, y_obs) else 0.0
    ppc = run_ppc(result, simulator, y_obs, extreme_stat, n_samples=100)
    assert ppc["p_value"] == 0.0


# --- STR tests ---

def test_str_smoke(ma2_components):
    """Does it run without error?"""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    theta_grid = np.array([[0.3, 0.1], [0.6, 0.2], [-0.3, 0.1]])
    result = run_str(
        sampler, simulator, theta_grid, stat,
        credible_mass=0.90, n_simulations=1000, q=0.05
    )
    assert "covered" in result
    assert "coverage" in result


def test_str_output_shapes(ma2_components):
    """Check shapes of covered, lower, upper, and coverage outputs."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    theta_grid = np.array([[0.3, 0.1], [0.6, 0.2], [-0.3, 0.1]])
    result = run_str(
        sampler, simulator, theta_grid, stat,
        credible_mass=0.90, n_simulations=1000, q=0.05
    )
    assert result["covered"].shape == (3, 2)
    assert result["lower"].shape == (3, 2)
    assert result["upper"].shape == (3, 2)
    assert result["coverage"].shape == (2,)


def test_str_coverage_bounds(ma2_components):
    """Coverage must be in [0, 1] for each parameter."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    theta_grid = np.array([[0.3, 0.1], [0.6, 0.2], [-0.3, 0.1]])
    result = run_str(
        sampler, simulator, theta_grid, stat,
        credible_mass=0.90, n_simulations=1000, q=0.05
    )
    assert np.all(result["coverage"] >= 0.0)
    assert np.all(result["coverage"] <= 1.0)


# --- SBC tests ---

@pytest.mark.slow
def test_sbc_smoke(ma2_components):
    """Does it run without error?"""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    result = run_sbc(
        sampler, simulator, prior,
        n_trials=20, L=50, summary_statistic=stat,
        n_simulations=1000, q=0.1
    )
    assert "ranks" in result
    assert "ks_stat" in result
    assert "ks_pvalue" in result


@pytest.mark.slow
def test_sbc_rank_shapes(ma2_components):
    """Ranks should have shape (n_trials, n_params)."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    result = run_sbc(
        sampler, simulator, prior,
        n_trials=20, L=50, summary_statistic=stat,
        n_simulations=1000, q=0.1
    )
    assert result["ranks"].shape == (20, 2)
    assert result["thetas"].shape == (20, 2)
    assert result["ks_stat"].shape == (2,)
    assert result["ks_pvalue"].shape == (2,)


@pytest.mark.slow
def test_sbc_ranks_within_range(ma2_components):
    """All ranks must be integers in {0, ..., L}."""
    prior, simulator, stat, distance = ma2_components
    sampler = RejectionABC(prior, simulator, stat, distance)
    L = 50
    result = run_sbc(
        sampler, simulator, prior,
        n_trials=20, L=L, summary_statistic=stat,
        n_simulations=1000, q=0.1
    )
    assert np.all(result["ranks"] >= 0)
    assert np.all(result["ranks"] <= L)