import numpy as np
import matplotlib.pyplot as plt

def run_str(sampler, simulator, theta_grid, summary_statistic, credible_mass=0.90, **sampler_kwargs):
    """
    Run synthetic truth recovery.
    
    For each ground truth parameter in theta_grid, simulates data
    under it, runs the ABC sampler to obtain a posterior, and checks
    whether the true parameter falls within the credible interval.
    Repeating this across a grid spanning the prior support reveals
    region-specific recovery failures that global calibration checks
    such as SBC would average over.

    Note: this function is computationally expensive, requiring one
    full sampler call per grid point. Coverage is reported per
    parameter dimension, as a sampler may recover some parameters
    reliably while failing on others.

    Parameters
    ----------
    sampler         : BaseSampler
        An ABC sampler that can be used to sample from the posterior distribution.
    simulator       : callable
        Callable that accepts a parameter vector of shape ``(n_params,)``
        and returns simulated data in whatever form the summary statistic
        expects.
    theta_grid      : array-like
        A grid of parameter values to evaluate.
    summary_statistic : BaseSummaryStatistic
        A fitted summary statistic object exposing a ``transform`` method.
        Must be fitted (via ``fit``) on pilot simulations before being
        passed here.
    credible_mass    : float, optional
        The credible mass for the confidence interval. Default is 0.90 for a 90% interval.
    **sampler_kwargs : dict
        Additional keyword arguments to pass to the sampler's ``sample`` method.

    Returns
    -------
    dict with keys:
        "theta_grid" : np.ndarray, shape (n_grid, n_params)
            The grid of parameter values evaluated.
        "covered"    : np.ndarray, shape (n_grid, n_params)
            Boolean array indicating whether the true parameter value was covered by the credible interval.
        "coverage"   : np.ndarray, shape (n_params,)
            The mean coverage for each parameter across the grid.
        "lower"      : np.ndarray, shape (n_grid, n_params)
            The lower bounds of the credible intervals for each parameter.
        "upper"      : np.ndarray, shape (n_grid, n_params)
            The upper bounds of the credible intervals for each parameter.
    """
    n_params = theta_grid.shape[1] if theta_grid.ndim > 1 else 1
    covered = np.zeros((len(theta_grid), n_params), dtype=bool)
    lower = np.zeros((len(theta_grid), n_params), dtype=float)
    upper = np.zeros((len(theta_grid), n_params), dtype=float)
    
    for i, theta in enumerate(theta_grid):
        y_sim = simulator(theta)
        s_obs = summary_statistic.transform(y_sim)
        result = sampler.sample(s_obs, **sampler_kwargs)
        lower[i], upper[i] = result.credible_interval(credible_mass)
        covered[i] = (theta >= lower[i]) & (theta <= upper[i])
    
    return {
        "theta_grid": theta_grid,
        "covered": covered,
        "coverage": np.mean(covered, axis=0),
        "lower": lower,
        "upper": upper
    }


def plot_str_results(str_results):
    """
    Plot the results of synthetic truth recovery.

    Parameters
    ----------
    str_results : dict
        Output from the ``run_str`` function containing "theta_grid", "covered", "lower", and "upper".
    """
    n_params = str_results["theta_grid"].shape[1] if str_results["theta_grid"].ndim > 1 else 1
    for j in range(n_params):
        plt.figure(figsize=(8, 5))
        for i, theta_star in enumerate(str_results["theta_grid"][:, j]):
            color = "green" if str_results["covered"][i, j] else "red"
            plt.vlines(theta_star, str_results["lower"][i, j], 
                    str_results["upper"][i, j], color=color, alpha=0.5)
            plt.scatter(theta_star, theta_star, color="black", s=10, zorder=5)
        plt.plot([], [], color="green", label="Covered")
        plt.plot([], [], color="red", label="Not covered")
        plt.xlabel(f"True $\\theta_{j}$")
        plt.ylabel(f"Credible interval for $\\theta_{j}$")
        plt.title(f"STR — Parameter {j} | Coverage: {str_results['coverage'][j]:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.show()