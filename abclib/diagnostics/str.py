import os
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


def plot_str_results(str_results, output_dir=None,             
        filename="str_results.png", parameter_names=None):
    """
    Plot the results of synthetic truth recovery.

    Parameters
    ----------
    str_results : dict
        Output from the ``run_str`` function containing "theta_grid", "covered", "lower", and "upper".
    output_dir : str, optional
        Saves the plot to this directory if provided. Default is current directory.
    filename   : str, optional
        The filename to use when saving the plot. Default is "str_results.png".
    parameter_names : list of str, optional
        Names of the parameters to use in plot labels. If None, parameters will be labeled by index.
    """
    theta_grid = str_results["theta_grid"]
    n_params = theta_grid.shape[1] if theta_grid.ndim > 1 else 1

    if parameter_names is None:
        parameter_names = [f"Parameter {j}" for j in range(n_params)]

    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = np.array(axes).flatten()

    for j in range(n_params):
        ax = axes[j]
        for i, theta_star in enumerate(theta_grid[:, j]):
            color = "green" if str_results["covered"][i, j] else "red"
            ax.vlines(
                theta_star,
                str_results["lower"][i, j],
                str_results["upper"][i, j],
                color=color, alpha=0.5
            )
            ax.scatter(theta_star, theta_star, color="black", s=10, zorder=5)
        ax.plot([], [], color="green", label="Covered")
        ax.plot([], [], color="red", label="Not covered")
        ax.set_xlabel(f"True ${parameter_names[j]}$")
        ax.set_ylabel(f"Credible interval for ${parameter_names[j]}$")
        ax.set_title(
            f"STR — {parameter_names[j]} | "
            f"Coverage: {str_results['coverage'][j]:.2f}",
            fontsize=10
        )
        ax.legend(fontsize=8)

    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir or ".", filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved STR plot to {path}")