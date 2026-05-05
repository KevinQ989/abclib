from .ma2_model import prior, simulator, SUMMARY_FUNCTIONS, H_FUNCTION, autocovariance
from abclib.utils import run_pilot
from abclib.distance import euclidean
from abclib.results import ABCResult
import abclib
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def exact_posterior_grid(y, n_grid=200):
    """
    Evaluate the exact MA(2) posterior on a grid over the invertibility region.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Observed time series.
    n_grid : int, optional
        Number of grid points per axis. Default is 200.

    Returns
    -------
    T1 : np.ndarray, shape (n_grid, n_grid)
        Grid of theta1 values.
    T2 : np.ndarray, shape (n_grid, n_grid)
        Grid of theta2 values.
    prob : np.ndarray, shape (n_grid, n_grid)
        Normalised posterior density on the grid.
    """
    n1 = np.linspace(-1, 1, n_grid)
    n2 = np.linspace(-1, 1, n_grid)
    T1, T2 = np.meshgrid(n1, n2)
    log_prob = np.full((n_grid, n_grid), -np.inf)

    for i in range(n_grid):
        for j in range(n_grid):
            theta = np.array([T1[i, j], T2[i, j]])
            if (theta[0] + theta[1] < 1
                    and theta[1] - theta[0] < 1
                    and abs(theta[1]) < 1):
                Sigma = autocovariance(theta, T=len(y))
                log_prob[i, j] = stats.multivariate_normal.logpdf(
                    y, mean=np.zeros(len(y)), cov=Sigma
                )

    log_prob -= np.max(log_prob)
    prob = np.exp(log_prob)
    prob /= np.sum(prob)
    return T1, T2, prob


def plot_results(results_dict, exact_grid, true_theta):
    """
    Plot ABC posterior samples against the exact posterior for each method.

    Parameters
    ----------
    results_dict : dict[str, ABCResult]
        Mapping of method names to ABCResult objects.
    exact_grid : tuple
        (T1, T2, prob) returned by exact_posterior_grid.
    true_theta : np.ndarray, shape (2,)
        True parameter vector to mark on each plot.
    """
    T1, T2, prob = exact_grid
    n_methods = len(results_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))

    if n_methods == 1:
        axes = [axes]

    for ax, (label, result) in zip(axes, results_dict.items()):
        ax.contour(T1, T2, prob, levels=10, cmap='Blues')
        ax.scatter(
            result.samples[:, 0], result.samples[:, 1],
            alpha=0.2, s=5, color='orange', label='ABC samples'
        )
        ax.scatter(
            true_theta[0], true_theta[1],
            color='red', marker='+', s=150, zorder=5, label='True θ'
        )
        ax.set_title(f"{label}\nacc={result.acceptance_rate:.4f}")
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


def print_summary(results_dict):
    """
    Print a summary table of posterior means, 90% credible intervals,
    and acceptance rates for each method.

    Parameters
    ----------
    results_dict : dict[str, ABCResult]
        Mapping of method names to ABCResult objects.
    """
    header = f"{'Method':<30} {'Mean':>20} {'90% CI Lower':>20} {'90% CI Upper':>20} {'Acc. Rate':>12}"
    print(header)
    print("-" * len(header))

    for method, result in results_dict.items():
        mean = result.posterior_mean()
        lower, upper = result.credible_interval(alpha=0.90)
        print(
            f"{method:<30} "
            f"{np.array2string(mean, precision=3):>20} "
            f"{np.array2string(lower, precision=3):>20} "
            f"{np.array2string(upper, precision=3):>20} "
            f"{result.acceptance_rate:>12.4f}"
        )


def main():
    np.random.seed(0)  # For reproducibility
    true_theta = np.array([0.6, 0.2])
                
    ##############################################
    # Simulate observed data and pilot data
    ##############################################
    observed_data = simulator(true_theta)
    pilot_thetas, pilot_sims = run_pilot(prior, simulator, n_pilot=2000)

    #############################################
    # Fit summary statistics on pilot data
    #############################################
    handcrafted_summary = abclib.HandCraftedSummary(SUMMARY_FUNCTIONS)
    handcrafted_summary.fit(pilot_thetas, pilot_sims)
    s_obs_handcrafted = handcrafted_summary.transform(observed_data)

    semi_automatic_summary = abclib.SemiAutomaticSummary(H_FUNCTION)
    semi_automatic_summary.fit(pilot_thetas, pilot_sims)
    s_obs_semiauto = semi_automatic_summary.transform(observed_data)

    ############################################
    # Handcrafted vs. Semi-Automatic summaries
    ############################################
    handcrafted_rejection_abc : ABCResult = abclib.RejectionABC(
        prior = prior,
        simulator = simulator,
        summary_statistic = handcrafted_summary,
        distance = euclidean
    ).sample(s_obs_handcrafted, n_simulations=10_000, q=0.05)

    semi_automatic_rejection_abc : ABCResult = abclib.RejectionABC(
        prior = prior,
        simulator = simulator,
        summary_statistic = semi_automatic_summary,
        distance = euclidean
    ).sample(s_obs_semiauto, n_simulations=10_000, q=0.05)

    ############################################
    # Rejection vs. SMC vs. MCMC ABC
    ############################################
    # handcrafted_smc_abc : ABCResult = abclib.SMCABC(
    #     prior = prior,
    #     simulator = simulator,
    #     summary_statistic = handcrafted_summary,
    #     distance = euclidean
    # ).sample(s_obs_handcrafted, n_simulations=10_000, q=0.05)

    # handcrafted_mcmc_abc : ABCResult = abclib.MCMCABC(
    #     prior = prior,
    #     simulator = simulator,
    #     summary_statistic = handcrafted_summary,
    #     distance = euclidean
    # ).sample(s_obs_handcrafted, n_simulations=10_000, q=0.05)

    ############################################
    # Regression Adjustment
    ############################################

    ############################################
    # Synthetic Likelihood
    ############################################

    ############################################
    # Exact posterior and results
    ############################################
    exact_grid = exact_posterior_grid(observed_data, n_grid=200)

    results_dict = {
        "Rejection ABC (HC)":   handcrafted_rejection_abc,
        "Rejection ABC (SA)":   semi_automatic_rejection_abc,
        # "SMC-ABC":            handcrafted_smc_abc,
        # "MCMC-ABC":           handcrafted_mcmc_abc,
        # "Regression Adj.":    regression_adjustment_result,
        # "Synthetic Likelihood": synthetic_likelihood_result
    }

    print_summary(results_dict)
    plot_results(results_dict, exact_grid, true_theta=true_theta)

if __name__ == "__main__":
    main()
