from .ma2_model import prior, prior_pdf, prior_density, simulator, SUMMARY_FUNCTIONS, H_FUNCTION, autocovariance
from abclib.utils import run_pilot
from abclib.distance import euclidean
from abclib.results import ABCResult, SLResult
import os
import abclib
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    plt.savefig(os.path.join(OUTPUT_DIR, "posterior_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved posterior_comparison.png")


def print_result(results_dict):
    """
    Print a summary table of posterior means, 90% credible intervals,
    and acceptance rates for each method.

    Parameters
    ----------
    results_dict : dict[str, ABCResult / SLResult]
        Mapping of method names to ABCResult or SLResult objects.
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
    np.random.seed(0)
    true_theta = np.array([0.6, 0.2])

    ##############################################
    # Simulate observed data and pilot data
    ##############################################
    print("Simulating observed data and running pilot...")
    observed_data = simulator(true_theta)
    pilot_thetas, pilot_sims = run_pilot(prior, simulator, n_pilot=2000)
    print(f"  Pilot complete: {len(pilot_thetas)} draws.")

    #############################################
    # Fit summary statistics on pilot data
    #############################################
    print("\nFitting summary statistics...")
    handcrafted_summary = abclib.HandCraftedSummary(SUMMARY_FUNCTIONS)
    handcrafted_summary.fit(pilot_thetas, pilot_sims)
    s_obs_handcrafted = handcrafted_summary.transform(observed_data)
    print("  HandCraftedSummary fitted.")

    semi_automatic_summary = abclib.SemiAutomaticSummary(H_FUNCTION)
    semi_automatic_summary.fit(pilot_thetas, pilot_sims)
    s_obs_semiauto = semi_automatic_summary.transform(observed_data)
    print("  SemiAutomaticSummary fitted.")

    ############################################
    # Handcrafted vs. Semi-Automatic summaries
    ############################################
    print("\nRunning Rejection ABC (HandCrafted)...")
    handcrafted_rejection_abc : ABCResult = abclib.RejectionABC(
        prior=prior, simulator=simulator,
        summary_statistic=handcrafted_summary, distance=euclidean
    ).sample(s_obs_handcrafted, n_simulations=10_000, q=0.05)
    print(f"  Done. Accepted {len(handcrafted_rejection_abc.samples)} samples "
          f"(acc={handcrafted_rejection_abc.acceptance_rate:.4f}).")

    print("\nRunning Rejection ABC (Semi-Automatic)...")
    semi_automatic_rejection_abc : ABCResult = abclib.RejectionABC(
        prior=prior, simulator=simulator,
        summary_statistic=semi_automatic_summary, distance=euclidean
    ).sample(s_obs_semiauto, n_simulations=10_000, q=0.05)
    print(f"  Done. Accepted {len(semi_automatic_rejection_abc.samples)} samples "
          f"(acc={semi_automatic_rejection_abc.acceptance_rate:.4f}).")

    ############################################
    # Rejection vs. SMC vs. MCMC ABC
    ############################################
    print("\nRunning SMC-ABC...")
    handcrafted_smc_abc : ABCResult = abclib.SMCABC(
        prior=prior, simulator=simulator,
        summary_statistic=handcrafted_summary, distance=euclidean,
        prior_density=prior_density
    ).sample(s_obs_handcrafted, M=10_000, T=5, q=0.05)
    print(f"  Done. Final epsilon={handcrafted_smc_abc.epsilon:.4f} "
          f"(acc={handcrafted_smc_abc.acceptance_rate:.4f}).")

    print("\nRunning MCMC-ABC...")
    handcrafted_mcmc_abc : ABCResult = abclib.MCMCABC(
        prior=prior, simulator=simulator,
        summary_statistic=handcrafted_summary, distance=euclidean,
        prior_pdf=prior_pdf, proposal_std=0.1
    ).sample(s_obs_handcrafted, n_samples=10_000, epsilon=0.05)
    print(f"  Done. {len(handcrafted_mcmc_abc.samples)} chain states "
          f"(acc={handcrafted_mcmc_abc.acceptance_rate:.4f}).")

    ############################################
    # Regression Adjustment
    ############################################
    print("\nRunning Regression Adjustment...")
    prior_bounds = [(-1, 1), (-1, 1)]
    reg_adj = abclib.RegressionAdjustment(prior_bounds)
    reg_adj.fit(handcrafted_rejection_abc, s_obs_handcrafted)
    regression_adjustment : ABCResult = reg_adj.adjust(
        result=handcrafted_rejection_abc, s_obs=s_obs_handcrafted
    )
    print(f"  Done. Adjusted {len(regression_adjustment.samples)} samples.")

    ############################################
    # Synthetic Likelihood
    ############################################
    print("\nRunning Synthetic Likelihood (this may take a while)...")
    synthetic_likelihood_result : SLResult = abclib.SyntheticLikelihood(
        prior=prior, simulator=simulator,
        summary_statistic=handcrafted_summary,
        prior_pdf=prior_pdf, proposal_std=0.1
    ).sample(s_obs_handcrafted, n_simulations=10_000, M=100)
    print(f"  Done. {len(synthetic_likelihood_result.samples)} samples "
          f"(acc={synthetic_likelihood_result.acceptance_rate:.4f}). "
          f"Total simulator calls: {synthetic_likelihood_result.n_simulations:,}.")

    ############################################
    # Exact posterior and results
    ############################################
    print("\nComputing exact posterior grid (this may take a while)...")
    exact_grid = exact_posterior_grid(observed_data, n_grid=200)
    print("  Done.")

    results_dict = {
        "Rejection ABC (HC)":   handcrafted_rejection_abc,
        "Rejection ABC (SA)":   semi_automatic_rejection_abc,
        "SMC-ABC":              handcrafted_smc_abc,
        "MCMC-ABC":             handcrafted_mcmc_abc,
        "Regression Adj.":      regression_adjustment,
        "Synthetic Likelihood": synthetic_likelihood_result
    }

    print("\n--- Results Summary ---")
    print_result(results_dict)
    plot_results(results_dict, exact_grid, true_theta=true_theta)

    ###############################################
    # Diagnostics
    ###############################################
    print("\nRunning PPC (mean)...")
    ppc_mean = abclib.run_ppc(
        result=regression_adjustment, simulator=simulator,
        y_obs=observed_data, test_statistic=np.mean, n_samples=1000
    )
    print(f"  p-value: {ppc_mean['p_value']:.3f} | "
          f"t_obs: {ppc_mean['t_obs']:.4f} | "
          f"t_rep mean: {np.mean(ppc_mean['t_rep']):.4f}")

    print("\nRunning PPC (variance)...")
    ppc_var = abclib.run_ppc(
        result=regression_adjustment, simulator=simulator,
        y_obs=observed_data, test_statistic=np.var, n_samples=1000
    )
    print(f"  p-value: {ppc_var['p_value']:.3f} | "
          f"t_obs: {ppc_var['t_obs']:.4f} | "
          f"t_rep mean: {np.mean(ppc_var['t_rep']):.4f}")

    print("\nRunning SBC (100 trials, this will be slow)...")
    sbc_result = abclib.run_sbc(
        sampler=abclib.RejectionABC(
            prior=prior, simulator=simulator,
            summary_statistic=handcrafted_summary, distance=euclidean
        ),
        simulator=simulator, prior=prior,
        n_trials=100, L=100,
        summary_statistic=handcrafted_summary,
        n_simulations=2000, q=0.05
    )
    print(f"  Done. KS p-values per parameter: {sbc_result['ks_pvalue'].round(3)}")
    abclib.plot_rank_histogram(sbc_result, n_bins=20, output_dir=OUTPUT_DIR)

    print("\nRunning STR (3 grid points)...")
    str_result = abclib.run_str(
        sampler=abclib.RejectionABC(
            prior=prior, simulator=simulator,
            summary_statistic=handcrafted_summary, distance=euclidean
        ),
        simulator=simulator,
        theta_grid=np.array([[0.6, 0.2], [0.5, 0.1], [0.7, 0.3]]),
        summary_statistic=handcrafted_summary,
        credible_mass=0.90, n_simulations=2000, q=0.05
    )
    print(f"  Done. Coverage per parameter: {str_result['coverage'].round(3)}")
    abclib.plot_str_results(str_result, output_dir=OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
