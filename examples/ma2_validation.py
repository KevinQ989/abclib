from .ma2_model import MA2
from .validation import run_validation
from .config import Config, RejectionConfig, SMCConfig, MCMCConfig, SyntheticLikelihoodConfig
import numpy as np
import scipy.stats as stats
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def exact_posterior_grid(model, y, n_grid=200):
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
                Sigma = model.autocovariance(theta)
                log_prob[i, j] = stats.multivariate_normal.logpdf(
                    y, mean=np.zeros(len(y)), cov=Sigma
                )

    log_prob -= np.max(log_prob)
    prob = np.exp(log_prob)
    prob /= np.sum(prob)
    return T1, T2, prob


def main():
    model = MA2(T=100)
    true_theta = np.array([0.6, 0.2])
    observed_data = model.simulator(true_theta)

    config = Config(
        methods = ["rejection_sa", "smc", "mcmc", "synthetic_likelihood"],
        n_pilot = 2_000,
        seed = 0,
        output_dir = OUTPUT_DIR,
        rejection = RejectionConfig(
            n_simulations=10_000, q=0.05
        ),
        smc = SMCConfig(
            M=10_000, T=5, q=0.05
        ),
        mcmc = MCMCConfig(
            n_samples=10_000, epsilon=0.05, proposal_std=0.3
        ),
        synthetic_likelihood=SyntheticLikelihoodConfig(
            n_simulations=10_000, M=100, proposal_std=0.1
        ),
    )

    exact_grid = exact_posterior_grid(model, observed_data, n_grid=200)
    validation_result = run_validation(
        model = model,
        config = config,
        true_theta = true_theta,
        observed_data = observed_data,
        exact_grid = exact_grid
    )


if __name__ == "__main__":
    main()