import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def run_sbc(sampler, simulator, prior, n_trials, L, summary_statistic, **sampler_kwargs):
    """
    Run simulation-based calibration.

    For each trial, draws a ground truth parameter from the prior,
    simulates data under it, runs the ABC sampler to obtain L posterior
    draws, and computes the rank of the true parameter within those
    draws. Under a correctly calibrated posterior, ranks are uniform
    on {0, ..., L} for each parameter dimension independently.

    Note: this function is computationally expensive, requiring
    n_trials full sampler calls. Each call runs the sampler to
    completion with the provided sampler_kwargs.

    Parameters
    ----------
    sampler           : BaseSampler
        A fitted, ready-to-call sampler instance. The sampler's
        summary_statistic must already be fitted before passing here.
    simulator         : callable
        Callable accepting a parameter vector of shape ``(n_params,)``
        and returning simulated data.
    prior             : callable
        Callable with no arguments returning a single draw from the
        prior as a 1D ``np.ndarray`` of shape ``(n_params,)``.
    n_trials          : int
        Number of (theta*, y*) pairs to generate. Larger values give
        a more reliable rank histogram but require more compute.
    L                 : int
        Number of posterior draws to obtain per trial. The rank of
        theta* is computed within these L draws, so L controls the
        resolution of the rank histogram.
    summary_statistic : BaseSummaryStatistic
        Fitted summary statistic object used to compute s_obs from
        each simulated dataset.
    **sampler_kwargs  : dict
        Additional keyword arguments passed to sampler.sample, such
        as epsilon and n_simulations.

    Returns
    -------
    dict with keys:
        "ranks"     : np.ndarray, shape (n_trials, n_params)
            Rank of theta* within the L posterior draws for each
            parameter dimension. Each entry is an integer in
            {0, ..., L}.
        "thetas"    : np.ndarray, shape (n_trials, n_params)
            Ground truth parameter vectors used in each trial.
        "ks_stat"   : np.ndarray, shape (n_params,)
            KS test statistic against the uniform distribution,
            one value per parameter dimension.
        "ks_pvalue" : np.ndarray, shape (n_params,)
            KS test p-value, one value per parameter dimension.
            Values below 0.05 provide evidence against calibration.

    Raises
    ------
    ValueError
        If the sampler returns fewer than L posterior samples in any trial.
    """
    n_params = len(prior())
    ranks = np.zeros((n_trials, n_params), dtype=int)
    thetas = np.zeros((n_trials, n_params), dtype=float)
    
    for i in range(n_trials):
        theta_star = prior()
        y_star = simulator(theta_star)
        s_obs = summary_statistic.transform(y_star)
        result = sampler.sample(s_obs, **sampler_kwargs)
        posterior_samples = result.samples
        if len(posterior_samples) < L:
            raise ValueError(
                f"Sampler returned {len(posterior_samples)} samples but L={L} required. "
                f"Increase n_simulations in sampler_kwargs or reduce L."
            )
        if len(posterior_samples) > L:
            idx = np.random.choice(len(posterior_samples), size=L, replace=False)
            posterior_samples = posterior_samples[idx]
        for j in range(n_params):
            ranks[i, j] = np.sum(posterior_samples[:, j] < theta_star[j])
        thetas[i] = theta_star
    
    ks_stat = np.zeros(n_params)
    ks_pvalue = np.zeros(n_params)
    for j in range(n_params):
        ks_stat[j], ks_pvalue[j] = stats.kstest(ranks[:, j], stats.randint(0, L + 1).cdf)
    
    return {
        "ranks": ranks,
        "thetas": thetas,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue
    }


def plot_rank_histogram(sbc_result, n_bins=20, output_dir=None):
    """
    Plot rank histogram for each parameter dimension.

    A uniform histogram indicates a well-calibrated posterior.
    U-shaped indicates overdispersion, hill-shaped indicates
    underdispersion, skewed indicates bias.

    Parameters
    ----------
    sbc_result : dict
        Output from run_sbc containing "ranks" and "ks_pvalue".
    n_bins     : int, optional
        Number of bins for the histogram. Default is 20.
    output_dir : str, optional
        If provided, saves the plot to this directory instead of showing it.
    """
    n_params = sbc_result["ranks"].shape[1]
    for i in range(n_params):
        plt.figure()
        plt.hist(sbc_result["ranks"][:, i], bins=n_bins, density=True, alpha=0.7)
        plt.title(f"Parameter {i} Rank Histogram (KS p={sbc_result['ks_pvalue'][i]:.3f})")
        plt.xlabel("Rank of True Parameter")
        plt.ylabel("Density")
        path = output_dir or "."
        plt.savefig(os.path.join(path, f"sbc_rank_histogram_param{i}.png"), dpi=150, bbox_inches='tight')
        plt.close()