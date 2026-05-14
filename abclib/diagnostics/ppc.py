import numpy as np

def run_ppc(result, simulator, y_obs, test_statistic, n_samples):
    """
    Run posterior predictive checks.

    For each sample, draws a parameter vector from the ABC posterior,
    simulates a replicate dataset under it, and evaluates a test
    statistic on the replicate. The distribution of test statistic
    values across replicates is compared to the test statistic computed
    on the observed data via a posterior predictive p-value.

    Note: the diagnostic is only as informative as the choice of
    test_statistic. A statistic insensitive to model misfit will
    produce uninformative p-values even when the model is wrong.
    Additionally, PPC uses the observed data twice --- once to fit
    the posterior and once to evaluate it --- making it conservative.
    
    Parameters
    ----------
    result         : ABCResult
        Object containing posterior samples, distances, simulator call count,
        and the tolerance used.
    simulator      : callable
        Callable that accepts a parameter vector of shape ``(n_params,)``
        and returns simulated data in whatever form the summary statistic
        expects.
    y_obs          : array-like
        The observed data to compare against.
    test_statistic : callable
        A function that takes data as input and returns a scalar test statistic.
    n_samples      : int
        The number of posterior predictive samples to generate.

    Returns
    -------
    dict with keys:
        "t_rep": np.ndarray, shape (n_samples,)
            The test statistic computed on each posterior predictive sample.
        "t_obs": float
            The test statistic computed on the observed data.
        "p_value": float
            The proportion of posterior predictive samples where the test statistic
            is as extreme or more extreme than the observed test statistic.
    """
    indices = np.random.choice(len(result.samples), size=n_samples, replace=True)
    t_obs = test_statistic(y_obs)
    t_rep = np.array([test_statistic(simulator(result.samples[idx])) for idx in indices])
    p_value = float(np.mean(t_rep >= t_obs))
    return {
        "t_rep": t_rep,
        "t_obs": t_obs,
        "p_value": p_value
    }