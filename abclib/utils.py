"""
Utility functions shared across samplers and summary statistics.

Functions
---------
run_pilot
    Draw from the prior, simulate data, and return (theta, simulations).
select_epsilon
    Choose a tolerance threshold from a quantile of empirical distances.
"""

import numpy as np


def run_pilot(prior, simulator, n_pilot):
    """
    Run pilot simulations by drawing from the prior and simulating data.

    Parameters
    ----------
    prior   : callable
        Callable with no arguments that returns a single draw from the
        prior as a 1D ``np.ndarray`` of shape ``(n_params,)``.
    simulator : callable
        Callable that accepts a parameter vector of shape ``(n_params,)``
        and returns simulated data in whatever form the summary statistic
        expects.
    n_pilot  : int
        Number of pilot simulations to run.

    Returns
    -------
    thetas      : np.ndarray, shape (n_pilot, n_params)
        Array of parameter vectors drawn from the prior.
    simulations : list of length n_pilot
        List of simulated datasets corresponding to each parameter vector.
    """
    thetas = []
    simulations = []
    for _ in range(n_pilot):
        theta = prior()
        sim = simulator(theta)
        thetas.append(theta)
        simulations.append(sim)
    return np.array(thetas), simulations


def select_epsilon(distances, quantile):
    """
    Select a tolerance threshold from a quantile of empirical distances.

    Parameters
    ----------
    distances : np.ndarray, shape (n_pilot,)
        Array of distances between simulated and observed summary statistics
        from pilot simulations.
    quantile  : float in [0, 1]
        Quantile to use for selecting the tolerance threshold. For example,
        ``quantile=0.01`` selects the 1% quantile, meaning that the
        tolerance will be set to the distance below which 1% of the pilot
        distances fall.

    Returns
    -------
    epsilon : float
        Tolerance threshold corresponding to the specified quantile.

    Raises
    ------
    ValueError
        If ``quantile`` is not in the range [0, 1].
    """
    if not 0 <= quantile <= 1:
        raise ValueError(f"Quantile must be in the range [0, 1], got {quantile}.")
    return float(np.quantile(distances, quantile))