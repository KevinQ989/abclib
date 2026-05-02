"""
Distance functions for comparing observed and simulated summary vectors.
 
All functions follow the signature ``d(s_obs, s_sim, ...) -> float``
and expect pre-normalised summary vectors. Normalisation is handled
inside the summary statistic object, not here.
 
Functions
---------
euclidean
    Standard L2 distance. Assumes statistics are already on comparable
    scales (i.e. normalised).
mahalanobis
    Accounts for correlation and scale via the inverse covariance matrix.
    Requires an estimated covariance matrix as input.
"""

import numpy as np
import numpy.linalg as la


def euclidean(s_obs, s_sim):
    """
    Compute the Euclidean distance between two summary vectors.
 
    Treats all summary statistics as independent and equally scaled.
    Appropriate when statistics have been normalised by prior-predictive
    standard deviation. Simple and computationally cheap.
 
    Parameters
    ----------
    s_obs : np.ndarray, shape (n_statistics,)
        Summary vector for the observed data.
    s_sim : np.ndarray, shape (n_statistics,)
        Summary vector for a simulated dataset.
 
    Returns
    -------
    distance : float
        Euclidean distance between ``s_obs`` and ``s_sim``.
    """
    return np.sqrt(np.sum((s_obs - s_sim) ** 2))


def mahalanobis(s_obs, s_sim, cov):
    """
    Compute the Mahalanobis distance between two summary vectors.
 
    Accounts for the scale and correlation structure of the summary
    statistics via the inverse covariance matrix. More informative than
    Euclidean distance when statistics are correlated, but requires
    estimating the covariance matrix from pilot simulations, which is
    unreliable when ``n_statistics`` is large relative to the pilot
    sample size.
 
    Parameters
    ----------
    s_obs : np.ndarray, shape (n_statistics,)
        Summary vector for the observed data.
    s_sim : np.ndarray, shape (n_statistics,)
        Summary vector for a simulated dataset.
    cov   : np.ndarray, shape (n_statistics, n_statistics)
        Covariance matrix of the summary statistics estimated from
        prior-predictive simulations. Typically computed once before
        inference and reused. Must be positive definite.
 
    Returns
    -------
    distance : float
        Mahalanobis distance between ``s_obs`` and ``s_sim``.
 
    Raises
    ------
    np.linalg.LinAlgError
        If ``cov`` is singular and cannot be inverted.
 
    Notes
    -----
    The covariance matrix should be estimated from a sufficiently large
    pilot sample (at least ``10 * n_statistics`` draws recommended) to
    ensure a stable inverse.
    """
    cov_inv = la.inv(cov)
    diff = s_obs - s_sim
    return float(np.sqrt(diff @ cov_inv @ diff))