"""
MA(2) validation model for abclib.

Provides a prior sampler, simulator, and summary statistic functions
for a second-order moving average model with known exact posterior.
Used to validate every abclib method before application to the
Lotka-Volterra case study.
"""
import numpy as np
import scipy.linalg as la

def prior():
    """
    Sample from the uniform prior over the MA(2) invertibility region.

    Returns
    -------
    theta : np.ndarray, shape (2,)
        Parameter vector [theta1, theta2] satisfying the invertibility
        constraints.
    """
    while True:
        theta1 = np.random.uniform(-1, 1)
        theta2 = np.random.uniform(-1, 1)
        if theta1 + theta2 < 1 and theta2 - theta1 < 1 and abs(theta2) < 1:
            return np.array([theta1, theta2])


def simulator(theta, T=100):
    """
    Simulate a single MA(2) time series.

    Parameters
    ----------
    theta : np.ndarray, shape (2,)
        Parameter vector [theta1, theta2].
    T : int, optional
        Length of the time series. Default is 100.

    Returns
    -------
    y : np.ndarray, shape (T,)
        Simulated time series.
    """
    theta1, theta2 = theta
    eps = np.random.normal(loc=0, scale=1, size=T)
    y = np.zeros(T)
    y[0] = eps[0]
    y[1] = eps[1] + theta1 * eps[0]
    for t in range(2, T):
        y[t] = eps[t] + theta1 * eps[t-1] + theta2 * eps[t-2]
    return y


def _autocorr(y, lag):
    """Compute sample autocorrelation at a given lag, returning 0 if constant."""
    if np.std(y) == 0 or np.std(y[lag:]) == 0:
        return 0.0
    return float(np.corrcoef(y[:-lag], y[lag:])[0, 1])


SUMMARY_FUNCTIONS = {
    "autocorr_lag1": lambda y: _autocorr(y, 1),
    "autocorr_lag2": lambda y: _autocorr(y, 2),
}


def H_FUNCTION(y):
    return np.array([_autocorr(y, 1), _autocorr(y, 2)])


def autocovariance(theta, T=100):
    """
    Compute the theoretical autocovariance matrix for an MA(2) process.

    Parameters
    ----------
    theta : np.ndarray, shape (2,)
        Parameter vector [theta1, theta2].
    T : int, optional
        Length of the time series. Default is 100.

    Returns
    -------
    Sigma : np.ndarray, shape (T, T)
        Symmetric Toeplitz autocovariance matrix.
    """
    theta1, theta2 = theta
    gamma0 = 1 + theta1 ** 2 + theta2 ** 2
    gamma1 = theta1 * (1 + theta2)
    gamma2 = theta2
    return la.toeplitz(
        [gamma0, gamma1, gamma2] + [0] * (T - 3)
    )
