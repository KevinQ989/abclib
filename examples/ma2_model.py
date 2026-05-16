"""
MA(2) validation model for abclib.

Provides a prior sampler, simulator, and summary statistic functions
for a second-order moving average model with known exact posterior.
Used to validate every abclib method before application to the
Lotka-Volterra case study.
"""
from .model import Model
import numpy as np
import scipy.linalg as la


class MA2(Model):
    """
    Second-order moving average model with tractable exact posterior.

    The MA(2) process is defined as:
        y_t = eps_t + theta1 * eps_{t-1} + theta2 * eps_{t-2}
    where eps_t ~ iid N(0, 1).

    The prior is uniform over the invertibility region R, defined by:
        theta1 + theta2 < 1
        theta2 - theta1 < 1
        -1 < theta2 < 1

    The exact posterior is Gaussian with a known autocovariance matrix,
    serving as ground truth for validating all abclib methods.

    Parameters
    ----------
    T : int, optional
        Length of the simulated time series. Default is 100.
    """

    def __init__(self, T=100):
        super().__init__(name="MA(2)")
        self.T = T


    @property
    def prior_bounds(self):
        """
        Bounds of the prior support for each parameter.

        Returns a conservative box containing the invertibility region.
        Used by RegressionAdjustment to reflect adjusted samples back
        into the valid region.

        Returns
        -------
        list of (float, float)
            [(-1, 1), (-1, 1)] for theta1 and theta2.
        """
        return [(-1, 1), (-1, 1)]


    def prior(self):
        """
        Sample from the uniform prior over the MA(2) invertibility region.

        Uses rejection sampling within the bounding box [-1, 1]^2 until
        a draw satisfying all invertibility constraints is found.

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


    def prior_pdf(self, theta):
        """
        Evaluate the uniform prior density over the MA(2) invertibility region.

        The invertibility region has area 2, so the uniform density is 1/2.
        However, since prior_pdf is only used in MH acceptance ratios where
        the normalising constant cancels, the exact value matters only for
        the ratio between two evaluations.

        Parameters
        ----------
        theta : np.ndarray, shape (2,)
            Parameter vector [theta1, theta2].

        Returns
        -------
        float
            Prior density at the given theta. Returns 0.0 if theta is
            outside the invertibility region.
        """
        theta1, theta2 = theta
        if theta1 + theta2 < 1 and theta2 - theta1 < 1 and abs(theta2) < 1:
            return 0.25
        return 0.0


    def prior_density(self, thetas):
        """
        Vectorised prior density for a batch of parameter vectors.

        Required by SMCABC for importance weight computation.

        Parameters
        ----------
        thetas : np.ndarray, shape (n_thetas, 2)
            Array of parameter vectors to evaluate.

        Returns
        -------
        np.ndarray, shape (n_thetas,)
            Prior densities for each input theta.
        """
        densities = np.zeros(thetas.shape[0])
        for i, theta in enumerate(thetas):
            densities[i] = self.prior_pdf(theta)
        return densities


    def simulator(self, theta):
        """
        Simulate a single MA(2) time series.

        Parameters
        ----------
        theta : np.ndarray, shape (2,)
            Parameter vector [theta1, theta2].

        Returns
        -------
        y : np.ndarray, shape (T,)
            Simulated time series.
        """
        theta1, theta2 = theta
        eps = np.random.normal(loc=0, scale=1, size=self.T)
        y = np.zeros(self.T)
        y[0] = eps[0]
        y[1] = eps[1] + theta1 * eps[0]
        for t in range(2, self.T):
            y[t] = eps[t] + theta1 * eps[t - 1] + theta2 * eps[t - 2]
        return y


    def _autocorr(self, y, lag):
        """
        Compute sample autocorrelation at a given lag.

        Returns 0 if the series or lagged series is constant, to avoid
        division by zero in np.corrcoef.

        Parameters
        ----------
        y : np.ndarray, shape (T,)
            Time series.
        lag : int
            Lag at which to compute autocorrelation.

        Returns
        -------
        float
            Sample autocorrelation at the given lag.
        """
        if np.std(y) == 0 or np.std(y[lag:]) == 0:
            return 0.0
        return float(np.corrcoef(y[:-lag], y[lag:])[0, 1])


    @property
    def SUMMARY_FUNCTIONS(self):
        """
        Dictionary of named scalar summary functions for HandCraftedSummary.

        The lag-1 and lag-2 sample autocorrelations are approximately
        sufficient statistics for the MA(2) model, as the theoretical
        autocorrelations are direct functions of theta1 and theta2.

        Returns
        -------
        dict[str, callable]
            Keys: 'autocorr_lag1', 'autocorr_lag2'.
            Each callable takes y of shape (T,) and returns a scalar.
        """
        return {
            "autocorr_lag1": lambda y: self._autocorr(y, 1),
            "autocorr_lag2": lambda y: self._autocorr(y, 2),
        }


    def H_FUNCTION(self, y):
        """
        Compute candidate summary statistics for SemiAutomaticSummary.

        Parameters
        ----------
        y : np.ndarray, shape (T,)
            Simulated time series.

        Returns
        -------
        h : np.ndarray, shape (2,)
            Lag-1 and lag-2 sample autocorrelations.
        """
        return np.array([self._autocorr(y, 1), self._autocorr(y, 2)])


    def autocovariance(self, theta):
        """
        Compute the theoretical autocovariance matrix for an MA(2) process.

        The autocovariance function of an MA(2) process has a known closed
        form. The resulting Toeplitz matrix is used to evaluate the exact
        Gaussian likelihood for comparison with the ABC posterior.

        Parameters
        ----------
        theta : np.ndarray, shape (2,)
            Parameter vector [theta1, theta2].

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
            [gamma0, gamma1, gamma2] + [0] * (self.T - 3)
        )