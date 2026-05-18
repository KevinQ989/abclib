from .base import BaseSummaryStatistic
import numpy as np

class SemiAutomaticSummary(BaseSummaryStatistic):
    """
    Semi-automatic summary statistics via pilot regression.

    Learns optimal linear weights from a pilot run by regressing each
    parameter onto a set of candidate statistics ``h(y)``. The resulting
    summary vector has one component per parameter, each approximating
    the posterior mean $E[\\theta_j \\mid y]$.

    Parameters
    ----------
    h : callable
        Transformation applied to raw simulations to produce candidate
        statistics. Must accept a single simulation and return a 1D
        ``np.ndarray`` of fixed length across all simulations.
    """
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.intercepts_ = None
        self.coefficients_ = None


    def fit(self, thetas, simulations):
        """
        Fit a separate OLS regression for each parameter.

        Parameters
        ----------
        thetas : np.ndarray, shape (n_pilot, n_params)
            Parameter vectors from the pilot run.
        simulations : list of length n_pilot
            Raw simulated datasets from the pilot run.

        Returns
        -------
        self
            Returns the instance to allow method chaining.
        """
        n_pilot, n_params = thetas.shape
        h_values = np.array([self.h(s) for s in simulations])
        X = np.hstack([np.ones((n_pilot, 1)), h_values])

        self.intercepts_ = np.zeros(n_params)
        self.coefficients_ = np.zeros((n_params, h_values.shape[1]))

        for j in range(n_params):
            beta = np.linalg.lstsq(X, thetas[:, j], rcond=None)[0]
            self.intercepts_[j] = beta[0]
            self.coefficients_[j] = beta[1:]

        return super().fit(thetas, simulations)


    def transform(self, simulation):
        """
        Compute the summary statistic vector for a single simulation.

        Parameters
        ----------
        simulation : array-like
            A single raw simulated dataset in the same format as
            elements of ``simulations`` passed to ``fit``.

        Returns
        -------
        summary : np.ndarray, shape (n_params,)
            Summary vector with one component per parameter, where
            ``n_params`` is determined by ``thetas`` passed to ``fit``.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called before ``transform``.
        """
        super().transform(simulation)
        h_val = self.h(simulation)
        return self.intercepts_ + self.coefficients_ @ h_val