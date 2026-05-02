from abc import ABC, abstractmethod
import numpy as np

class BaseSampler(ABC):
    """
    Abstract base class for all ABC samplers.

    Parameters
    ----------
    prior             : callable
        Callable with no arguments that returns a single draw from the
        prior as a 1D ``np.ndarray`` of shape ``(n_params,)``.
    simulator         : callable
        Callable that accepts a parameter vector of shape ``(n_params,)``
        and returns simulated data in whatever form the summary statistic
        expects.
    summary_statistic : BaseSummaryStatistic
        A fitted summary statistic object exposing a ``transform`` method.
        Must be fitted (via ``fit``) on pilot simulations before being
        passed here.
    distance          : callable
        Callable of the form ``d(s_obs, s_sim) -> float`` that returns
        the scalar distance between two summary vectors.
    """
    def __init__(self, prior, simulator, summary_statistic, distance):
        self.prior = prior
        self.simulator = simulator
        self.summary_statistic = summary_statistic
        self.distance = distance
    
    @abstractmethod
    def sample(self, s_obs, n_sample, **kwargs):
        """
        Draw samples from the ABC posterior.

        Parameters
        ----------
        s_obs     : np.ndarray, shape (n_statistics,)
            Summary statistics computed from the observed data.
        n_samples : int
            Number of posterior samples to return.
        
        Returns
        -------
        ABCResult
            Object containing posterior samples, distances, simulator
            call count, and the tolerance used.
        """
        pass

    def _simulate_and_summarise(self, theta):
        """Simulate data and compute summary statistics."""
        y_sim = self.simulator(theta)
        s_sim = self.summary_statistic.transform(y_sim)
        return s_sim