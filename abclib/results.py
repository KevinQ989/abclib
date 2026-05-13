from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class ABCResult:
    """
    Container for ABC posterior samples and run diagnostics.
 
    Returned by all sampler ``sample`` methods. Stores everything
    needed to run downstream diagnostics (SBC, PPC, STR) and to
    compare methods against one another.
 
    Parameters
    ----------
    samples           : np.ndarray, shape (n_samples, n_params)
        Accepted parameter vectors forming the ABC posterior sample.
    distances         : np.ndarray, shape (n_samples,)
        Distance value for each accepted sample.
    summaries         : np.ndarray, shape (n_samples, n_statistics)
        Summary vectors of accepted samples.
    n_simulations     : int
        Total number of simulator calls made, including rejected draws.
    epsilon           : float
        Tolerance threshold used for acceptance.
    summary_statistic : BaseSummaryStatistic, optional
        The fitted summary statistic object used during inference.
        Stored so that downstream tools (e.g. PPC) can apply the same
        transformation to new simulations without re-fitting.
    epsilons          : np.ndarray, shape (n_iterations,), optional
        For sequential methods, the sequence of tolerances used across iterations.
 
    Properties
    ----------
    acceptance_rate : float
        Fraction of simulator calls that produced an accepted sample.
    n_params        : int
        Dimensionality of the parameter space.
    """
    samples: np.ndarray
    distances: np.ndarray
    summaries: np.ndarray
    n_simulations: int
    epsilon: float
    summary_statistic: object
    epsilons: Optional[np.ndarray] = field(default=None, repr=False)
 

    @property
    def acceptance_rate(self):
        """Fraction of draws accepted out of all simulator calls."""
        return len(self.samples) / self.n_simulations
 

    @property
    def n_params(self):
        """Number of parameters in the posterior sample."""
        return self.samples.shape[1] if self.samples.ndim > 1 else 1
 

    def posterior_mean(self):
        """
        Compute the posterior mean for each parameter.
 
        Returns
        -------
        mean : np.ndarray, shape (n_params,)
            Mean of the accepted samples.
        """
        return self.samples.mean(axis=0)
 

    def credible_interval(self, alpha=0.90):
        """
        Compute equal-tailed credible intervals for each parameter.
 
        Parameters
        ----------
        alpha : float, optional
            Credible mass. Default is 0.90 for a 90% interval.
 
        Returns
        -------
        lower : np.ndarray, shape (n_params,)
            Lower bound of the credible interval.
        upper : np.ndarray, shape (n_params,)
            Upper bound of the credible interval.
        """
        tail = (1 - alpha) / 2
        lower = np.quantile(self.samples, tail, axis=0)
        upper = np.quantile(self.samples, 1 - tail, axis=0)
        return lower, upper
 

    def __repr__(self):
        return (
            f"ABCResult("
            f"n_samples={len(self.samples)}, "
            f"n_simulations={self.n_simulations}, "
            f"acceptance_rate={self.acceptance_rate:.4f}, "
            f"epsilon={self.epsilon:.4f})"
        )
    

@dataclass
class SLResult:
    """
    Container for synthetic likelihood samples and diagnostics.
 
    Parameters
    ----------
    samples           : np.ndarray, shape (n_samples, n_params)
        Parameter vectors sampled from the ABC posterior using synthetic likelihood.
    likelihoods       : np.ndarray, shape (n_samples,)
        Estimated synthetic likelihood values for each sample.
    n_simulations     : int
        Total number of simulator calls made across all parameter draws and replicates.
    n_accepted        : int
        Number of accepted samples.
    summary_statistic : object
        The fitted summary statistic object used during inference.
    """
    samples: np.ndarray
    likelihoods: np.ndarray
    n_simulations: int
    n_accepted: int
    summary_statistic: object
 
 
    @property
    def acceptance_rate(self):
        """Fraction of MH proposals accepted."""
        return self.n_accepted / len(self.samples)


    @property
    def n_params(self):
        """Number of parameters in the posterior sample."""
        return self.samples.shape[1] if self.samples.ndim > 1 else 1
 

    def posterior_mean(self):
        """
        Compute the posterior mean for each parameter.
 
        Returns
        -------
        mean : np.ndarray, shape (n_params,)
            Mean of the accepted samples.
        """
        return self.samples.mean(axis=0)
 

    def credible_interval(self, alpha=0.90):
        """
        Compute equal-tailed credible intervals for each parameter.
 
        Parameters
        ----------
        alpha : float, optional
            Credible mass. Default is 0.90 for a 90% interval.
 
        Returns
        -------
        lower : np.ndarray, shape (n_params,)
            Lower bound of the credible interval.
        upper : np.ndarray, shape (n_params,)
            Upper bound of the credible interval.
        """
        tail = (1 - alpha) / 2
        lower = np.quantile(self.samples, tail, axis=0)
        upper = np.quantile(self.samples, 1 - tail, axis=0)
        return lower, upper
 

    def __repr__(self):
        return (
            f"SLResult("
            f"n_samples={len(self.samples)}, "
            f"n_simulations={self.n_simulations}, "
            f"acceptance_rate={self.acceptance_rate:.4f})"
        )