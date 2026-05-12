from abclib.base import BaseSampler
from abclib.results import ABCResult
import numpy as np

class MCMCABC(BaseSampler):
    """
    Monte Carlo Markov Chain ABC sampler.

    Creates a Markov chain with stationary distribution approximating
    the ABC posterior using a Metropolis-Hastings algorithm.
    The proposal distribution is an isotropic Gaussian random walk.

    Parameters
    ----------
    prior             : callable
        Callable with no arguments returning a single draw from the
        prior as a 1D ``np.ndarray`` of shape ``(n_params,)``.
    simulator         : callable
        Callable accepting a parameter vector and returning simulated
        data.
    summary_statistic : BaseSummaryStatistic
        Fitted summary statistic object exposing a ``transform`` method.
    distance          : callable
        Callable of the form ``d(s_obs, s_sim) -> float``.
    prior_pdf         : callable
        Callable of the form ``prior_pdf(theta) -> float`` returning the
        prior density at a given parameter vector.
    proposal_std      : float
        Standard deviation of the isotropic Gaussian proposal distribution.
    """
    def __init__(self, prior, simulator, summary_statistic, distance, prior_pdf, proposal_std):
        super().__init__(prior, simulator, summary_statistic, distance)
        self.prior_pdf = prior_pdf  
        self.proposal_std = proposal_std
    

    def sample(self, s_obs, n_samples, epsilon, **kwargs):
        """
        Draw samples from the ABC posterior via MCMC.

        Parameters
        ----------
        s_obs     : np.ndarray, shape (n_statistics,)
            Summary statistics computed from the observed data.
        n_samples : int
            Number of MCMC samples to draw.
        epsilon   : float
            Tolerance threshold for accepting proposed samples.

        Returns
        -------
        ABCResult
            Object containing posterior samples, distances, simulator
            call count, and the tolerance used.
        """
        total_simulations = 0
        samples = []
        distances = []
        summaries = []

        while len(samples) == 0:
            theta_current = self.prior()
            s_sim_current = self._simulate_and_summarise(theta_current)
            d_current = self.distance(s_obs, s_sim_current)
            total_simulations += 1

            if d_current <= epsilon:
                samples.append(theta_current)
                distances.append(d_current)
                summaries.append(s_sim_current)
    
        while len(samples) < n_samples:
            theta_proposed = self._propose(theta_current)
            s_sim_proposed = self._simulate_and_summarise(theta_proposed)
            d_proposed = self.distance(s_obs, s_sim_proposed)
            total_simulations += 1

            if d_proposed <= epsilon:
                alpha = self._acceptance_probability(theta_current, theta_proposed)
                if np.random.rand() < alpha:
                    theta_current = theta_proposed
                    s_sim_current = s_sim_proposed
                    d_current = d_proposed

            samples.append(theta_current)
            distances.append(d_current)
            summaries.append(s_sim_current)

        return ABCResult(
            samples = np.array(samples),
            distances = np.array(distances),
            summaries = np.array(summaries),
            n_simulations = total_simulations,
            epsilon = epsilon,
            summary_statistic = self.summary_statistic
        )

    
    def _propose(self, theta_current):
        """
        Draw a proposal from the isotropic Gaussian kernel.

        Parameters
        ----------
        theta_current : np.ndarray, shape (n_params,)
            Current parameter vector in the MCMC chain.

        Returns
        -------
        theta_proposed : np.ndarray, shape (n_params,)
            Proposed parameter vector drawn from the proposal distribution.
        """
        return theta_current + np.random.randn(len(theta_current)) * self.proposal_std


    def _acceptance_probability(self, theta_current, theta_proposed):
        """
        Compute the acceptance probability for new particles.

        Parameters
        ----------
        theta_current  : np.ndarray, shape (n_params,)
            Current parameter vector in the MCMC chain.
        theta_proposed : np.ndarray, shape (n_params,)
            Proposed parameter vector drawn from the proposal distribution.
        
        Returns
        -------
        prob : float
            Acceptance probability for the proposed parameter vector. 0 <= prob <= 1.
        """
        if self.prior_pdf(theta_current) == 0:
            return 0.0
        return min(1, self.prior_pdf(theta_proposed) / self.prior_pdf(theta_current))