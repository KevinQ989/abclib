from abclib.results import SLResult
import numpy as np
import scipy.stats as stats

class SyntheticLikelihood:
    """
    Synthetic likelihood method for ABC inference.

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
    """
    def __init__(self, prior, simulator, summary_statistic, prior_pdf, proposal_std):
        if not getattr(summary_statistic, "_is_fitted", False):
            raise ValueError(
                "summary_statistic must be fitted before being passed to SyntheticLikelihood."
            )
        self.prior = prior
        self.simulator = simulator
        self.summary_statistic = summary_statistic
        self.prior_pdf = prior_pdf
        self.proposal_std = proposal_std
    

    def sample(self, s_obs, n_simulations, M, **kwargs):
        """
        Draw samples from the ABC posterior using a synthetic likelihood.
        
        Parameters
        ----------
        s_obs         : np.ndarray, shape (n_statistics,)
            Summary statistics computed from the observed data.
        n_simulations : int
            Number of prior draws and simulator calls to make.
        M             : int
            Number of replicates per parameter draw to estimate the synthetic likelihood.

        Returns
        -------
        SLResult
            Contains accepted samples, their synthetic log likelihoods, and total simulator calls.
        """
        thetas = []
        log_likelihoods = []
        n_accepted = 0

        theta_current = self.prior()
        log_likelihood_current = self._log_likelihood(theta_current, M, s_obs)
        while log_likelihood_current == -np.inf:
            theta_current = self.prior()
            log_likelihood_current = self._log_likelihood(theta_current, M, s_obs)
        thetas.append(theta_current)
        log_likelihoods.append(log_likelihood_current)

        for _ in range(1, n_simulations):
            theta_proposed = self._propose(theta_current)
            log_likelihood_proposed = self._log_likelihood(theta_proposed, M, s_obs)

            alpha = self._acceptance_probability(
                theta_current, theta_proposed,
                log_likelihood_current, log_likelihood_proposed
            )
            if np.random.rand() < alpha:
                theta_current = theta_proposed
                log_likelihood_current = log_likelihood_proposed
                n_accepted += 1

            thetas.append(theta_current)
            log_likelihoods.append(log_likelihood_current)


        return SLResult(
            samples = np.array(thetas),
            log_likelihoods = np.array(log_likelihoods),
            n_simulations = n_simulations * M,
            n_accepted = n_accepted,
            summary_statistic = self.summary_statistic
        )
    
    
    def _log_likelihood(self, theta, M, s_obs):
        """
        Estimate the synthetic likelihood at a given parameter vector.

        Parameters
        ----------
        theta : np.ndarray, shape (n_params,)
            Parameter vector at which to estimate the synthetic likelihood.
        M     : int
            Number of replicates to use for estimating the synthetic likelihood.
        s_obs : np.ndarray, shape (n_statistics,)
            Summary statistics computed from the observed data.

        Returns
        -------
        float
            Estimated synthetic likelihood value at the given parameter vector.
        """
        summaries = np.array([self.summary_statistic.transform(self.simulator(theta)) for _ in range(M)])
        if summaries.ndim == 1:
            summaries = summaries[:, np.newaxis]
        if not np.all(np.isfinite(summaries)):
            return -np.inf
        mean_sim = np.mean(summaries, axis=0)
        cov_sim = np.cov(summaries, rowvar=False)
        cov_sim += 1e-6 * np.eye(len(cov_sim))
        if not np.all(np.isfinite(cov_sim)):
            return -np.inf
        return stats.multivariate_normal.logpdf(s_obs, mean=mean_sim, cov=cov_sim)

    
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


    def _acceptance_probability(self, theta_current, theta_proposed, log_likelihood_current, log_likelihood_proposed):
        """
        Compute the acceptance probability for new particles.

        Parameters
        ----------
        theta_current  : np.ndarray, shape (n_params,)
            Current parameter vector in the MCMC chain.
        theta_proposed : np.ndarray, shape (n_params,)
            Proposed parameter vector drawn from the proposal distribution.
        log_likelihood_current : float
            Synthetic likelihood at the current parameter vector.
        log_likelihood_proposed : float
            Synthetic likelihood at the proposed parameter vector.
        
        Returns
        -------
        prob : float
            Acceptance probability for the proposed parameter vector. 0 <= prob <= 1.
        """
        if log_likelihood_current == -np.inf:
            return 1.0
        log_prior_ratio = np.log(
            self.prior_pdf(theta_proposed) + 1e-300
        ) - np.log(
            self.prior_pdf(theta_current) + 1e-300
        )
        log_alpha = log_likelihood_proposed + log_prior_ratio - log_likelihood_current
        if log_alpha >= 0:
            return 1.0
        return float(min(1, np.exp(log_alpha)))