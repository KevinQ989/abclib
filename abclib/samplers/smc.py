from abclib.base import BaseSampler
from abclib.results import ABCResult
from abclib.utils import select_epsilon
import numpy as np

class SMCABC(BaseSampler):
    """
    Sequential Monte Carlo ABC sampler.

    Evolves a population of M particles through successively tighter
    tolerances, concentrating simulation effort near the posterior.
    Addresses the inefficiency of rejection ABC by avoiding uniform
    sampling from the prior at small tolerances.

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
    prior_density     : callable
        Callable accepting a 2D ``np.ndarray`` of shape
        ``(n_particles, n_params)`` and returning a 1D ``np.ndarray``
        of shape ``(n_particles,)`` containing the prior density for
        each particle. Must return 0 for particles outside the prior
        support.
    """
    def __init__(self, prior, simulator, summary_statistic, distance, prior_density):
        super().__init__(prior, simulator, summary_statistic, distance)
        self.prior_density = prior_density

    def sample(self, s_obs, M, T, q=0.50, **kwargs):
        """
        Draw samples from the ABC posterior via SMC.

        Parameters
        ----------
        s_obs : np.ndarray, shape (n_statistics,)
            Summary statistics computed from the observed data.
        M     : int
            Number of particles to maintain at each stage.
        T     : int
            Number of SMC stages.
        q     : float, optional
            Quantile used to set the initial tolerance. Default is 0.50.

        Returns
        -------
        ABCResult
            Contains accepted samples, their distances, summaries, total simulator calls,
            final epsilon, and the full epsilon sequence across stages.
        """
        total_simulations = 0
        all_epsilons = []

        thetas, weights, distances, summaries, epsilon, n_simulations = self._initialise(s_obs, M, q)
        total_simulations += n_simulations
        all_epsilons.append(epsilon)
        
        eps = np.median(distances)
        for t in range(1, T):
            thetas_old = thetas.copy()
            weights_old = weights.copy()
            
            thetas_new = []
            distances_new = []
            summaries_new = []

            while len(thetas_new) < M:
                theta_star = self._perturb(thetas, weights)
                s_sim = self._simulate_and_summarise(theta_star)
                d = self.distance(s_obs, s_sim)
                total_simulations += 1
                if d <= eps:
                    thetas_new.append(theta_star)
                    distances_new.append(d)
                    summaries_new.append(s_sim)

            thetas = np.array(thetas_new)
            distances = np.array(distances_new)
            summaries = np.array(summaries_new)

            weights = self._compute_weights(thetas, thetas_old, weights_old)

            if self._ess(weights) < M / 2:
                thetas, distances, summaries, weights = self._resample(thetas, weights, distances, summaries)

            eps = np.median(distances)
            all_epsilons.append(eps)

        return ABCResult(
            samples = thetas,
            distances = distances,
            summaries = summaries,
            n_simulations = total_simulations,
            epsilon = all_epsilons[-1],
            summary_statistic = self.summary_statistic,
            epsilons = np.array(all_epsilons)
        )


    def _initialise(self, s_obs, M, q):
        """
        Initialise the particle population by sampling from the prior.

        Draws particles until ``M`` are accepted, setting the
        initial tolerance as the ``q``-quantile of all distances
        observed during initialisation.

        Parameters
        ----------
        s_obs : np.ndarray, shape (n_statistics,)
            Summary statistics of the observed data.
        M     : int
            Number of particles to collect.
        q     : float
            Quantile used to set the initial tolerance epsilon.

        Returns
        -------
        thetas        : np.ndarray, shape (M, n_params)
            Initial particle population.
        weights       : np.ndarray, shape (M,)
            Uniform initial weights summing to 1.
        distances     : np.ndarray, shape (M,)
            Distances of accepted particles.
        summaries     : np.ndarray, shape (M, n_statistics)
            Summary vectors of accepted particles.
        epsilon       : float
            Initial tolerance set as the q-quantile of all distances.
        n_simulations : int
            Total simulator calls made during initialisation.
        """
        all_thetas = []
        all_distances = []
        all_summaries = []
        n_simulations = 0

        while len(all_thetas) < M:
            theta = self.prior()
            s_sim = self._simulate_and_summarise(theta)
            d = self.distance(s_obs, s_sim)
            n_simulations += 1
            all_thetas.append(theta)
            all_distances.append(d)
            all_summaries.append(s_sim)
        
        all_thetas = np.array(all_thetas)
        all_distances = np.array(all_distances)
        all_summaries = np.array(all_summaries)

        epsilon = select_epsilon(all_distances, q)
        accepted_mask = all_distances <= epsilon

        thetas = all_thetas[accepted_mask]
        distances = all_distances[accepted_mask]
        summaries = all_summaries[accepted_mask]

        while len(thetas) < M:
            theta = self.prior()
            s_sim = self._simulate_and_summarise(theta)
            d = self.distance(s_obs, s_sim)
            n_simulations += 1
            if d <= epsilon:
                thetas = np.vstack([thetas, theta[None, :]])
                distances = np.append(distances, d)
                summaries = np.vstack([summaries, s_sim[None, :]])

        weights = np.ones(len(thetas)) / len(thetas)

        return thetas, weights, distances, summaries, epsilon, n_simulations


    def _compute_bandwidth(self, thetas, weights):
        """
        Compute the adaptive Gaussian perturbation bandwidth.

        Follows Beaumont et al. (2009) for parameter k,
        h_k = 2 * sqrt(weighted variance of theta_k).

        Parameters
        ----------
        thetas  : np.ndarray, shape (n_particles, n_params)
            Current particle population.
        weights : np.ndarray, shape (n_particles,)
            Normalised particle weights.

        Returns
        -------
        bandwidth : np.ndarray, shape (n_params,)
            Perturbation bandwidth for each parameter dimension.
        """
        mean = np.average(thetas, axis=0, weights=weights)
        var = np.average((thetas - mean)**2, axis=0, weights=weights)
        return 2 * np.sqrt(var)
    

    def _perturb(self, thetas, weights):
        """
        Sample and perturb one particle using a Gaussian kernel.

        Samples a particle from the current population proportional to
        weights, perturbs it with Gaussian noise scaled by the adaptive
        bandwidth, and repeats until the perturbed particle has non-zero
        prior density.

        Parameters
        ----------
        thetas  : np.ndarray, shape (n_particles, n_params)
            Current particle population.
        weights : np.ndarray, shape (n_particles,)
            Normalised particle weights.

        Returns
        -------
        theta_star : np.ndarray, shape (n_params,)
            Perturbed particle within prior support.
        """
        bandwidth = self._compute_bandwidth(thetas, weights)
        while True:
            idx = np.random.choice(len(thetas), p=weights)
            theta_star = thetas[idx] + np.random.normal(
                0, bandwidth, size=thetas.shape[1]
            )
            if self.prior_density(theta_star[None, :])[0] > 0:
                return theta_star
            

    def _proposal_density(self, thetas_new, thetas_old, weights_old,
                          bandwidth):
        """
        Compute the Gaussian mixture proposal density for new particles.

        Parameters
        ----------
        thetas_new  : np.ndarray, shape (n_new, n_params)
            New particle positions to evaluate.
        thetas_old  : np.ndarray, shape (n_old, n_params)
            Previous particle population.
        weights_old : np.ndarray, shape (n_old,)
            Normalised weights of the previous population.
        bandwidth   : np.ndarray, shape (n_params,)
            Perturbation bandwidth vector.

        Returns
        -------
        density : np.ndarray, shape (n_new,)
            Proposal density evaluated at each new particle.
        """
        diff = (thetas_new[:, None, :] - thetas_old[None, :, :]) / bandwidth
        log_kernel = -0.5 * np.sum(diff**2, axis=-1)
        log_norm = -np.sum(np.log(bandwidth * np.sqrt(2 * np.pi)))
        kernels = np.exp(log_kernel + log_norm)
        return kernels @ weights_old


    def _compute_weights(self, thetas_new, thetas_old, weights_old):
        """
        Compute normalised importance weights for the new population.

        Parameters
        ----------
        thetas_new  : np.ndarray, shape (n_particles, n_params)
            New particle population.
        thetas_old  : np.ndarray, shape (n_particles, n_params)
            Previous particle population.
        weights_old : np.ndarray, shape (n_particles,)
            Normalised weights from the previous stage.

        Returns
        -------
        weights : np.ndarray, shape (n_particles,)
            Normalised importance weights for the new population.
        """
        bandwidth = self._compute_bandwidth(thetas_old, weights_old)
        prior_density = self.prior_density(thetas_new)
        proposal_density = self._proposal_density(thetas_new, thetas_old, weights_old, bandwidth)
        raw_weights = np.where(proposal_density > 0, prior_density / proposal_density, 0.0)
        total = raw_weights.sum()
        if total == 0:
            return np.ones(len(thetas_new)) / len(thetas_new)
        return raw_weights / total


    def _resample(self, thetas, weights, distances, summaries):
        """
        Systematic resampling proportional to particle weights.

        Parameters
        ----------
        thetas    : np.ndarray, shape (n_particles, n_params)
            Current particle population.
        weights   : np.ndarray, shape (n_particles,)
            Normalised particle weights.
        distances : np.ndarray, shape (n_particles,)
            Distances of current particles.
        summaries : np.ndarray, shape (n_particles, n_statistics)
            Summary vectors of current particles.

        Returns
        -------
        thetas    : np.ndarray
            Resampled particles.
        distances : np.ndarray
            Resampled distances.
        summaries : np.ndarray
            Resampled summaries.
        weights   : np.ndarray
            Reset uniform weights summing to 1.
        """
        n = len(thetas)
        weights = weights / np.sum(weights)
        idx = np.random.choice(n, size=n, replace=True, p=weights)
        return thetas[idx], distances[idx], summaries[idx], np.ones(n) / n


    def _ess(self, weights):
        """
        Compute the effective sample size of the particle population.

        Parameters
        ----------
        weights : np.ndarray, shape (n_particles,)
            Normalised particle weights.

        Returns
        -------
        ess : float
            Effective sample size. Equal to n_particles when weights
            are uniform; approaches 1 under severe degeneracy.
        """
        return 1.0 / np.sum(weights**2)