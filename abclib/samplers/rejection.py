from abclib.base import BaseSampler
from abclib.results import ABCResult
from abclib.utils import select_epsilon
import numpy as np

class RejectionABC(BaseSampler):    
    def sample(self, s_obs, n_simulations, q=0.05, **kwargs):
        """
        Draw samples from the ABC posterior via rejection sampling.

        Runs a reference table of ``n_simulations`` prior draws, then
        accepts those whose distance to ``s_obs`` falls below the
        ``q``-quantile of all distances.

        Parameters
        ----------
        s_obs         : np.ndarray, shape (n_statistics,)
            Summary statistics computed from the observed data.
        n_simulations : int
            Number of prior draws and simulator calls to make.
        q             : float, optional
            Quantile of the distance distribution used to set epsilon.
            Default is 0.05, accepting the closest 5% of draws.

        Returns
        -------
        ABCResult
            Contains accepted samples, their distances, total simulator
            calls, and the threshold epsilon.

        Raises
        ------
        ValueError
            If no samples are accepted. Indicates epsilon is too small
            or summary statistics are poorly specified.
        """
        thetas = []
        distances = []

        for _ in range(n_simulations):
            theta = self.prior()
            s_sim = self._simulate_and_summarise(theta)
            d = self.distance(s_obs, s_sim)

            thetas.append(theta)
            distances.append(d)
        
        thetas = np.array(thetas)
        distances = np.array(distances)
        epsilon = select_epsilon(distances, q)
        accepted_mask = distances <= epsilon

        if not np.any(accepted_mask):
            raise ValueError(
                f"No samples accepted. Consider increasing q (currently {q}) "
                "or improving your summary statistics."
            )

        return ABCResult(
            samples = thetas[accepted_mask],
            distances = distances[accepted_mask],
            n_simulations = n_simulations,
            epsilon = epsilon,
            summary_statistic = self.summary_statistic
        )