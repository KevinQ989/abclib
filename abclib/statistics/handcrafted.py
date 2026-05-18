from .base import BaseSummaryStatistic
import numpy as np

class HandCraftedSummary(BaseSummaryStatistic):
    """
    Hand-crafted summary statistics with prior-predictive normalisation.

    The user supplies a dictionary of named summary functions. During
    ``fit``, the prior-predictive standard deviation of each statistic
    is computed from pilot simulations and used to normalise the output
    of ``transform``. This ensures no single statistic dominates the
    distance purely by scale.

    Parameters
    ----------
    functions : dict[str, callable]
        Dictionary mapping statistic names to callables. Each callable
        accepts a single raw simulation and returns either a scalar or
        a 1D ``np.ndarray``. All outputs are concatenated into a single
        summary vector.
    """
    def __init__(self, functions):
        super().__init__()
        if not isinstance(functions, dict) or len(functions) == 0:
            raise ValueError("functions must be a non-empty dictionary.")
        self.functions = functions
        self.names = list(functions.keys())
        self.scale_ = None


    def _compute_raw(self, simulation):
        """Apply all summary functions and concatenate into a vector."""
        parts = []
        for fn in self.functions.values():
            part = np.atleast_1d(np.asarray(fn(simulation), dtype=np.float64))
            parts.append(part)
        return np.concatenate(parts)


    def fit(self, thetas, simulations):
        """
        Compute prior-predictive standard deviation for normalisation.

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
        raw = np.array([self._compute_raw(s) for s in simulations])
        self.scale_ = np.std(raw, axis=0, ddof=1)
        self.scale_[self.scale_ == 0] = 1.0
        return super().fit(thetas, simulations)


    def transform(self, simulation):
        """
        Compute normalised summary statistics for a single simulation.
        
        Parameters
        ----------
        simulation : array-like
            A single raw simulated dataset, in the same format as the
            elements of ``simulations`` passed to ``fit``.

        Returns
        -------
        s : np.ndarray, shape (n_statistics,)
            Normalised summary statistic vector.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called before ``transform``.
        """
        super().transform(simulation)
        return self._compute_raw(simulation) / self.scale_