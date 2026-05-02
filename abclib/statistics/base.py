from abc import ABC, abstractmethod
import numpy as np

class BaseSummaryStatistic(ABC):
    """
    Abstract base class for all summary statistic transformations.
 
    Follows a fit / transform interface, mirroring the scikit-learn
    convention. ``fit`` is called once on pilot simulations before
    inference begins; ``transform`` is called inside the ABC loop for
    every proposed simulation.
 
    The separation matters because some summary statistics must learn
    parameters from data before they can be applied:
 
    - :class:`~abclib.statistics.HandCraftedSummary` uses ``fit`` to
      compute prior-predictive standard deviations for normalisation.
    - :class:`~abclib.statistics.SemiAutomaticSummary` uses ``fit`` to
      run a pilot regression and learn optimal linear weights.
 
    In both cases ``transform`` presents the same interface to the
    sampler, so statistics are interchangeable without modifying sampler
    code.
    """

    @abstractmethod
    def fit(self, thetas, simulations):
        """
        Learn any parameters required for the transformation.
 
        Called once on pilot simulations before inference. For
        hand-crafted statistics this computes normalisation scaling;
        for semi-automatic ABC this fits the pilot regression.
 
        Parameters
        ----------
        thetas : np.ndarray, shape (n_pilot, n_params)
            Parameter vectors from the pilot run.
        simulations : list of length n_pilot
            Raw simulated datasets from the pilot run.
            
        Returns
        -------
        self
            Returns the instance to allow method chaining, e.g.
            ``stat.fit(thetas, sims).transform(y_obs)``.
        """
        pass

    @abstractmethod
    def transform(self, simulation):
        """
        Compute the summary statistic vector for a single simulation.
 
        Called inside the ABC loop for every proposed parameter draw.
        Must return a vector of fixed length regardless of the input.
 
        Parameters
        ----------
        simulation : array-like
            A single raw simulated dataset, in the same format as the
            elements of ``simulations`` passed to ``fit``.
 
        Returns
        -------
        s : np.ndarray, shape (n_statistics,)
            Summary statistic vector, normalised and ready for distance
            comparison.
 
        Raises
        ------
        RuntimeError
            If ``fit`` has not been called before ``transform``.
        """
        pass


    def fit_transform(self, thetas, simulations):
        """
        Fit on pilot simulations and transform them in one step.
 
        Convenience method equivalent to calling ``fit`` then applying
        ``transform`` to each element of ``simulations``.
 
        Parameters
        ----------
        thetas : np.ndarray, shape (n_pilot, n_params)
            Parameter vectors from the pilot run.
        simulations : list of length n_pilot
            Raw simulated datasets from the pilot run.
 
        Returns
        -------
        S : np.ndarray, shape (n_pilot, n_statistics)
            Summary statistic matrix for all pilot simulations.
        """
        self.fit(thetas, simulations)
        return np.array([self.transform(s) for s in simulations])