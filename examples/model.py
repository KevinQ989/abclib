from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for simulator-based models used in ABC inference.

    Subclasses must implement prior sampling, prior density evaluation,
    simulation, and summary statistic definitions. The base class exposes
    a consistent interface consumed by abclib samplers and the validation
    runner.

    Parameters
    ----------
    name : str
        Human-readable name for the model, used in plot titles and
        print output.
    """
    def __init__(self, name):
        self.name = name


    @property
    @abstractmethod
    def prior_bounds(self):
        """
        Bounds of the prior support for each parameter.

        Used by RegressionAdjustment to reflect adjusted samples back
        into the valid region.

        Returns
        -------
        list of (float, float)
            List of (lower, upper) tuples, one per parameter.
        """
        pass


    @abstractmethod
    def prior(self):
        """
        Sample a single parameter vector from the prior.

        Returns
        -------
        theta : np.ndarray, shape (n_params,)
            Parameter vector drawn from the prior.
        """
        pass


    @abstractmethod
    def prior_pdf(self, theta):
        """
        Evaluate the prior density at a single parameter vector.

        Parameters
        ----------
        theta : np.ndarray, shape (n_params,)

        Returns
        -------
        float
            Prior density. Returns 0 if theta is outside the support.
        """
        pass


    @abstractmethod
    def prior_density(self, thetas):
        """
        Vectorised prior density for a batch of parameter vectors.

        Required by SMCABC for importance weight computation.

        Parameters
        ----------
        thetas : np.ndarray, shape (n_thetas, n_params)

        Returns
        -------
        np.ndarray, shape (n_thetas,)
            Prior densities for each input theta.
        """
        pass


    @abstractmethod
    def simulator(self, theta):
        """
        Simulate one dataset under the given parameter vector.

        Parameters
        ----------
        theta : np.ndarray, shape (n_params,)

        Returns
        -------
        y : np.ndarray
            Simulated dataset. Shape is model-dependent.
        """
        pass


    @property
    @abstractmethod
    def SUMMARY_FUNCTIONS(self):
        """
        Dictionary of named scalar summary functions.

        Used to construct a HandCraftedSummary. Each value is a callable
        that takes a simulated dataset and returns a scalar.

        Returns
        -------
        dict[str, callable]
            Mapping of statistic name to function.
        """
        pass


    @abstractmethod
    def H_FUNCTION(self, y):
        """
        Compute the candidate summary vector for SemiAutomaticSummary.

        Parameters
        ----------
        y : np.ndarray
            Simulated dataset.

        Returns
        -------
        h : np.ndarray, shape (n_summaries,)
            Candidate summary statistics.
        """
        pass