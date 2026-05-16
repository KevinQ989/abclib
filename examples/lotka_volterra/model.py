"""
Stochastic Lotka-Volterra predator-prey model for abclib.

Provides a prior sampler, Euler-Maruyama simulator with lognormal
observation noise, and summary statistic functions for a four-parameter
predator-prey model. Used as the primary intractable case study in the
abclib validation pipeline.
"""
from ..model import Model
import numpy as np


class LotkaVolterra(Model):
    """
    Stochastic Lotka-Volterra predator-prey model.

    The deterministic skeleton is:
        dx/dt = alpha * x - beta * x * y
        dy/dt = delta * x * y - gamma * y

    where x is the prey population and y is the predator population.
    The simulator integrates these ODEs using the Euler-Maruyama method
    and adds lognormal observation noise at each time step.

    The likelihood is intractable because the transition density of the
    stochastic process has no closed form, and the lognormal observation
    noise requires integrating over the latent trajectory.

    Parameters
    ----------
    T : int, optional
        Number of observed time steps. Default is 100.
    x0 : float, optional
        Initial prey population. Default is 50.
    y0 : float, optional
        Initial predator population. Default is 20.
    dt : float, optional
        Euler-Maruyama step size. Default is 0.1.
    sigma : float, optional
        Standard deviation of lognormal observation noise. Default is 0.1.
    """
    def __init__(self, T=100, x0=50, y0=20, dt=0.1, sigma=0.1):
        super().__init__(name="Lotka-Volterra")
        self.T = T
        self.x0 = x0
        self.y0 = y0
        self.dt = dt
        self.sigma = sigma


    @property
    def prior_bounds(self):
        """
        Bounds of the uniform prior support for each parameter.

        Bounds are chosen to cover ecologically plausible parameter
        ranges while keeping the prior relatively informative.

        Returns
        -------
        list of (float, float)
            [(0.1, 1.0), (0.01, 0.5), (0.01, 0.5), (0.1, 1.0)]
            for alpha, beta, delta, gamma respectively.
        """
        return [
            (0.1, 1.0),   # alpha: prey growth rate
            (0.01, 0.5),  # beta:  predation rate
            (0.01, 0.5),  # delta: predator reproduction efficiency
            (0.1, 1.0),   # gamma: predator mortality rate
        ]


    def prior(self):
        """
        Sample a single parameter vector from the uniform prior.

        Each parameter is sampled independently from its marginal
        uniform distribution defined by prior_bounds.

        Returns
        -------
        theta : np.ndarray, shape (4,)
            Parameter vector [alpha, beta, delta, gamma].
        """
        return np.array([
            np.random.uniform(*self.prior_bounds[i])
            for i in range(4)
        ])


    def prior_pdf(self, theta):
        """
        Evaluate the uniform prior density at a single parameter vector.

        Returns the product of four independent uniform densities, or 0
        if any parameter falls outside its bounds.

        Parameters
        ----------
        theta : np.ndarray, shape (4,)
            Parameter vector [alpha, beta, delta, gamma].

        Returns
        -------
        float
            Prior density at theta. Returns 0.0 if theta is out of bounds.
        """
        for i, (lo, hi) in enumerate(self.prior_bounds):
            if not (lo <= theta[i] <= hi):
                return 0.0
        density = 1.0
        for lo, hi in self.prior_bounds:
            density /= (hi - lo)
        return density


    def prior_density(self, thetas):
        """
        Vectorised prior density for a batch of parameter vectors.

        Required by SMCABC for importance weight computation.

        Parameters
        ----------
        thetas : np.ndarray, shape (n_thetas, 4)
            Array of parameter vectors to evaluate.

        Returns
        -------
        np.ndarray, shape (n_thetas,)
            Prior densities for each input theta.
        """
        densities = np.zeros(thetas.shape[0])
        for i, theta in enumerate(thetas):
            densities[i] = self.prior_pdf(theta)
        return densities


    def simulator(self, theta):
        """
        Simulate one observed trajectory under the given parameters.

        Integrates the Lotka-Volterra ODEs using Euler-Maruyama and adds
        independent lognormal observation noise at each time step. Populations
        are clipped at 1e-6 to prevent numerical extinction.

        Parameters
        ----------
        theta : np.ndarray, shape (4,)
            Parameter vector [alpha, beta, delta, gamma].

        Returns
        -------
        observations : np.ndarray, shape (T, 2)
            Observed prey (column 0) and predator (column 1) populations
            at each time step, including observation noise.
        """
        alpha, beta, delta, gamma = theta
        x, y = float(self.x0), float(self.y0)
        observations = np.zeros((self.T, 2))

        for t in range(self.T):
            dx = (alpha * x - beta * x * y) * self.dt
            dy = (delta * x * y - gamma * y) * self.dt
            x = np.clip(x + dx, 1e-6, 1e6)
            y = np.clip(y + dy, 1e-6, 1e6)
            observations[t, 0] = x * np.exp(np.random.normal(0, self.sigma))
            observations[t, 1] = y * np.exp(np.random.normal(0, self.sigma))

        return observations


    def _cross_correlation(self, data):
        """
        Compute the cross-correlation between log prey and log predator.

        A negative value is characteristic of predator-prey cycles, where
        predator peaks follow prey peaks with a phase lag.

        Parameters
        ----------
        data : np.ndarray, shape (T, 2)
            Observed prey and predator populations.

        Returns
        -------
        float
            Pearson correlation between log x and log y. Returns 0.0 if
            either series is constant.
        """
        log_x = np.log(data[:, 0] + 1e-8)
        log_y = np.log(data[:, 1] + 1e-8)
        if np.std(log_x) == 0 or np.std(log_y) == 0:
            return 0.0
        return float(np.corrcoef(log_x, log_y)[0, 1])


    def _oscillation_frequency(self, data):
        """
        Estimate the dominant oscillation frequency of the prey population.

        Computes the FFT of the demeaned log prey series and returns the
        frequency with the largest amplitude among positive frequencies.
        The linearised Lotka-Volterra system oscillates at approximately
        sqrt(alpha * gamma), making this statistic sensitive to both
        prey growth and predator mortality rates.

        Parameters
        ----------
        data : np.ndarray, shape (T, 2)
            Observed prey and predator populations.

        Returns
        -------
        float
            Dominant oscillation frequency in cycles per time step.
            Returns 0.0 if no positive frequencies are found.
        """
        log_x = np.log(data[:, 0] + 1e-8)
        prey_fft = np.fft.fft(log_x - np.mean(log_x))
        freqs = np.fft.fftfreq(len(data))
        mask = freqs > 0
        if not np.any(mask):
            return 0.0
        dominant_freq = freqs[mask][np.argmax(np.abs(prey_fft[mask]))]
        return float(dominant_freq)


    @property
    def SUMMARY_FUNCTIONS(self):
        """
        Dictionary of named scalar summary functions for HandCraftedSummary.

        All statistics are computed on log-transformed populations following
        Wood (2010), since population counts span several orders of magnitude.
        Each statistic targets a different aspect of the predator-prey dynamics.

        Returns
        -------
        dict[str, callable]
            Keys: 'mean_log_x', 'mean_log_y', 'var_log_x', 'var_log_y',
            'corr_log_xy', 'oscillation_freq'.
            Each callable takes data of shape (T, 2) and returns a scalar.
        """
        return {
            "mean_log_x":      lambda data: np.mean(np.log(data[:, 0] + 1e-8)),
            "mean_log_y":      lambda data: np.mean(np.log(data[:, 1] + 1e-8)),
            "var_log_x":       lambda data: np.var(np.log(data[:, 0] + 1e-8)),
            "var_log_y":       lambda data: np.var(np.log(data[:, 1] + 1e-8)),
            "corr_log_xy":     lambda data: self._cross_correlation(data),
            "oscillation_freq": lambda data: self._oscillation_frequency(data),
        }


    def H_FUNCTION(self, data):
        """
        Compute candidate summary statistics for SemiAutomaticSummary.

        Computes all six summary statistics on log-transformed populations.
        This vector serves as the input to the pilot regression in
        SemiAutomaticSummary.

        Parameters
        ----------
        data : np.ndarray, shape (T, 2)
            Observed prey and predator populations.

        Returns
        -------
        h : np.ndarray, shape (6,)
            Summary statistic vector: [mean_log_x, mean_log_y,
            var_log_x, var_log_y, corr_log_xy, oscillation_freq].
        """
        log_x = np.log(data[:, 0] + 1e-8)
        log_y = np.log(data[:, 1] + 1e-8)
        return np.array([
            np.mean(log_x),
            np.mean(log_y),
            np.var(log_x),
            np.var(log_y),
            self._cross_correlation(data),
            self._oscillation_frequency(data),
        ])