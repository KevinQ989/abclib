from abclib.results import ABCResult
import numpy as np

class RegressionAdjustment:
    """
    Post-processing correction for tolerance-induced bias in rejection ABC.

    Fits a local linear regression of accepted parameter draws on the
    deviation of their summary vectors from the observed summaries. The
    fitted coefficients are used to shift each accepted draw towards the
    value it would have taken had its summaries exactly matched s_obs.

    Adjusted samples that fall outside the prior support are reflected
    back into the valid region.

    Parameters
    ----------
    prior_bounds : list of (float, float)
        List of (lower, upper) tuples, one per parameter, defining the
        prior support for boundary reflection.
    """
    def __init__(self, prior_bounds):
        self.prior_bounds = prior_bounds
        self.intercepts_ = None
        self.coeffs_ = None
        self.scale_ = None
    

    def fit(self, result, s_obs):
        """
        Fit a linear regression of accepted parameters on summary deviations.

        Parameters
        ----------
        result : ABCResult
            Result from rejection ABC containing accepted samples and
            their associated summary vectors.
        s_obs : np.ndarray, shape (n_statistics,)
            Summary statistics of the observed data.

        Returns
        -------
        self
            Returns the instance to allow method chaining.
        """
        X = result.summaries - s_obs
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        X = X / self.scale_

        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        coeffs = np.linalg.lstsq(X_design, result.samples, rcond=None)[0]
        self.intercepts_ = coeffs[0]
        self.coeffs_ = coeffs[1:]
        return self
    

    def adjust(self, result, s_obs):
        """
        Apply the regression correction and return an adjusted ABCResult.

        Parameters
        ----------
        result : ABCResult
            Result from rejection ABC containing accepted samples and
            their associated summary vectors.
        s_obs : np.ndarray, shape (n_statistics,)
            Summary statistics of the observed data.

        Returns
        -------
        ABCResult
            New result object with adjusted samples. Distances, epsilon,
            and simulator call count are inherited from the input result.

        Raises
        ------
        RuntimeError
            If fit has not been called before adjust.
        """
        if self.intercepts_ is None:
            raise RuntimeError("Call fit() before adjust().")

        X = (result.summaries - s_obs) / self.scale_
        adjustments = X @ self.coeffs_
        adjusted_samples = result.samples - adjustments
        adjusted_samples = self._reflect(adjusted_samples)

        return ABCResult(
            samples = adjusted_samples,
            distances = result.distances,
            summaries = result.summaries,
            n_simulations = result.n_simulations,
            epsilon = result.epsilon,
            summary_statistic = result.summary_statistic,
        )
    

    def _reflect(self, values):
        """
        Reflect values outside prior bounds back into the valid region.

        Applies repeated reflection until all values are within bounds.
        For values very far outside the bounds, clips after 10 iterations
        as a fallback.

        Parameters
        ----------
        values : np.ndarray, shape (n_samples, n_params)
            Array of parameter vectors to reflect.
        
        Returns
        -------
        np.ndarray, shape (n_samples, n_params)
            Reflected parameter vectors within prior bounds.
        """
        reflected = np.copy(values)
        for j in range(reflected.shape[1]):
            lower, upper = self.prior_bounds[j]
            for _ in range(10):
                too_low = reflected[:, j] < lower
                too_high = reflected[:, j] > upper
                if not (too_low.any() or too_high.any()):
                    break
                reflected[:, j] = np.where(
                    too_low, 2 * lower - reflected[:, j], reflected[:, j]
                )
                reflected[:, j] = np.where(
                    too_high, 2 * upper - reflected[:, j], reflected[:, j]
                )
            reflected[:, j] = np.clip(reflected[:, j], lower, upper)
        return reflected