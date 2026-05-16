"""
Configuration dataclasses for abclib validation runs.

Each method has its own config dataclass with method-specific parameters.
These are collected into a top-level Config object passed to run_validation.
"""
from dataclasses import dataclass, field
from typing import List, Callable
import numpy as np


@dataclass
class RejectionConfig:
    """
    Configuration for Rejection ABC.

    Parameters
    ----------
    n_simulations : int
        Total number of simulator calls. Default is 10,000.
    q : float
        Acceptance quantile. Default is 0.05.
    """
    n_simulations: int = 10_000
    q: float = 0.05


@dataclass
class SMCConfig:
    """
    Configuration for SMC-ABC.

    Parameters
    ----------
    M : int
        Number of particles. Default is 500.
    T : int
        Number of SMC stages. Default is 5.
    q : float
        Quantile for adaptive tolerance schedule. Default is 0.05.
    """
    M: int = 500
    T: int = 5
    q: float = 0.05


@dataclass
class MCMCConfig:
    """
    Configuration for MCMC-ABC.

    Parameters
    ----------
    n_samples : int
        Number of MCMC chain states to collect. Default is 10,000.
    epsilon : float
        Fixed tolerance threshold for the epsilon-gate. Default is 0.05.
    proposal_std : float
        Standard deviation of the isotropic Gaussian proposal kernel.
        Default is 0.3.
    """
    n_samples: int = 10_000
    epsilon: float = 0.05
    proposal_std: float = 0.3


@dataclass
class SyntheticLikelihoodConfig:
    """
    Configuration for Synthetic Likelihood MCMC.

    Parameters
    ----------
    n_simulations : int
        Number of MCMC steps. Default is 10,000.
    M : int
        Number of simulator calls per step to estimate the synthetic
        likelihood. Default is 100.
    proposal_std : float
        Standard deviation of the isotropic Gaussian proposal kernel.
        Default is 0.1.
    """
    n_simulations: int = 10_000
    M: int = 100
    proposal_std: float = 0.1


@dataclass
class SBCConfig:
    """
    Configuration for Simulation-Based Calibration.

    Parameters
    ----------
    n_trials : int
        Number of (theta*, y*) pairs to generate. Default is 500.
    L : int
        Number of posterior draws per trial used for rank computation.
        Default is 100.
    """
    n_trials: int = 500
    L: int = 100


@dataclass
class PPCConfig:
    """
    Configuration for Posterior Predictive Checks.

    Parameters
    ----------
    n_samples : int
        Number of posterior draws to use for PPC. Default is 1,000.
    test_statistics : list of callable
        List of test statistics T(y) -> float to evaluate. Each must
        accept a simulated dataset in the same format as the model
        simulator output and return a scalar. Default is [np.mean, np.var].
    """
    n_samples: int = 1_000
    test_statistics: List[Callable] = field(
        default_factory=lambda: [np.mean, np.var]
    )


@dataclass
class STRConfig:
    """
    Configuration for Synthetic Truth Recovery.

    Parameters
    ----------
    n_grid_points : int
        Number of grid points per parameter axis. The full grid is
        constructed from model.prior_bounds and filtered to the prior
        support. Default is 6.
    credible_mass : float
        Credible interval mass for coverage check. Default is 0.90.
    """
    n_grid_points: int = 6
    credible_mass: float = 0.90


@dataclass
class Config:
    """
    Top-level configuration for a validation run.

    The baseline methods (rejection HC and regression adjustment) are
    always run. Additional methods are controlled by the methods list.

    Parameters
    ----------
    methods : list of str
        Additional methods to run beyond the baseline. Valid entries:
        'rejection_sa', 'smc', 'mcmc', 'synthetic_likelihood'.
    n_pilot : int
        Number of pilot simulations for fitting summary statistics.
        Default is 2,000.
    seed : int
        Random seed for reproducibility. Default is 0.
    output_dir : str
        Directory to save plots. Default is 'plots'.
    rejection : RejectionConfig
        Configuration for rejection ABC (used for baseline and optional
        rejection_sa).
    smc : SMCConfig
        Configuration for SMC-ABC.
    mcmc : MCMCConfig
        Configuration for MCMC-ABC.
    synthetic_likelihood : SyntheticLikelihoodConfig
        Configuration for synthetic likelihood MCMC.
    sbc : SBCConfig
        Configuration for SBC diagnostics.
    ppc : PPCConfig
        Configuration for PPC diagnostics.
    str_ : STRConfig
        Configuration for STR diagnostics.
    """
    methods: List[str] = field(default_factory=list)
    n_pilot: int = 2_000
    seed: int = 0
    output_dir: str = "plots"
    rejection: RejectionConfig = field(default_factory=RejectionConfig)
    smc: SMCConfig = field(default_factory=SMCConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    synthetic_likelihood: SyntheticLikelihoodConfig = field(
        default_factory=SyntheticLikelihoodConfig
    )
    sbc: SBCConfig = field(default_factory=SBCConfig)
    ppc: PPCConfig = field(default_factory=PPCConfig)
    str_: STRConfig = field(default_factory=STRConfig)