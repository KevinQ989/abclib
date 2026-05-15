"""
abclib: A Python library for Approximate Bayesian Computation (ABC).

Quick Start
-----------
>>> from abclib import RejectionABC, HandCraftedSummary
>>> from abclib.distance import euclidean
>>> from abclib.utils import run_pilot, select_epsilon

Samplers
--------
- RejectionABC: Basic ABC rejection sampler.
- SMCABC: Sequential Monte Carlo ABC sampler.
- MCMCABC: Markov Chain Monte Carlo ABC sampler.

Summary Statistics
------------------
- HandCraftedSummary: User-defined summary statistics based on domain knowledge.
- SemiAutomaticSummary: Automatically learned summary statistics using regression.

Post Processing
-------------------
- RegressionAdjustment: Adjust posterior samples using local linear regression.

Diagnostics
-------------------
- run_ppc: Assess model fit using posterior predictive checks.
- run_sbc: Evaluate the calibration of ABC posteriors.
- run_str: Test the ability of ABC to recover known parameters.
"""

from .samplers import RejectionABC, SMCABC, MCMCABC
from .statistics import HandCraftedSummary, SemiAutomaticSummary
from .postprocessing import RegressionAdjustment
from .synthetic_likelihood import SyntheticLikelihood
from .diagnostics import run_ppc, run_sbc, run_str, plot_rank_histogram, plot_str_results


__all__ = [
    "RejectionABC",
    "SMCABC",
    "MCMCABC",
    "HandCraftedSummary",
    "SemiAutomaticSummary",
    "RegressionAdjustment",
    "SyntheticLikelihood",
    "run_ppc",
    "run_sbc",
    "run_str",
    "plot_rank_histogram",
    "plot_str_results"
]

__version__ = "0.1.0"