"""
Validation runner for abclib models.

Provides run_validation, which takes a Model and Config and runs the
full ABC validation pipeline: pilot fitting, inference, diagnostics,
and plotting. Results are returned as a ValidationResult dataclass.

All methods are optional. Dependencies are resolved automatically:
- regression_adj requires rejection_hc
- sbc_rejection requires rejection_hc
- sbc_adjusted requires regression_adj (and therefore rejection_hc)
- ppc requires regression_adj (and therefore rejection_hc)
- rejection_sa, smc, mcmc, synthetic_likelihood, str are independent
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

import abclib
from abclib.distance import euclidean
from abclib.results import ABCResult, SLResult
from abclib.utils import run_pilot
from abclib.diagnostics.sbc import run_sbc, plot_rank_histogram
from abclib.diagnostics.ppc import run_ppc
from abclib.diagnostics.str import run_str, plot_str_results

from .model import Model
from .config import Config


@dataclass
class ValidationResult:
    """
    Container for the outputs of a full validation run.

    Parameters
    ----------
    results : dict
        Mapping of method name to ABCResult or SLResult.
        Contains only the methods that were actually run.
    diagnostics : dict
        Mapping of diagnostic name to result dict.
        Contains only the diagnostics that were actually run.
        Possible keys: 'sbc_rejection', 'sbc_adjusted',
        'ppc_{fn.__name__}', 'str'.
    config : Config
        The configuration used for this run.
    model_name : str
        Name of the model as returned by model.name.
    """
    results: dict
    diagnostics: dict
    config: Config
    model_name: str


class _AdjustedSampler:
    """
    Wraps RejectionABC with a regression adjustment step.

    Used internally by run_validation to pass an adjusted sampler
    to run_sbc, which expects a single callable sampler.
    """
    def __init__(self, base_sampler, prior_bounds):
        self.base_sampler = base_sampler
        self.prior_bounds = prior_bounds

    def sample(self, s_obs, **kwargs):
        result = self.base_sampler.sample(s_obs, **kwargs)
        adj = abclib.RegressionAdjustment(self.prior_bounds)
        adj.fit(result, s_obs)
        return adj.adjust(result, s_obs)


def _resolve_methods(requested: list) -> list:
    """
    Resolve method dependencies and return the full ordered list to run.

    Parameters
    ----------
    requested : list of str
        Methods requested by the user.

    Returns
    -------
    list of str
        Full list of methods to run, with dependencies added and
        ordered so dependencies always run before dependents.
    """
    resolved = list(requested)
    order = [
        "rejection_hc",
        "rejection_sa",
        "regression_adj",
        "smc",
        "mcmc",
        "synthetic_likelihood",
        "sbc_rejection",
        "sbc_adjusted",
        "ppc",
        "str",
    ]
    
    if "all" in resolved:
        return order

    if "sbc_adjusted" in resolved and "regression_adj" not in resolved:
        resolved.append("regression_adj")
    if "ppc" in resolved and "regression_adj" not in resolved:
        resolved.append("regression_adj")
    if "regression_adj" in resolved and "rejection_hc" not in resolved:
        resolved.append("rejection_hc")
    if "sbc_rejection" in resolved and "rejection_hc" not in resolved:
        resolved.append("rejection_hc")

    return [m for m in order if m in resolved]


def _print_results(results: dict):
    """Print a summary table of posterior means, CIs, and acceptance rates."""
    header = (
        f"{'Method':<30} {'Mean':>20} {'90% CI Lower':>20} "
        f"{'90% CI Upper':>20} {'Acc. Rate':>12}"
    )
    print(header)
    print("-" * len(header))
    for method, result in results.items():
        mean = result.posterior_mean()
        lower, upper = result.credible_interval(alpha=0.90)
        print(
            f"{method:<30} "
            f"{np.array2string(mean, precision=3):>20} "
            f"{np.array2string(lower, precision=3):>20} "
            f"{np.array2string(upper, precision=3):>20} "
            f"{result.acceptance_rate:>12.4f}"
        )


def _plot_results_scatter(results: dict, true_theta: np.ndarray,
                  output_dir: str, model_name: str, parameter_names: list,
                  exact_grid: Optional[tuple] = None):
    """
    Plot posterior samples in a grid, overlaid on exact contours if provided.

    Parameters
    ----------
    results : dict
        Mapping of method name to ABCResult or SLResult.
    true_theta : np.ndarray, shape (n_params,)
        True parameter vector to mark on each histogram.
    output_dir : str
        Directory to save the figure.
    model_name : str
        Used to construct the output filename.
    parameter_names : list of str
        Names of the parameters, used for axis labels.
    exact_grid : tuple, optional
        (T1, T2, prob) from exact_posterior_grid, used to overlay exact
        posterior contours. If None, samples are plotted without contours.
    """
    methods = list(results.items())
    n_methods = len(methods)
    if n_methods == 0:
        return

    n_cols = min(3, n_methods)
    n_rows = int(np.ceil(n_methods / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_methods == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, (label, result) in enumerate(methods):
        ax = axes[idx]
        if exact_grid is not None:
            T1, T2, prob = exact_grid
            ax.contour(T1, T2, prob, levels=10, cmap="Blues")
        ax.scatter(
            result.samples[:, 0], result.samples[:, 1],
            alpha=0.2, s=5, color="orange", label="ABC samples"
        )
        ax.scatter(
            true_theta[0], true_theta[1],
            color="red", marker="+", s=150, zorder=5, label="True θ"
        )
        ax.set_title(f"{label}\nacc={result.acceptance_rate:.4f}")
        ax.set_xlabel(parameter_names[0])
        ax.set_ylabel(parameter_names[1])
        ax.legend(fontsize=7)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    prefix = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{prefix}_posterior_comparison.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved posterior comparison to {path}")


def _plot_results_histogram(results: dict, true_theta: np.ndarray,
                  output_dir: str, model_name: str, parameter_names: list):
    """
    Plot marginal posterior histograms for each parameter and method.

    For models with more than 2 parameters, scatter plots are
    uninformative. This function plots one row per method and one
    column per parameter, with the true parameter marked as a vertical
    line on each histogram.

    Parameters
    ----------
    results : dict
        Mapping of method name to ABCResult or SLResult.
    true_theta : np.ndarray, shape (n_params,)
        True parameter vector to mark on each histogram.
    output_dir : str
        Directory to save the figure.
    model_name : str
        Used to construct the output filename.
    parameter_names : list of str
        Names of the parameters, used for axis labels.
    """
    if not results:
        return

    method_names = list(results.keys())
    n_methods = len(method_names)
    n_params = results[method_names[0]].samples.shape[1]

    fig, axes = plt.subplots(
        n_methods, n_params,
        figsize=(4 * n_params, 3 * n_methods),
        squeeze=False
    )

    for row, method in enumerate(method_names):
        result = results[method]
        for col in range(n_params):
            ax = axes[row, col]
            ax.hist(
                result.samples[:, col],
                bins=30,
                density=True,
                color="steelblue",
                alpha=0.7,
                edgecolor="none"
            )
            ax.axvline(
                true_theta[col],
                color="red",
                linewidth=1.5,
                linestyle="--",
                label="True θ" if col == 0 else None
            )
            if row == 0:
                ax.set_title(f"{parameter_names[col]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    method.replace("_", " "),
                    fontsize=8,
                    rotation=45,
                    ha="right",
                    labelpad=40
                )
            ax.set_xlabel("")
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{model_name} — Marginal Posteriors\n"
        f"(acc rates: "
        + ", ".join(
            f"{m}={results[m].acceptance_rate:.3f}"
            for m in method_names
        ) + ")",
        fontsize=9
    )

    plt.tight_layout()
    prefix = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{prefix}_posterior_comparison.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved posterior comparison to {path}")


def run_validation(
    model: Model,
    config: Config,
    true_theta: np.ndarray,
    observed_data: np.ndarray,
    exact_grid: Optional[tuple] = None,
) -> ValidationResult:
    """
    Run the ABC validation pipeline for a given model and config.

    All methods are optional. Dependencies are resolved automatically.
    Only the methods listed in config.methods are run. Pilot simulations
    and summary statistic fitting always run regardless of which methods
    are requested, since all methods share the same fitted statistics.

    Parameters
    ----------
    model : Model
        A Model subclass instance providing prior, simulator, and
        summary statistic definitions.
    config : Config
        Full configuration for the validation run.
    true_theta : np.ndarray, shape (n_params,)
        True parameter vector used to generate observed_data.
    observed_data : np.ndarray
        Observed dataset generated from true_theta.
    exact_grid : tuple, optional
        (T1, T2, prob) from exact_posterior_grid, used to overlay exact
        posterior contours on the posterior comparison plot. If None,
        samples are plotted without contours.

    Returns
    -------
    ValidationResult
        Contains all ABCResult/SLResult objects and diagnostic outputs
        for the methods that were run.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    prefix = (
        model.name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    results = {}
    diagnostics = {}
    methods = _resolve_methods(config.methods)

    # ------------------------------------------------------------------
    # Pilot simulations and summary statistic fitting
    # Always runs — all methods share the same fitted statistics.
    # ------------------------------------------------------------------
    print(f"\n[{model.name}] Running pilot ({config.n_pilot} simulations)...")
    pilot_thetas, pilot_sims = run_pilot(
        model.prior, model.simulator, n_pilot=config.n_pilot
    )
    print(f"  Pilot complete: {len(pilot_thetas)} draws.")

    print("\nFitting summary statistics...")
    hc_stat = abclib.HandCraftedSummary(model.SUMMARY_FUNCTIONS)
    hc_stat.fit(pilot_thetas, pilot_sims)
    s_obs_hc = hc_stat.transform(observed_data)
    print("  HandCraftedSummary fitted.")

    sa_stat = abclib.SemiAutomaticSummary(model.H_FUNCTION)
    sa_stat.fit(pilot_thetas, pilot_sims)
    s_obs_sa = sa_stat.transform(observed_data)
    print("  SemiAutomaticSummary fitted.")

    # ------------------------------------------------------------------
    # Rejection ABC (HandCrafted)
    # ------------------------------------------------------------------
    if "rejection_hc" in methods:
        print("\nRunning Rejection ABC (HandCrafted)...")
        rejection_hc = abclib.RejectionABC(
            prior=model.prior,
            simulator=model.simulator,
            summary_statistic=hc_stat,
            distance=euclidean
        ).sample(
            s_obs_hc,
            n_simulations=config.rejection.n_simulations,
            q=config.rejection.q
        )
        results["rejection_hc"] = rejection_hc
        print(f"  Done. Accepted {len(rejection_hc.samples)} samples "
              f"(acc={rejection_hc.acceptance_rate:.4f}).")

    # ------------------------------------------------------------------
    # Rejection ABC (Semi-Automatic)
    # ------------------------------------------------------------------
    if "rejection_sa" in methods:
        print("\nRunning Rejection ABC (Semi-Automatic)...")
        rejection_sa = abclib.RejectionABC(
            prior=model.prior,
            simulator=model.simulator,
            summary_statistic=sa_stat,
            distance=euclidean
        ).sample(
            s_obs_sa,
            n_simulations=config.rejection.n_simulations,
            q=config.rejection.q
        )
        results["rejection_sa"] = rejection_sa
        print(f"  Done. Accepted {len(rejection_sa.samples)} samples "
              f"(acc={rejection_sa.acceptance_rate:.4f}).")

    # ------------------------------------------------------------------
    # Regression Adjustment
    # ------------------------------------------------------------------
    if "regression_adj" in methods:
        print("\nRunning Regression Adjustment...")
        reg_adj = abclib.RegressionAdjustment(model.prior_bounds)
        reg_adj.fit(results["rejection_hc"], s_obs_hc)
        regression_adj = reg_adj.adjust(results["rejection_hc"], s_obs_hc)
        results["regression_adj"] = regression_adj
        print(f"  Done. Adjusted {len(regression_adj.samples)} samples.")

    # ------------------------------------------------------------------
    # SMC-ABC
    # ------------------------------------------------------------------
    if "smc" in methods:
        print("\nRunning SMC-ABC...")
        smc = abclib.SMCABC(
            prior=model.prior,
            simulator=model.simulator,
            summary_statistic=hc_stat,
            distance=euclidean,
            prior_density=model.prior_density
        ).sample(
            s_obs_hc,
            M=config.smc.M,
            T=config.smc.T,
            q=config.smc.q
        )
        results["smc"] = smc
        print(f"  Done. Final epsilon={smc.epsilon:.4f} "
              f"(acc={smc.acceptance_rate:.4f}).")

    # ------------------------------------------------------------------
    # MCMC-ABC
    # ------------------------------------------------------------------
    if "mcmc" in methods:
        print("\nRunning MCMC-ABC...")
        mcmc = abclib.MCMCABC(
            prior=model.prior,
            simulator=model.simulator,
            summary_statistic=hc_stat,
            distance=euclidean,
            prior_pdf=model.prior_pdf,
            proposal_std=config.mcmc.proposal_std
        ).sample(
            s_obs_hc,
            n_samples=config.mcmc.n_samples,
            epsilon=config.mcmc.epsilon
        )
        results["mcmc"] = mcmc
        print(f"  Done. {len(mcmc.samples)} chain states "
              f"(acc={mcmc.acceptance_rate:.4f}).")

    # ------------------------------------------------------------------
    # Synthetic Likelihood
    # ------------------------------------------------------------------
    if "synthetic_likelihood" in methods:
        print("\nRunning Synthetic Likelihood (this may take a while)...")
        sl = abclib.SyntheticLikelihood(
            prior=model.prior,
            simulator=model.simulator,
            summary_statistic=hc_stat,
            prior_pdf=model.prior_pdf,
            proposal_std=config.synthetic_likelihood.proposal_std
        ).sample(
            s_obs_hc,
            n_simulations=config.synthetic_likelihood.n_simulations,
            M=config.synthetic_likelihood.M
        )
        results["synthetic_likelihood"] = sl
        print(f"  Done. {len(sl.samples)} samples "
              f"(acc={sl.acceptance_rate:.4f}). "
              f"Total simulator calls: {sl.n_simulations:,}.")

    # ------------------------------------------------------------------
    # Posterior comparison plot
    # Only runs if any inference results exist.
    # ------------------------------------------------------------------
    inference_keys = [
        "rejection_hc", "rejection_sa", "regression_adj",
        "smc", "mcmc", "synthetic_likelihood"
    ]
    inference_results = {
        k: v for k, v in results.items() if k in inference_keys
    }
    if inference_results:
        print("\nPlotting posterior comparison...")
        if len(true_theta) == 2:
            _plot_results_scatter(
                results=inference_results,
                true_theta=true_theta,
                output_dir=config.output_dir,
                model_name=model.name,
                parameter_names=model.parameter_names,
                exact_grid=exact_grid
            )
        else:
            _plot_results_histogram(
                results=inference_results,
                true_theta=true_theta,
                output_dir=config.output_dir,
                model_name=model.name,
                parameter_names=model.parameter_names
            )
        print("\n--- Results Summary ---")
        _print_results(inference_results)

    # ------------------------------------------------------------------
    # PPC on regression_adj
    # ------------------------------------------------------------------
    if "ppc" in methods:
        print("\nRunning PPC...")
        for fn in config.ppc.test_statistics:
            name = fn.__name__
            print(f"  PPC ({name})...")
            ppc = run_ppc(
                result=results["regression_adj"],
                simulator=model.simulator,
                y_obs=observed_data,
                test_statistic=fn,
                n_samples=config.ppc.n_samples
            )
            diagnostics[f"ppc_{name}"] = ppc
            print(f"    p-value: {ppc['p_value']:.3f} | "
                  f"t_obs: {ppc['t_obs']:.4f} | "
                  f"t_rep mean: {np.mean(ppc['t_rep']):.4f}")

    # ------------------------------------------------------------------
    # SBC on rejection HC
    # ------------------------------------------------------------------
    if "sbc_rejection" in methods:
        print("\nRunning SBC on rejection HC...")
        sbc = run_sbc(
            sampler=abclib.RejectionABC(
                prior=model.prior,
                simulator=model.simulator,
                summary_statistic=hc_stat,
                distance=euclidean
            ),
            simulator=model.simulator,
            prior=model.prior,
            n_trials=config.sbc.n_trials,
            L=config.sbc.L,
            summary_statistic=hc_stat,
            n_simulations=config.rejection.n_simulations,
            q=config.rejection.q
        )
        diagnostics["sbc_rejection"] = sbc
        print(f"  Done. KS p-values: {sbc['ks_pvalue'].round(3)}")
        plot_rank_histogram(
            sbc,
            n_bins=20,
            output_dir=config.output_dir,
            filename=f"{prefix}_sbc_rank_histogram_rejection.png",
            parameter_names=model.parameter_names
        )

    # ------------------------------------------------------------------
    # SBC on regression-adjusted
    # ------------------------------------------------------------------
    if "sbc_adjusted" in methods:
        print("\nRunning SBC on regression-adjusted...")
        sbc_adj = run_sbc(
            sampler=_AdjustedSampler(
                base_sampler=abclib.RejectionABC(
                    prior=model.prior,
                    simulator=model.simulator,
                    summary_statistic=hc_stat,
                    distance=euclidean
                ),
                prior_bounds=model.prior_bounds
            ),
            simulator=model.simulator,
            prior=model.prior,
            n_trials=config.sbc.n_trials,
            L=config.sbc.L,
            summary_statistic=hc_stat,
            n_simulations=config.rejection.n_simulations,
            q=config.rejection.q
        )
        diagnostics["sbc_adjusted"] = sbc_adj
        print(f"  Done. KS p-values: {sbc_adj['ks_pvalue'].round(3)}")
        plot_rank_histogram(
            sbc_adj,
            n_bins=20,
            output_dir=config.output_dir,
            filename=f"{prefix}_sbc_rank_histogram_reg_adj.png",
            parameter_names=model.parameter_names
        )

    # ------------------------------------------------------------------
    # STR
    # ------------------------------------------------------------------
    if "str" in methods:
        print("\nRunning STR...")
        bounds = model.prior_bounds
        axes_grids = [
            np.linspace(
                lo + (hi - lo) * 0.1,
                hi - (hi - lo) * 0.1,
                config.str_.n_grid_points
            )
            for lo, hi in bounds
        ]
        mesh = np.array(np.meshgrid(*axes_grids)).T.reshape(-1, len(bounds))
        theta_grid = np.array([
            theta for theta in mesh
            if model.prior_pdf(theta) > 0
        ])
        print(f"  Grid size: {len(theta_grid)} points.")
        str_result = run_str(
            sampler=abclib.RejectionABC(
                prior=model.prior,
                simulator=model.simulator,
                summary_statistic=hc_stat,
                distance=euclidean
            ),
            simulator=model.simulator,
            theta_grid=theta_grid,
            summary_statistic=hc_stat,
            credible_mass=config.str_.credible_mass,
            n_simulations=config.rejection.n_simulations,
            q=config.rejection.q
        )
        diagnostics["str"] = str_result
        print(f"  Done. Coverage per parameter: "
              f"{str_result['coverage'].round(3)}")
        plot_str_results(
            str_result,
            output_dir=config.output_dir,
            filename=f"{prefix}_str_results.png",
            parameter_names=model.parameter_names
        )

    print(f"\n[{model.name}] Validation complete.")
    return ValidationResult(
        results=results,
        diagnostics=diagnostics,
        config=config,
        model_name=model.name
    )