"""Extreme-value and interval statistics for the campaign (pure NumPy/SciPy, no Drake)."""

from tether.evt.acer import ACERFit, acer_estimates, acer_extrapolate, fit_acer
from tether.evt.bootstrap import (
    moving_block_indices,
    paired_bootstrap_difference,
    seed_bootstrap_rate,
    seed_bootstrap_statistic,
)
from tether.evt.calibration import (
    auroc,
    auroc_interval,
    ece_bootstrap,
    expected_calibration_error,
    logistic_recalibration,
    reliability_bins,
)
from tether.evt.declustering import runs_decluster, upcrossing_counts
from tether.evt.dependence import (
    chi_empirical,
    chibar_empirical,
    dependence_bootstrap,
    eta_ledford_tawn,
    gaussian_copula_chi,
    gaussian_copula_chibar,
    gaussian_eta,
    pseudo_observations,
    spearman_to_pearson,
)
from tether.evt.gof import anderson_darling_normal, ks_parametric_bootstrap, rayleigh_scale_mle
from tether.evt.gpd import (
    GPDFit,
    exceedance_probability,
    fit_gpd,
    mean_residual_life,
    parameter_stability,
    profile_shape_interval,
    profile_tail_rate_interval,
    select_threshold,
    tail_rate,
)
from tether.evt.hill import hill_estimator, hill_plot, log_slope_fit, tail_index_from_rates
from tether.evt.poisson import censored_upper_bound, poisson_interval, rate_interval
from tether.evt.proportions import newcombe_difference, wilson_interval

__all__ = [
    "ACERFit",
    "GPDFit",
    "acer_estimates",
    "acer_extrapolate",
    "anderson_darling_normal",
    "auroc",
    "auroc_interval",
    "censored_upper_bound",
    "chi_empirical",
    "chibar_empirical",
    "dependence_bootstrap",
    "ece_bootstrap",
    "eta_ledford_tawn",
    "exceedance_probability",
    "expected_calibration_error",
    "fit_acer",
    "fit_gpd",
    "gaussian_copula_chi",
    "gaussian_copula_chibar",
    "gaussian_eta",
    "hill_estimator",
    "hill_plot",
    "ks_parametric_bootstrap",
    "log_slope_fit",
    "logistic_recalibration",
    "mean_residual_life",
    "moving_block_indices",
    "newcombe_difference",
    "paired_bootstrap_difference",
    "parameter_stability",
    "poisson_interval",
    "profile_shape_interval",
    "profile_tail_rate_interval",
    "pseudo_observations",
    "rate_interval",
    "rayleigh_scale_mle",
    "reliability_bins",
    "runs_decluster",
    "seed_bootstrap_rate",
    "seed_bootstrap_statistic",
    "select_threshold",
    "spearman_to_pearson",
    "tail_index_from_rates",
    "tail_rate",
    "upcrossing_counts",
    "wilson_interval",
]
