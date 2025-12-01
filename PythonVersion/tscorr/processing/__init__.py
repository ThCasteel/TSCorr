"""Processing submodule for TSCorr."""

from .displacement import (
    DisplacementPair,
    DisplacementStack,
    compute_displacement_magnitude,
    filter_displacement_by_angle,
    load_displacement_pair,
    parse_correlation_filename,
    resample_displacement,
)
from .velocity import (
    FitModel,
    VelocityResult,
    estimate_velocity_dask,
    estimate_velocity_grid,
    fit_constant_displacement,
    fit_constant_linear,
    fit_linear_velocity,
)
from .coregistration import (
    coregister_displacement_stack,
    save_coregistration_results,
)

__all__ = [
    "DisplacementPair",
    "DisplacementStack",
    "compute_displacement_magnitude",
    "filter_displacement_by_angle",
    "load_displacement_pair",
    "parse_correlation_filename",
    "resample_displacement",
    "FitModel",
    "VelocityResult",
    "estimate_velocity_dask",
    "estimate_velocity_grid",
    "fit_constant_displacement",
    "fit_constant_linear",
    "fit_linear_velocity",
    "coregister_displacement_stack",
    "save_coregistration_results",
]
