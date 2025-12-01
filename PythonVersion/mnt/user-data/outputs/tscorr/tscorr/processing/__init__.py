"""
Processing modules for TSCorr.

Provides displacement loading, filtering, coregistration, and
velocity estimation functionality.
"""

from .displacement import (
    DisplacementPair,
    DisplacementStack,
    parse_correlation_filename,
    load_displacement_pair,
    filter_displacement_by_angle,
    compute_displacement_magnitude,
    resample_displacement,
)

from .velocity import (
    FitModel,
    VelocityResult,
    fit_linear_velocity,
    fit_constant_displacement,
    fit_constant_linear,
    estimate_velocity_grid,
    estimate_velocity_dask,
)

from .coregistration import (
    CoregistrationResult,
    compute_offsets,
    apply_coregistration,
    coregister_displacement_stack,
    save_coregistration_results,
    load_coregistration_results,
)

__all__ = [
    # Displacement
    "DisplacementPair",
    "DisplacementStack",
    "parse_correlation_filename",
    "load_displacement_pair",
    "filter_displacement_by_angle",
    "compute_displacement_magnitude",
    "resample_displacement",
    # Velocity
    "FitModel",
    "VelocityResult",
    "fit_linear_velocity",
    "fit_constant_displacement",
    "fit_constant_linear",
    "estimate_velocity_grid",
    "estimate_velocity_dask",
    # Coregistration
    "CoregistrationResult",
    "compute_offsets",
    "apply_coregistration",
    "coregister_displacement_stack",
    "save_coregistration_results",
    "load_coregistration_results",
]
