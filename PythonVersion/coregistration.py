"""
Coregistration of displacement maps.

Estimates and removes systematic offsets between displacement measurements
using stable reference areas (rock, non-deforming regions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..config import TSCorrConfig


@dataclass
class CoregistrationResult:
    """Result of coregistration for a displacement pair."""
    pair_index: int
    date_master: str
    date_secondary: str
    offset_x: float         # Mean offset in x
    offset_y: float         # Mean offset in y
    offset_x_std: float     # Std of offset in x
    offset_y_std: float     # Std of offset in y
    n_control_points: int   # Number of control points used
    coverage_ratio: float   # Fraction of control area with valid data


def compute_offsets(
    dx: xr.DataArray,
    dy: xr.DataArray,
    control_mask: xr.DataArray | NDArray | None = None,
    magnitude_threshold: float = 10.0,
) -> tuple[float, float, float, float]:
    """
    Compute systematic offsets from displacement map.

    Parameters
    ----------
    dx : xr.DataArray
        X-component of displacement.
    dy : xr.DataArray
        Y-component of displacement.
    control_mask : xr.DataArray or ndarray, optional
        Boolean mask where True indicates stable reference areas.
        If None, uses entire image with magnitude filtering.
    magnitude_threshold : float
        Maximum displacement magnitude to consider stable.

    Returns
    -------
    tuple
        (offset_x, offset_y, offset_x_std, offset_y_std)
    """
    # Compute magnitude
    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    # Create validity mask
    valid = ~(np.isnan(dx) | np.isnan(dy))

    # Apply magnitude threshold
    valid = valid & (magnitude < magnitude_threshold)

    # Apply control mask if provided
    if control_mask is not None:
        if isinstance(control_mask, xr.DataArray):
            control_mask = control_mask.values
        
        # Interpolate control mask to displacement grid if needed
        if control_mask.shape != dx.shape:
            from scipy.interpolate import RegularGridInterpolator

            # This is a simplified version - actual implementation
            # should handle coordinate systems properly
            valid = valid & (control_mask > 0)
        else:
            valid = valid & (control_mask > 0)

    # Extract valid values
    dx_valid = dx.values[valid.values if hasattr(valid, 'values') else valid]
    dy_valid = dy.values[valid.values if hasattr(valid, 'values') else valid]

    if len(dx_valid) < 10:
        return np.nan, np.nan, np.nan, np.nan

    # Compute robust statistics (median + MAD)
    offset_x = np.nanmedian(dx_valid)
    offset_y = np.nanmedian(dy_valid)

    # Use MAD for robust std estimate
    offset_x_std = 1.4826 * np.nanmedian(np.abs(dx_valid - offset_x))
    offset_y_std = 1.4826 * np.nanmedian(np.abs(dy_valid - offset_y))

    return offset_x, offset_y, offset_x_std, offset_y_std


def apply_coregistration(
    dx: xr.DataArray,
    dy: xr.DataArray,
    offset_x: float,
    offset_y: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Apply coregistration offsets to displacement map.

    Parameters
    ----------
    dx : xr.DataArray
        X-component of displacement.
    dy : xr.DataArray
        Y-component of displacement.
    offset_x : float
        Offset to remove in x.
    offset_y : float
        Offset to remove in y.

    Returns
    -------
    tuple
        Corrected (dx, dy) arrays.
    """
    dx_corrected = dx - offset_x
    dy_corrected = dy - offset_y
    return dx_corrected, dy_corrected


def load_control_surface(
    filepath: str,
    config: TSCorrConfig,
) -> xr.DataArray:
    """
    Load control surface mask (ice/water exclusion).

    Parameters
    ----------
    filepath : str
        Path to control surface file.
    config : TSCorrConfig
        Configuration object.

    Returns
    -------
    xr.DataArray
        Boolean mask where True indicates stable (non-ice, non-water) areas.
    """
    from ..io.raster import read_geotiff

    # Load ice/water mask
    mask = read_geotiff(filepath)

    # Invert if needed (depends on mask convention)
    # Assume mask values: 0 = stable, 1 = ice/water
    control = mask == 0

    return control


def coregister_displacement_stack(
    pairs: list,
    control_mask: xr.DataArray | None = None,
    scarp_mask: xr.DataArray | None = None,
    magnitude_threshold: float = 10.0,
) -> list[CoregistrationResult]:
    """
    Coregister a stack of displacement pairs.

    Parameters
    ----------
    pairs : list
        List of DisplacementPair objects.
    control_mask : xr.DataArray, optional
        Mask of stable reference areas.
    scarp_mask : xr.DataArray, optional
        Mask of deforming areas to exclude from coregistration.
    magnitude_threshold : float
        Maximum displacement to consider stable.

    Returns
    -------
    list
        List of CoregistrationResult objects.
    """
    results = []

    for i, pair in enumerate(pairs):
        if pair.dx is None or pair.dy is None:
            continue

        # Create combined mask
        if control_mask is not None and scarp_mask is not None:
            # Use control areas that are not in scarp
            mask = control_mask & ~scarp_mask
        elif control_mask is not None:
            mask = control_mask
        else:
            mask = None

        # Compute offsets
        offset_x, offset_y, std_x, std_y = compute_offsets(
            pair.dx, pair.dy, mask, magnitude_threshold
        )

        # Apply correction
        if not np.isnan(offset_x):
            pair.dx, pair.dy = apply_coregistration(
                pair.dx, pair.dy, offset_x, offset_y
            )

        # Count control points
        if mask is not None:
            valid = ~(np.isnan(pair.dx) | np.isnan(pair.dy))
            n_control = int(np.sum(valid.values & mask.values))
            coverage = n_control / np.sum(mask.values) if np.sum(mask.values) > 0 else 0
        else:
            n_control = int(np.sum(~np.isnan(pair.dx.values)))
            coverage = 1.0

        results.append(CoregistrationResult(
            pair_index=i,
            date_master=pair.date_master.strftime("%Y%m%d"),
            date_secondary=pair.date_secondary.strftime("%Y%m%d"),
            offset_x=float(offset_x) if not np.isnan(offset_x) else 0.0,
            offset_y=float(offset_y) if not np.isnan(offset_y) else 0.0,
            offset_x_std=float(std_x) if not np.isnan(std_x) else 0.0,
            offset_y_std=float(std_y) if not np.isnan(std_y) else 0.0,
            n_control_points=n_control,
            coverage_ratio=float(coverage),
        ))

    return results


def save_coregistration_results(
    results: list[CoregistrationResult],
    filepath: str,
) -> None:
    """
    Save coregistration results to file.

    Parameters
    ----------
    results : list
        List of CoregistrationResult objects.
    filepath : str
        Output file path.
    """
    with open(filepath, "w") as f:
        f.write("# Coregistration Results\n")
        f.write("# index, date_master, date_secondary, offset_x, offset_y, "
                "std_x, std_y, n_points, coverage\n")
        
        for r in results:
            f.write(f"{r.pair_index}, {r.date_master}, {r.date_secondary}, "
                    f"{r.offset_x:.6f}, {r.offset_y:.6f}, "
                    f"{r.offset_x_std:.6f}, {r.offset_y_std:.6f}, "
                    f"{r.n_control_points}, {r.coverage_ratio:.4f}\n")


def load_coregistration_results(filepath: str) -> list[CoregistrationResult]:
    """
    Load coregistration results from file.

    Parameters
    ----------
    filepath : str
        Path to coregistration results file.

    Returns
    -------
    list
        List of CoregistrationResult objects.
    """
    results = []

    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split(",")
            if len(parts) >= 9:
                results.append(CoregistrationResult(
                    pair_index=int(parts[0]),
                    date_master=parts[1].strip(),
                    date_secondary=parts[2].strip(),
                    offset_x=float(parts[3]),
                    offset_y=float(parts[4]),
                    offset_x_std=float(parts[5]),
                    offset_y_std=float(parts[6]),
                    n_control_points=int(parts[7]),
                    coverage_ratio=float(parts[8]),
                ))

    return results
