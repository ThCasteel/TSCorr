"""
Velocity and rate estimation from displacement time series.

This module implements time-series fitting algorithms for estimating
ground motion rates from stacked displacement measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import optimize

if TYPE_CHECKING:
    from ..config import TSCorrConfig


class FitModel(IntEnum):
    """Time series fitting models."""
    LINEAR = 1          # Linear velocity (ice melting)
    CONSTANT = 2        # Constant displacement (landslides)
    CONSTANT_LINEAR = 3 # Constant + linear (volcanic)


@dataclass
class VelocityResult:
    """Container for velocity estimation results."""
    rate_x: xr.DataArray         # X-component rate (m/yr)
    rate_y: xr.DataArray         # Y-component rate (m/yr)
    rate_magnitude: xr.DataArray # Rate magnitude (m/yr)
    rate_x_std: xr.DataArray     # Uncertainty in x-rate
    rate_y_std: xr.DataArray     # Uncertainty in y-rate
    rate_std: xr.DataArray       # Uncertainty in magnitude
    n_observations: xr.DataArray # Number of valid observations per pixel


def fit_linear_velocity(
    times: NDArray,
    displacements: NDArray,
    uncertainties: NDArray | None = None,
) -> tuple[float, float]:
    """
    Fit linear velocity model to displacement time series.

    Parameters
    ----------
    times : ndarray
        Time values (typically in days from reference).
    displacements : ndarray
        Displacement measurements.
    uncertainties : ndarray, optional
        Measurement uncertainties for weighted fitting.

    Returns
    -------
    tuple
        (velocity, velocity_std) in displacement_units / time_units.
    """
    valid = ~np.isnan(displacements)
    if valid.sum() < 2:
        return np.nan, np.nan

    t = times[valid]
    d = displacements[valid]

    if uncertainties is not None:
        w = 1.0 / uncertainties[valid] ** 2
    else:
        w = None

    # Linear fit: d = v * t + offset
    try:
        if w is not None:
            # Weighted least squares
            A = np.vstack([t, np.ones_like(t)]).T
            W = np.diag(w)
            AW = W @ A
            coeffs = np.linalg.lstsq(AW, W @ d, rcond=None)[0]
            velocity = coeffs[0]

            # Estimate uncertainty
            residuals = d - (velocity * t + coeffs[1])
            mse = np.sum(w * residuals ** 2) / (len(d) - 2)
            cov = mse * np.linalg.inv(A.T @ W @ A)
            velocity_std = np.sqrt(cov[0, 0])
        else:
            coeffs = np.polyfit(t, d, 1)
            velocity = coeffs[0]

            # Estimate uncertainty
            residuals = d - np.polyval(coeffs, t)
            mse = np.sum(residuals ** 2) / (len(d) - 2)
            velocity_std = np.sqrt(mse / np.sum((t - t.mean()) ** 2))

    except (np.linalg.LinAlgError, ValueError):
        return np.nan, np.nan

    return velocity, velocity_std


def fit_constant_displacement(
    displacements: NDArray,
    uncertainties: NDArray | None = None,
) -> tuple[float, float]:
    """
    Fit constant displacement model (robust mean).

    Used for sudden events like landslides or earthquakes where
    displacement is approximately constant across all measurements.

    Parameters
    ----------
    displacements : ndarray
        Displacement measurements.
    uncertainties : ndarray, optional
        Measurement uncertainties for weighted averaging.

    Returns
    -------
    tuple
        (mean_displacement, std_displacement)
    """
    valid = ~np.isnan(displacements)
    if valid.sum() < 1:
        return np.nan, np.nan

    d = displacements[valid]

    if uncertainties is not None:
        w = 1.0 / uncertainties[valid] ** 2
        mean_d = np.sum(w * d) / np.sum(w)
        var_d = 1.0 / np.sum(w)
        std_d = np.sqrt(var_d)
    else:
        mean_d = np.nanmean(d)
        std_d = np.nanstd(d) / np.sqrt(valid.sum())

    return mean_d, std_d


def fit_constant_linear(
    times: NDArray,
    displacements: NDArray,
    event_time: float,
    uncertainties: NDArray | None = None,
) -> tuple[float, float, float, float]:
    """
    Fit constant + linear model for volcanic events.

    Model: d = offset + step * H(t - event_time) + velocity * t
    where H is the Heaviside step function.

    Parameters
    ----------
    times : ndarray
        Time values.
    displacements : ndarray
        Displacement measurements.
    event_time : float
        Time of the step event.
    uncertainties : ndarray, optional
        Measurement uncertainties.

    Returns
    -------
    tuple
        (step_displacement, step_std, velocity, velocity_std)
    """
    valid = ~np.isnan(displacements)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan

    t = times[valid]
    d = displacements[valid]

    # Design matrix: [1, H(t-t0), t]
    step = (t >= event_time).astype(float)
    A = np.column_stack([np.ones_like(t), step, t])

    try:
        if uncertainties is not None:
            w = 1.0 / uncertainties[valid] ** 2
            W = np.diag(w)
            coeffs = np.linalg.lstsq(W @ A, W @ d, rcond=None)[0]

            residuals = d - A @ coeffs
            mse = np.sum(w * residuals ** 2) / (len(d) - 3)
            cov = mse * np.linalg.inv(A.T @ W @ A)
        else:
            coeffs = np.linalg.lstsq(A, d, rcond=None)[0]

            residuals = d - A @ coeffs
            mse = np.sum(residuals ** 2) / (len(d) - 3)
            cov = mse * np.linalg.inv(A.T @ A)

        step_disp = coeffs[1]
        velocity = coeffs[2]
        step_std = np.sqrt(cov[1, 1])
        velocity_std = np.sqrt(cov[2, 2])

    except (np.linalg.LinAlgError, ValueError):
        return np.nan, np.nan, np.nan, np.nan

    return step_disp, step_std, velocity, velocity_std


def estimate_velocity_grid(
    dx_stack: xr.DataArray,
    dy_stack: xr.DataArray,
    model: FitModel = FitModel.CONSTANT,
    event_time: float | None = None,
) -> VelocityResult:
    """
    Estimate velocity/rate for each pixel from displacement stack.

    Parameters
    ----------
    dx_stack : xr.DataArray
        Stacked x-displacements with dims (time, y, x).
    dy_stack : xr.DataArray
        Stacked y-displacements with dims (time, y, x).
    model : FitModel
        Time series model to fit.
    event_time : float, optional
        Event time for CONSTANT_LINEAR model.

    Returns
    -------
    VelocityResult
        Estimated velocities and uncertainties.
    """
    # Convert times to numeric (days from first observation)
    times = dx_stack.time.values
    t0 = times[0]
    t_numeric = np.array([(t - t0).astype("timedelta64[D]").astype(float) 
                          for t in times])

    # Get dimensions
    ny, nx = dx_stack.shape[1], dx_stack.shape[2]

    # Initialize output arrays
    rate_x = np.full((ny, nx), np.nan)
    rate_y = np.full((ny, nx), np.nan)
    rate_x_std = np.full((ny, nx), np.nan)
    rate_y_std = np.full((ny, nx), np.nan)
    n_obs = np.zeros((ny, nx), dtype=int)

    # Compute arrays if dask
    if hasattr(dx_stack.data, "compute"):
        dx_data = dx_stack.values
        dy_data = dy_stack.values
    else:
        dx_data = dx_stack.values
        dy_data = dy_stack.values

    # Fit model at each pixel
    for j in range(ny):
        for i in range(nx):
            dx_ts = dx_data[:, j, i]
            dy_ts = dy_data[:, j, i]

            valid = ~(np.isnan(dx_ts) | np.isnan(dy_ts))
            n_obs[j, i] = valid.sum()

            if n_obs[j, i] < 2:
                continue

            if model == FitModel.LINEAR:
                # Linear velocity (m/day to m/year)
                vx, vx_std = fit_linear_velocity(t_numeric, dx_ts)
                vy, vy_std = fit_linear_velocity(t_numeric, dy_ts)
                rate_x[j, i] = vx * 365.25
                rate_y[j, i] = vy * 365.25
                rate_x_std[j, i] = vx_std * 365.25
                rate_y_std[j, i] = vy_std * 365.25

            elif model == FitModel.CONSTANT:
                # Constant displacement - report mean displacement
                rate_x[j, i], rate_x_std[j, i] = fit_constant_displacement(dx_ts)
                rate_y[j, i], rate_y_std[j, i] = fit_constant_displacement(dy_ts)

            elif model == FitModel.CONSTANT_LINEAR:
                if event_time is None:
                    raise ValueError("event_time required for CONSTANT_LINEAR model")
                # Fit constant + linear
                step_x, step_x_std, vx, vx_std = fit_constant_linear(
                    t_numeric, dx_ts, event_time
                )
                step_y, step_y_std, vy, vy_std = fit_constant_linear(
                    t_numeric, dy_ts, event_time
                )
                # Report step displacement as "rate"
                rate_x[j, i] = step_x
                rate_y[j, i] = step_y
                rate_x_std[j, i] = step_x_std
                rate_y_std[j, i] = step_y_std

    # Compute magnitude
    rate_mag = np.sqrt(rate_x ** 2 + rate_y ** 2)
    # Propagate uncertainty
    rate_std = np.sqrt(
        (rate_x / rate_mag * rate_x_std) ** 2 +
        (rate_y / rate_mag * rate_y_std) ** 2
    )

    # Create xarray DataArrays
    coords = {"y": dx_stack.y.values, "x": dx_stack.x.values}

    return VelocityResult(
        rate_x=xr.DataArray(rate_x, dims=["y", "x"], coords=coords),
        rate_y=xr.DataArray(rate_y, dims=["y", "x"], coords=coords),
        rate_magnitude=xr.DataArray(rate_mag, dims=["y", "x"], coords=coords),
        rate_x_std=xr.DataArray(rate_x_std, dims=["y", "x"], coords=coords),
        rate_y_std=xr.DataArray(rate_y_std, dims=["y", "x"], coords=coords),
        rate_std=xr.DataArray(rate_std, dims=["y", "x"], coords=coords),
        n_observations=xr.DataArray(n_obs, dims=["y", "x"], coords=coords),
    )


def estimate_velocity_dask(
    dx_stack: xr.DataArray,
    dy_stack: xr.DataArray,
    model: FitModel = FitModel.CONSTANT,
    event_time: float | None = None,
) -> VelocityResult:
    """
    Estimate velocity using dask for parallel computation.

    This is a memory-efficient version that processes chunks in parallel.

    Parameters
    ----------
    dx_stack : xr.DataArray
        Stacked x-displacements with dims (time, y, x).
    dy_stack : xr.DataArray
        Stacked y-displacements with dims (time, y, x).
    model : FitModel
        Time series model to fit.
    event_time : float, optional
        Event time for CONSTANT_LINEAR model.

    Returns
    -------
    VelocityResult
        Estimated velocities and uncertainties.
    """
    # Convert times to numeric
    times = dx_stack.time.values
    t0 = times[0]
    t_numeric = np.array([(t - t0).astype("timedelta64[D]").astype(float) 
                          for t in times])

    def fit_pixel_block(dx_block, dy_block):
        """Fit model to a block of pixels."""
        nt, ny, nx = dx_block.shape
        results = np.zeros((6, ny, nx))  # rate_x, rate_y, std_x, std_y, mag, n_obs

        for j in range(ny):
            for i in range(nx):
                dx_ts = dx_block[:, j, i]
                dy_ts = dy_block[:, j, i]

                valid = ~(np.isnan(dx_ts) | np.isnan(dy_ts))
                n_valid = valid.sum()
                results[5, j, i] = n_valid

                if n_valid < 2:
                    results[:5, j, i] = np.nan
                    continue

                if model == FitModel.LINEAR:
                    vx, vx_std = fit_linear_velocity(t_numeric, dx_ts)
                    vy, vy_std = fit_linear_velocity(t_numeric, dy_ts)
                    results[0, j, i] = vx * 365.25
                    results[1, j, i] = vy * 365.25
                    results[2, j, i] = vx_std * 365.25
                    results[3, j, i] = vy_std * 365.25
                elif model == FitModel.CONSTANT:
                    results[0, j, i], results[2, j, i] = fit_constant_displacement(dx_ts)
                    results[1, j, i], results[3, j, i] = fit_constant_displacement(dy_ts)

                # Magnitude
                results[4, j, i] = np.sqrt(results[0, j, i]**2 + results[1, j, i]**2)

        return results

    # Apply using dask map_blocks
    if hasattr(dx_stack.data, "map_blocks"):
        results = da.map_blocks(
            fit_pixel_block,
            dx_stack.data,
            dy_stack.data,
            dtype=float,
            drop_axis=0,
            new_axis=0,
            chunks=(6,) + dx_stack.chunks[1:],
        )
    else:
        # Fall back to non-dask version
        return estimate_velocity_grid(dx_stack, dy_stack, model, event_time)

    # Extract individual results
    coords = {"y": dx_stack.y.values, "x": dx_stack.x.values}

    return VelocityResult(
        rate_x=xr.DataArray(results[0], dims=["y", "x"], coords=coords),
        rate_y=xr.DataArray(results[1], dims=["y", "x"], coords=coords),
        rate_magnitude=xr.DataArray(results[4], dims=["y", "x"], coords=coords),
        rate_x_std=xr.DataArray(results[2], dims=["y", "x"], coords=coords),
        rate_y_std=xr.DataArray(results[3], dims=["y", "x"], coords=coords),
        rate_std=xr.DataArray(
            np.sqrt((results[0]/results[4]*results[2])**2 + 
                    (results[1]/results[4]*results[3])**2),
            dims=["y", "x"], coords=coords
        ),
        n_observations=xr.DataArray(results[5].astype(int), dims=["y", "x"], coords=coords),
    )
