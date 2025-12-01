"""
Tests for TSCorr displacement processing module.
"""

import numpy as np
import pytest
import xarray as xr

from tscorr.processing.displacement import (
    parse_correlation_filename,
    filter_displacement_by_angle,
    compute_displacement_magnitude,
)
from tscorr.processing.velocity import (
    FitModel,
    fit_linear_velocity,
    fit_constant_displacement,
)


class TestFilenameParser:
    """Tests for correlation filename parsing."""

    def test_standard_format(self):
        """Test standard filename format."""
        filename = "correlation_20200115120000_LC08_B8_sub_vs_20210620180000_LC08_B8_sub.tif"
        date_m, date_s = parse_correlation_filename(filename)
        
        assert date_m.year == 2020
        assert date_m.month == 1
        assert date_m.day == 15
        
        assert date_s.year == 2021
        assert date_s.month == 6
        assert date_s.day == 20

    def test_short_date_format(self):
        """Test 8-digit date format."""
        filename = "correlation_20200115_vs_20210620.tif"
        date_m, date_s = parse_correlation_filename(filename)
        
        assert date_m.year == 2020
        assert date_s.year == 2021

    def test_invalid_filename(self):
        """Test that invalid filenames raise error."""
        with pytest.raises(ValueError):
            parse_correlation_filename("invalid_filename.tif")


class TestDisplacementFiltering:
    """Tests for displacement filtering."""

    def test_angle_filter_consistent(self):
        """Test that consistent angles pass filter."""
        # Create displacement with consistent direction (all pointing east)
        x = np.arange(100)
        y = np.arange(100)
        
        dx_data = np.ones((100, 100)) * 5.0  # 5m east
        dy_data = np.zeros((100, 100))        # 0m north
        
        dx = xr.DataArray(dx_data, dims=["y", "x"], coords={"y": y, "x": x})
        dy = xr.DataArray(dy_data, dims=["y", "x"], coords={"y": y, "x": x})
        
        valid_mask = filter_displacement_by_angle(dx, dy, threshold_degrees=60.0)
        
        # Most pixels should be valid
        assert valid_mask.values.mean() > 0.8

    def test_angle_filter_random(self):
        """Test that random angles are filtered."""
        np.random.seed(42)
        x = np.arange(100)
        y = np.arange(100)
        
        # Random displacement
        dx_data = np.random.randn(100, 100) * 10
        dy_data = np.random.randn(100, 100) * 10
        
        dx = xr.DataArray(dx_data, dims=["y", "x"], coords={"y": y, "x": x})
        dy = xr.DataArray(dy_data, dims=["y", "x"], coords={"y": y, "x": x})
        
        valid_mask = filter_displacement_by_angle(dx, dy, threshold_degrees=30.0)
        
        # Most pixels should be invalid due to random directions
        assert valid_mask.values.mean() < 0.5


class TestMagnitude:
    """Tests for displacement magnitude calculation."""

    def test_magnitude_simple(self):
        """Test magnitude calculation."""
        x = np.arange(10)
        y = np.arange(10)
        
        dx = xr.DataArray(np.full((10, 10), 3.0), dims=["y", "x"], coords={"y": y, "x": x})
        dy = xr.DataArray(np.full((10, 10), 4.0), dims=["y", "x"], coords={"y": y, "x": x})
        
        mag = compute_displacement_magnitude(dx, dy)
        
        # 3-4-5 triangle
        np.testing.assert_allclose(mag.values, 5.0)


class TestVelocityFitting:
    """Tests for velocity fitting functions."""

    def test_linear_fit(self):
        """Test linear velocity fitting."""
        # Create linear data: d = 0.1 * t
        times = np.array([0, 30, 60, 90, 120])  # days
        velocity_true = 0.1  # m/day
        displacements = velocity_true * times + np.random.randn(5) * 0.01
        
        velocity, velocity_std = fit_linear_velocity(times, displacements)
        
        np.testing.assert_allclose(velocity, velocity_true, atol=0.02)
        assert velocity_std > 0

    def test_linear_fit_insufficient_data(self):
        """Test linear fit with insufficient data."""
        times = np.array([0])
        displacements = np.array([1.0])
        
        velocity, velocity_std = fit_linear_velocity(times, displacements)
        
        assert np.isnan(velocity)
        assert np.isnan(velocity_std)

    def test_constant_fit(self):
        """Test constant displacement fitting."""
        # Create constant data with noise
        np.random.seed(42)
        true_value = 5.0
        displacements = true_value + np.random.randn(20) * 0.1
        
        mean_d, std_d = fit_constant_displacement(displacements)
        
        np.testing.assert_allclose(mean_d, true_value, atol=0.1)
        assert std_d > 0

    def test_constant_fit_with_nan(self):
        """Test constant fit handles NaN values."""
        displacements = np.array([5.0, np.nan, 4.8, 5.2, np.nan, 5.1])
        
        mean_d, std_d = fit_constant_displacement(displacements)
        
        assert not np.isnan(mean_d)
        np.testing.assert_allclose(mean_d, 5.0, atol=0.2)


class TestFitModel:
    """Tests for FitModel enum."""

    def test_model_values(self):
        """Test model enum values."""
        assert FitModel.LINEAR == 1
        assert FitModel.CONSTANT == 2
        assert FitModel.CONSTANT_LINEAR == 3
