"""
Tests for TSCorr configuration module.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from tscorr.config import TSCorrConfig, load_config


@pytest.fixture
def sample_config_yaml():
    """Create a minimal valid configuration."""
    return """
paths:
  image_dir: "./test_images/"
  code_dir: "./tscorr/"
  output_dir: "./test_outputs/"
  image_subdir: "./test_imagesubdir/"
  corr_result_dir: "./test_results/"
  cosi_result_dir: "./test_cosiresults/"
  pic_dir: "./test_pics/"
  param_dir: "./test_para/"
  corr_file_list: "./corrfilelist.txt"
  sdm_list: "./sdmlist.txt"
  cosi_plus_list: "./cosipluslist.txt"

image:
  type: 1
  resolution: 15
  band_patterns:
    landsat8: "L*_B8.TIF"

projection:
  epsg: "epsg:32637"
  description: "UTM zone 37 north"

processing:
  tool: 4
  earthquake_mode: false
  apply_subsetting: true
  coverage_threshold: 80
  subtile_size_km: 10
  algorithm: 2
  output_resolution: 60
  cosi_step_size: 60
  pool_size: 1
  run_step: 2

temporal:
  season_start_month: 1
  season_end_month: 12
  year_start: 0
  year_end: 9999
  min_date: null
  max_date: null
  exclude_dates: []
  earthquake_epochs: []
  time_fix: 1

velocity:
  include_ice_water: 1
  points_only: false

scarp:
  mode: 0
  shapefile: null

plotting:
  enabled: false
  save_output: false
  scale_factor: 0.025
  scalebar_length: 20
  scalebar_position: [0, 0]

tolerances:
  float64_absolute: 1.0e-8
  float32_absolute: 1.0e-5
  geophysical_relative: 1.0e-6

dask:
  scheduler: "synchronous"
  num_workers: 1
  memory_limit: "4GB"
  chunk_size:
    x: 512
    y: 512
"""


@pytest.fixture
def config_file(sample_config_yaml, tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(sample_config_yaml)
    return config_path


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config(self, config_file):
        """Test that config loads without errors."""
        config = load_config(config_file)
        assert config is not None
        assert isinstance(config, TSCorrConfig)

    def test_paths_config(self, config_file):
        """Test paths configuration."""
        config = load_config(config_file)
        assert config.paths.image_dir.name == "test_images"
        assert config.paths.output_dir.name == "test_outputs"

    def test_image_config(self, config_file):
        """Test image configuration."""
        config = load_config(config_file)
        assert config.image.type == 1
        assert config.image.resolution == 15.0

    def test_projection_config(self, config_file):
        """Test projection configuration."""
        config = load_config(config_file)
        assert config.projection.epsg == "epsg:32637"
        assert config.projection.epsg_code == 32637

    def test_processing_config(self, config_file):
        """Test processing configuration."""
        config = load_config(config_file)
        assert config.processing.tool == 4
        assert config.processing.earthquake_mode is False
        assert config.processing.algorithm == 2

    def test_temporal_config(self, config_file):
        """Test temporal configuration."""
        config = load_config(config_file)
        assert config.temporal.season_start_month == 1
        assert config.temporal.season_end_month == 12
        assert config.temporal.min_date is None

    def test_dask_config(self, config_file):
        """Test Dask configuration."""
        config = load_config(config_file)
        assert config.dask.scheduler == "synchronous"
        assert config.dask.num_workers == 1
        assert config.dask.chunk_size["x"] == 512


class TestTemporalFiltering:
    """Tests for temporal filtering logic."""

    def test_date_validation(self, config_file):
        """Test date validation."""
        from datetime import datetime
        
        config = load_config(config_file)
        
        # Date within range should pass
        valid_date = datetime(2022, 6, 15)
        assert config.temporal.is_date_valid(valid_date) is True

    def test_date_outside_year_range(self, config_file):
        """Test date outside year range."""
        from datetime import datetime
        
        config = load_config(config_file)
        # Modify config to restrict years
        config.temporal.year_start = 2020
        config.temporal.year_end = 2023
        
        old_date = datetime(2019, 6, 15)
        assert config.temporal.is_date_valid(old_date) is False
        
        new_date = datetime(2021, 6, 15)
        assert config.temporal.is_date_valid(new_date) is True


class TestImageConfig:
    """Tests for image configuration."""

    def test_get_file_pattern_landsat(self, config_file):
        """Test file pattern for Landsat."""
        config = load_config(config_file)
        config.image.type = 1
        assert "L*_B8.TIF" in config.image.get_file_pattern()

    def test_get_file_pattern_worldview(self, config_file):
        """Test file pattern for WorldView."""
        config = load_config(config_file)
        config.image.type = 8
        pattern = config.image.get_file_pattern()
        assert "W" in pattern


class TestSetup:
    """Tests for environment setup."""

    def test_ensure_directories(self, config_file, tmp_path):
        """Test that output directories are created."""
        config = load_config(config_file)
        
        # Override paths to use tmp_path
        config.paths.output_dir = tmp_path / "outputs"
        config.paths.image_subdir = tmp_path / "imagesubdir"
        config.paths.corr_result_dir = tmp_path / "results"
        
        config.paths.ensure_directories()
        
        assert config.paths.output_dir.exists()
        assert config.paths.image_subdir.exists()
        assert config.paths.corr_result_dir.exists()
