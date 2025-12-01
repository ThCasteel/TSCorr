"""
Configuration management for TSCorr.

Loads configuration from YAML file and provides typed access to parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    """File and directory paths configuration."""
    image_dir: Path
    code_dir: Path
    output_dir: Path
    image_subdir: Path
    corr_result_dir: Path
    cosi_result_dir: Path
    pic_dir: Path
    param_dir: Path
    corr_file_list: Path
    sdm_list: Path
    cosi_plus_list: Path

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path) -> PathsConfig:
        """Create PathsConfig from dictionary, expanding paths relative to base_dir."""
        def resolve_path(p: str) -> Path:
            path = Path(os.path.expanduser(p))
            if not path.is_absolute():
                path = base_dir / path
            return path

        return cls(
            image_dir=resolve_path(d["image_dir"]),
            code_dir=resolve_path(d["code_dir"]),
            output_dir=resolve_path(d["output_dir"]),
            image_subdir=resolve_path(d["image_subdir"]),
            corr_result_dir=resolve_path(d["corr_result_dir"]),
            cosi_result_dir=resolve_path(d["cosi_result_dir"]),
            pic_dir=resolve_path(d["pic_dir"]),
            param_dir=resolve_path(d["param_dir"]),
            corr_file_list=resolve_path(d["corr_file_list"]),
            sdm_list=resolve_path(d["sdm_list"]),
            cosi_plus_list=resolve_path(d["cosi_plus_list"]),
        )

    def ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        for attr in ["output_dir", "image_subdir", "corr_result_dir", 
                     "cosi_result_dir", "pic_dir", "param_dir"]:
            path = getattr(self, attr)
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ImageConfig:
    """Image configuration."""
    type: int
    resolution: float
    band_patterns: dict[str, str]

    # Image type constants
    LANDSAT_8 = 1
    ASTER = 2
    PLANET = 3
    PGC = 4
    LANDSAT_7 = 5
    SUBSET = 6
    WORLDVIEW = 8

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImageConfig:
        return cls(
            type=d["type"],
            resolution=float(d["resolution"]),
            band_patterns=d["band_patterns"],
        )

    def get_file_pattern(self) -> str:
        """Get the file search pattern for the configured image type."""
        type_to_pattern = {
            self.LANDSAT_8: self.band_patterns.get("landsat8", "L*_B8.TIF"),
            self.ASTER: self.band_patterns.get("aster", "AST_L1T_*_V.tif"),
            self.PLANET: self.band_patterns.get("planet", "*AnalyticMS*.tif"),
            self.WORLDVIEW: self.band_patterns.get("worldview", "W*.tif"),
            self.LANDSAT_7: self.band_patterns.get("landsat8", "L*_B8.TIF"),
        }
        return type_to_pattern.get(self.type, "*.*")


@dataclass
class ProjectionConfig:
    """Projection configuration."""
    epsg: str
    description: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProjectionConfig:
        return cls(
            epsg=d["epsg"],
            description=d["description"],
        )

    @property
    def epsg_code(self) -> int:
        """Extract numeric EPSG code."""
        return int(self.epsg.lower().replace("epsg:", ""))


@dataclass
class ProcessingConfig:
    """Processing parameters configuration."""
    tool: int
    earthquake_mode: bool
    apply_subsetting: bool
    coverage_threshold: float
    force_warp: bool
    subtile_size_km: int
    algorithm: int
    output_resolution: float
    cosi_step_size: float
    pool_size: int
    run_step: int

    # Tool constants
    COSI_CORR = 1
    SETSM_SDM = 2
    MIMIC2 = 3
    COSI_CORR_PLUS = 4

    # Algorithm constants
    LINEAR = 1
    CONSTANT = 2
    CONSTANT_LINEAR = 3

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProcessingConfig:
        return cls(
            tool=d["tool"],
            earthquake_mode=d["earthquake_mode"],
            apply_subsetting=d["apply_subsetting"],
            coverage_threshold=float(d["coverage_threshold"]),
            force_warp=d.get("force_warp", False),
            subtile_size_km=d["subtile_size_km"],
            algorithm=d["algorithm"],
            output_resolution=float(d["output_resolution"]),
            cosi_step_size=float(d["cosi_step_size"]),
            pool_size=d["pool_size"],
            run_step=d["run_step"],
        )


@dataclass
class TemporalConfig:
    """Temporal filtering configuration."""
    season_start_month: int
    season_end_month: int
    year_start: int
    year_end: int
    min_date: datetime | None
    max_date: datetime | None
    exclude_dates: list[datetime]
    earthquake_epochs: list[datetime]
    time_fix: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemporalConfig:
        def parse_date(s: str | None) -> datetime | None:
            if s is None:
                return None
            return datetime.strptime(s, "%Y/%m/%d")

        def parse_dates(lst: list[str] | None) -> list[datetime]:
            if lst is None:
                return []
            return [parse_date(s) for s in lst if s is not None]

        return cls(
            season_start_month=d["season_start_month"],
            season_end_month=d["season_end_month"],
            year_start=d["year_start"],
            year_end=d["year_end"],
            min_date=parse_date(d.get("min_date")),
            max_date=parse_date(d.get("max_date")),
            exclude_dates=parse_dates(d.get("exclude_dates")),
            earthquake_epochs=parse_dates(d.get("earthquake_epochs")),
            time_fix=d.get("time_fix", 1),
        )

    def is_date_valid(self, dt: datetime) -> bool:
        """Check if a date passes all temporal filters."""
        if self.min_date and dt < self.min_date:
            return False
        if self.max_date and dt > self.max_date:
            return False
        if dt.year < self.year_start or dt.year > self.year_end:
            return False
        if dt.month < self.season_start_month or dt.month > self.season_end_month:
            return False
        if dt in self.exclude_dates:
            return False
        return True


@dataclass
class VelocityConfig:
    """Velocity calculation configuration."""
    include_ice_water: int
    points_only: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VelocityConfig:
        return cls(
            include_ice_water=d["include_ice_water"],
            points_only=d["points_only"],
        )


@dataclass
class ScarpConfig:
    """Scarp/ROI configuration."""
    mode: int
    shapefile: Path | None

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path) -> ScarpConfig:
        shapefile = None
        if d.get("shapefile"):
            shapefile = base_dir / d["shapefile"]
        return cls(
            mode=d["mode"],
            shapefile=shapefile,
        )


@dataclass
class PlottingConfig:
    """Plotting configuration."""
    enabled: bool
    save_output: bool
    scale_factor: float
    scalebar_length: float
    scalebar_position: tuple[float, float]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlottingConfig:
        pos = d.get("scalebar_position", [0, 0])
        return cls(
            enabled=d["enabled"],
            save_output=d["save_output"],
            scale_factor=float(d["scale_factor"]),
            scalebar_length=float(d["scalebar_length"]),
            scalebar_position=(float(pos[0]), float(pos[1])),
        )


@dataclass
class ToleranceConfig:
    """Numerical tolerance configuration for validation."""
    float64_absolute: float
    float32_absolute: float
    geophysical_relative: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToleranceConfig:
        return cls(
            float64_absolute=float(d["float64_absolute"]),
            float32_absolute=float(d["float32_absolute"]),
            geophysical_relative=float(d["geophysical_relative"]),
        )


@dataclass
class DaskConfig:
    """Dask parallel processing configuration."""
    scheduler: str
    num_workers: int
    memory_limit: str
    chunk_size: dict[str, int]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DaskConfig:
        return cls(
            scheduler=d["scheduler"],
            num_workers=d["num_workers"],
            memory_limit=d["memory_limit"],
            chunk_size=d["chunk_size"],
        )


@dataclass
class TSCorrConfig:
    """Main configuration container for TSCorr."""
    paths: PathsConfig
    image: ImageConfig
    projection: ProjectionConfig
    processing: ProcessingConfig
    temporal: TemporalConfig
    velocity: VelocityConfig
    scarp: ScarpConfig
    plotting: PlottingConfig
    tolerances: ToleranceConfig
    dask: DaskConfig
    config_path: Path = field(default_factory=lambda: Path("."))

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> TSCorrConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        base_dir = config_path.parent

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(
            paths=PathsConfig.from_dict(data["paths"], base_dir),
            image=ImageConfig.from_dict(data["image"]),
            projection=ProjectionConfig.from_dict(data["projection"]),
            processing=ProcessingConfig.from_dict(data["processing"]),
            temporal=TemporalConfig.from_dict(data["temporal"]),
            velocity=VelocityConfig.from_dict(data["velocity"]),
            scarp=ScarpConfig.from_dict(data["scarp"], base_dir),
            plotting=PlottingConfig.from_dict(data["plotting"]),
            tolerances=ToleranceConfig.from_dict(data["tolerances"]),
            dask=DaskConfig.from_dict(data["dask"]),
            config_path=config_path,
        )

    def setup(self) -> None:
        """Initialize the processing environment."""
        self.paths.ensure_directories()

    def to_yaml(self, output_path: str | Path) -> None:
        """Save current configuration to YAML file."""
        # Re-load the original yaml to preserve comments and structure
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        
        # Update with current values (simplified - just saves the loaded data)
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(config_path: str | Path = "config.yaml") -> TSCorrConfig:
    """Convenience function to load configuration."""
    return TSCorrConfig.from_yaml(config_path)
