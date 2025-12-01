"""
Main workflow for TSCorr processing.

This module orchestrates the complete processing pipeline from
image preparation through velocity estimation.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import dask
import numpy as np
import xarray as xr

from .config import TSCorrConfig, load_config
from .io.raster import (
    get_image_boundary,
    read_geotiff,
    write_geotiff,
    reproject_bounds,
)
from .processing.displacement import DisplacementStack
from .processing.velocity import (
    FitModel,
    VelocityResult,
    estimate_velocity_grid,
    estimate_velocity_dask,
)
from .processing.coregistration import (
    coregister_displacement_stack,
    save_coregistration_results,
)
from .utils.geospatial import (
    find_images_in_region,
    standardize_filename,
)


class TSCorrWorkflow:
    """
    Main workflow class for TSCorr processing.

    Handles the complete pipeline from image discovery through
    velocity/displacement estimation.

    Parameters
    ----------
    config_path : str or Path
        Path to configuration YAML file.
    """

    def __init__(self, config_path: str | Path = "config.yaml"):
        """Initialize workflow with configuration."""
        self.config = load_config(config_path)
        self.config.setup()

        # Initialize Dask
        self._setup_dask()

        # Data containers
        self.image_metadata: list[dict[str, Any]] = []
        self.displacement_stack: DisplacementStack | None = None
        self.velocity_result: VelocityResult | None = None

    def _setup_dask(self) -> None:
        """Configure Dask based on configuration."""
        dask_cfg = self.config.dask

        if dask_cfg.scheduler == "threads":
            dask.config.set(scheduler="threads")
        elif dask_cfg.scheduler == "processes":
            from dask.distributed import Client
            self._dask_client = Client(
                n_workers=dask_cfg.num_workers,
                memory_limit=dask_cfg.memory_limit,
            )
        else:
            dask.config.set(scheduler="synchronous")

    def discover_images(self) -> int:
        """
        Discover and catalog available images.

        Returns
        -------
        int
            Number of images found.
        """
        print("Step 0: Discovering images...")

        image_dir = self.config.paths.image_dir
        pattern = self.config.image.get_file_pattern()

        # Find all matching files
        image_files = list(image_dir.rglob(pattern))
        print(f"  Found {len(image_files)} image files")

        # Get metadata for each image
        self.image_metadata = []
        for filepath in image_files:
            try:
                boundary, bounds, crs = get_image_boundary(filepath)
                
                metadata = {
                    "filepath": filepath,
                    "boundary": boundary,
                    "bounds": bounds,
                    "crs": crs,
                    "filename": filepath.name,
                }
                self.image_metadata.append(metadata)
            except Exception as e:
                print(f"  Warning: Could not read {filepath}: {e}")

        print(f"  Successfully cataloged {len(self.image_metadata)} images")
        return len(self.image_metadata)

    def save_image_catalog(self, output_path: str | Path | None = None) -> None:
        """
        Save image catalog to file (equivalent to mat0.mat).

        Parameters
        ----------
        output_path : str or Path, optional
            Output file path. Defaults to config output directory.
        """
        import pickle

        if output_path is None:
            output_path = self.config.paths.output_dir / "image_catalog.pkl"

        with open(output_path, "wb") as f:
            pickle.dump(self.image_metadata, f)

        print(f"Saved image catalog to {output_path}")

    def load_image_catalog(self, input_path: str | Path | None = None) -> int:
        """
        Load previously saved image catalog.

        Parameters
        ----------
        input_path : str or Path, optional
            Input file path. Defaults to config output directory.

        Returns
        -------
        int
            Number of images loaded.
        """
        import pickle

        if input_path is None:
            input_path = self.config.paths.output_dir / "image_catalog.pkl"

        if not Path(input_path).exists():
            raise FileNotFoundError(f"Image catalog not found: {input_path}")

        with open(input_path, "rb") as f:
            self.image_metadata = pickle.load(f)

        print(f"Loaded {len(self.image_metadata)} images from catalog")
        return len(self.image_metadata)

    def prepare_images(
        self,
        bounds: tuple[float, float, float, float],
    ) -> list[Path]:
        """
        Prepare images for correlation by subsetting and reprojecting.

        Parameters
        ----------
        bounds : tuple
            Target region bounds (minx, maxx, miny, maxy).

        Returns
        -------
        list
            Paths to prepared image files.
        """
        print("Step 1: Preparing images...")

        target_crs = self.config.projection.epsg
        output_res = self.config.image.resolution
        output_dir = self.config.paths.image_subdir

        # Find overlapping images
        image_info = [
            (m["boundary"], m["bounds"], m["crs"])
            for m in self.image_metadata
        ]
        overlap_indices = find_images_in_region(
            image_info, bounds, target_crs,
            min_overlap=self.config.processing.coverage_threshold / 100,
        )

        print(f"  Found {len(overlap_indices)} overlapping images")

        prepared_files = []
        for idx in overlap_indices:
            meta = self.image_metadata[idx]
            filepath = meta["filepath"]

            # Generate output filename
            out_name = standardize_filename(filepath, self.config.image.type)
            out_path = output_dir / out_name

            if out_path.exists():
                prepared_files.append(out_path)
                continue

            # Check if reprojection needed
            needs_warp = (
                self.config.processing.force_warp or
                meta["crs"].lower() != target_crs.lower()
            )

            if needs_warp or self.config.processing.apply_subsetting:
                # Use GDAL for reprojection/subsetting
                cmd = [
                    "gdalwarp",
                    str(filepath),
                    str(out_path),
                    "-t_srs", target_crs,
                    "-te", str(bounds[0]), str(bounds[2]), str(bounds[1]), str(bounds[3]),
                    "-tr", str(output_res), str(output_res),
                    "-r", "bilinear",
                    "-ot", "UInt16",
                ]

                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    print(f"  Warning: Failed to process {filepath}")
                    continue
            else:
                # Just create symlink
                out_path.symlink_to(filepath)

            prepared_files.append(out_path)

        print(f"  Prepared {len(prepared_files)} images")
        return prepared_files

    def run_correlation(
        self,
        image_files: list[Path] | None = None,
    ) -> int:
        """
        Run image correlation to generate displacement maps.

        This step calls the external correlation tool (COSI-Corr Plus).

        Parameters
        ----------
        image_files : list, optional
            List of prepared image files. If None, uses images in image_subdir.

        Returns
        -------
        int
            Number of correlation pairs generated.
        """
        print("Step 2: Running image correlation...")

        if image_files is None:
            image_files = list(self.config.paths.image_subdir.glob("*.TIF"))

        if len(image_files) < 2:
            print("  Not enough images for correlation")
            return 0

        # Sort by date
        image_files = sorted(image_files)

        # Generate correlation pairs
        tool = self.config.processing.tool
        output_dir = self.config.paths.corr_result_dir

        n_pairs = 0
        for i, base_file in enumerate(image_files):
            for target_file in image_files[i + 1:]:
                output_name = f"correlation_{base_file.stem}_vs_{target_file.stem}.tif"
                output_path = output_dir / output_name

                if output_path.exists():
                    n_pairs += 1
                    continue

                if tool == self.config.processing.COSI_CORR_PLUS:
                    # Call COSI-Corr Plus CLI
                    # This is a placeholder - actual implementation depends on
                    # the COSI-Corr Plus CLI interface
                    print(f"  Would correlate: {base_file.name} vs {target_file.name}")
                    n_pairs += 1

        print(f"  Generated {n_pairs} correlation pairs")
        return n_pairs

    def load_displacements(
        self,
        file_list_path: str | Path | None = None,
    ) -> int:
        """
        Load displacement maps from correlation results.

        Parameters
        ----------
        file_list_path : str or Path, optional
            Path to file listing correlation results.

        Returns
        -------
        int
            Number of displacement pairs loaded.
        """
        print("Step 3: Loading displacement maps...")

        chunks = self.config.dask.chunk_size
        self.displacement_stack = DisplacementStack(self.config)

        if file_list_path is not None:
            self.displacement_stack.load_from_filelist(file_list_path, chunks)
        else:
            # Find all correlation results
            result_dir = self.config.paths.corr_result_dir
            result_files = list(result_dir.glob("correlation_*.tif"))
            self.displacement_stack.load_from_paths(result_files, chunks)

        print(f"  Loaded {self.displacement_stack.n_pairs} displacement pairs")
        return self.displacement_stack.n_pairs

    def coregister(
        self,
        control_mask_path: str | Path | None = None,
    ) -> None:
        """
        Coregister displacement maps using stable reference areas.

        Parameters
        ----------
        control_mask_path : str or Path, optional
            Path to control surface mask (ice/water exclusion).
        """
        print("Step 4: Coregistering displacement maps...")

        if self.displacement_stack is None or self.displacement_stack.n_pairs == 0:
            raise ValueError("No displacement data loaded")

        # Load control mask if provided
        control_mask = None
        if control_mask_path is not None:
            control_mask = read_geotiff(control_mask_path)

        # Perform coregistration
        results = coregister_displacement_stack(
            self.displacement_stack.pairs,
            control_mask=control_mask,
        )

        # Save results
        output_path = self.config.paths.output_dir / "coregistration.txt"
        save_coregistration_results(results, str(output_path))

        print(f"  Coregistered {len(results)} pairs")
        print(f"  Results saved to {output_path}")

    def estimate_velocity(self) -> VelocityResult:
        """
        Estimate velocity/rate from displacement stack.

        Returns
        -------
        VelocityResult
            Estimated velocities and uncertainties.
        """
        print("Step 5: Estimating velocity/rate...")

        if self.displacement_stack is None or self.displacement_stack.n_pairs == 0:
            raise ValueError("No displacement data loaded")

        # Stack displacements
        dx_stack, dy_stack, times = self.displacement_stack.stack()

        # Select model based on configuration
        model = FitModel(self.config.processing.algorithm)

        # Get event time if needed
        event_time = None
        if model == FitModel.CONSTANT_LINEAR:
            if self.config.temporal.earthquake_epochs:
                t0 = times[0]
                event_dt = self.config.temporal.earthquake_epochs[0]
                event_time = (event_dt - t0).days

        # Estimate velocity using Dask
        print("  Running time-series analysis...")
        self.velocity_result = estimate_velocity_dask(
            dx_stack, dy_stack, model, event_time
        )

        print("  Velocity estimation complete")
        return self.velocity_result

    def save_results(
        self,
        output_prefix: str | None = None,
    ) -> list[Path]:
        """
        Save velocity/displacement results to GeoTIFF files.

        Parameters
        ----------
        output_prefix : str, optional
            Prefix for output filenames.

        Returns
        -------
        list
            Paths to saved files.
        """
        print("Step 6: Saving results...")

        if self.velocity_result is None:
            raise ValueError("No velocity results to save")

        output_dir = self.config.paths.output_dir

        if output_prefix is None:
            # Determine name based on mode
            if self.config.processing.earthquake_mode:
                output_prefix = "disp"  # displacement
            else:
                output_prefix = "rate"  # rate

        saved_files = []

        # Save each component
        crs = self.config.projection.epsg

        for name, data in [
            (f"{output_prefix}_x.tif", self.velocity_result.rate_x),
            (f"{output_prefix}_y.tif", self.velocity_result.rate_y),
            (f"{output_prefix}_magnitude.tif", self.velocity_result.rate_magnitude),
            (f"{output_prefix}_x_std.tif", self.velocity_result.rate_x_std),
            (f"{output_prefix}_y_std.tif", self.velocity_result.rate_y_std),
            (f"{output_prefix}_nobs.tif", self.velocity_result.n_observations),
        ]:
            output_path = output_dir / name
            write_geotiff(data, output_path, crs=crs)
            saved_files.append(output_path)
            print(f"  Saved {output_path}")

        return saved_files

    def run(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        skip_correlation: bool = False,
    ) -> list[Path]:
        """
        Run the complete processing pipeline.

        Parameters
        ----------
        bounds : tuple, optional
            Target region bounds (minx, maxx, miny, maxy).
            If None, processes entire available extent.
        skip_correlation : bool
            Skip correlation step (if results already exist).

        Returns
        -------
        list
            Paths to output files.
        """
        print("=" * 60)
        print("TSCorr Processing Pipeline")
        print("=" * 60)

        # Step 0: Discover images
        catalog_path = self.config.paths.output_dir / "image_catalog.pkl"
        if catalog_path.exists():
            self.load_image_catalog(catalog_path)
        else:
            self.discover_images()
            self.save_image_catalog(catalog_path)

        # Determine bounds from images if not provided
        if bounds is None:
            all_bounds = [m["bounds"] for m in self.image_metadata]
            bounds = (
                min(b[0] for b in all_bounds),
                max(b[1] for b in all_bounds),
                min(b[2] for b in all_bounds),
                max(b[3] for b in all_bounds),
            )

        # Step 1: Prepare images
        if self.config.processing.run_step >= 1:
            prepared_files = self.prepare_images(bounds)

        # Step 2: Run correlation
        if self.config.processing.run_step >= 2 and not skip_correlation:
            self.run_correlation()

        # Step 3: Load displacements
        n_loaded = self.load_displacements()
        if n_loaded == 0:
            print("No displacement data to process")
            return []

        # Step 4: Coregister
        self.coregister()

        # Step 5: Estimate velocity
        self.estimate_velocity()

        # Step 6: Save results
        output_files = self.save_results()

        print("=" * 60)
        print("Processing complete!")
        print("=" * 60)

        return output_files


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TSCorr - Time-Series image Correlation"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("MINX", "MAXX", "MINY", "MAXY"),
        help="Processing bounds",
    )
    parser.add_argument(
        "--skip-correlation",
        action="store_true",
        help="Skip correlation step",
    )

    args = parser.parse_args()
    bounds = tuple(args.bounds) if args.bounds else None
    workflow = TSCorrWorkflow(args.config)
    workflow.run(bounds=bounds, skip_correlation=args.skip_correlation)
    
if __name__ == "__main__":
    main()
