"""
I/O operations for TSCorr.

Provides functions for reading and writing geospatial raster data
with lazy loading support via Dask/xarray.
"""

from .raster import (
    read_geotiff,
    read_geotiff_subset,
    write_geotiff,
    read_envi,
    get_image_boundary,
    reproject_bounds,
    RasterMetadata,
)

__all__ = [
    "read_geotiff",
    "read_geotiff_subset",
    "write_geotiff",
    "read_envi",
    "get_image_boundary",
    "reproject_bounds",
    "RasterMetadata",
]
