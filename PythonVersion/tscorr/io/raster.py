"""
Raster I/O operations with lazy loading support via Dask.

This module handles reading and writing of GeoTIFF and ENVI format files,
using rioxarray/xarray for memory-efficient operations.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import rioxarray  # noqa: F401 - needed for rio accessor
import xarray as xr
from numpy.typing import NDArray


@dataclass
class RasterMetadata:
    """Metadata for a raster dataset."""
    width: int
    height: int
    bands: int
    dtype: np.dtype
    crs: str
    transform: tuple[float, ...]
    bounds: tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    resolution: tuple[float, float]  # (dx, dy)
    nodata: float | None


def read_geotiff(
    filepath: str | Path,
    chunks: dict[str, int] | None = None,
    subset: dict[str, slice] | None = None,
) -> xr.DataArray:
    """
    Read a GeoTIFF file as an xarray DataArray with optional lazy loading.

    Parameters
    ----------
    filepath : str or Path
        Path to the GeoTIFF file.
    chunks : dict, optional
        Chunk sizes for dask arrays. If None, data is loaded eagerly.
        Example: {"x": 1024, "y": 1024}
    subset : dict, optional
        Spatial subset to read. Keys are 'x' and 'y' with slice values.
        Example: {"x": slice(100, 200), "y": slice(100, 200)}

    Returns
    -------
    xr.DataArray
        The raster data with coordinates and CRS information.

    Notes
    -----
    This replaces the MATLAB readGeotiff function with memory-efficient
    lazy loading via dask when chunks are specified.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Open with rioxarray for full geospatial support
    da = xr.open_dataarray(
        filepath,
        engine="rasterio",
        chunks=chunks,
    )

    # Apply spatial subset if requested
    if subset:
        if "x" in subset:
            da = da.isel(x=subset["x"])
        if "y" in subset:
            da = da.isel(y=subset["y"])

    # Ensure we have named dimensions
    if "x" not in da.dims:
        da = da.rename({"dim_1": "x", "dim_0": "y"})

    return da


def read_geotiff_subset(
    filepath: str | Path,
    bounds: tuple[float, float, float, float],
    chunks: dict[str, int] | None = None,
) -> xr.DataArray:
    """
    Read a spatial subset of a GeoTIFF file by map coordinates.

    Parameters
    ----------
    filepath : str or Path
        Path to the GeoTIFF file.
    bounds : tuple
        Bounding box as (minx, maxx, miny, maxy) in map coordinates.
    chunks : dict, optional
        Chunk sizes for dask arrays.

    Returns
    -------
    xr.DataArray
        The subset raster data.

    Notes
    -----
    Replaces MATLAB: readGeotiff(file, 'map_subset', [minx maxx miny maxy])
    """
    minx, maxx, miny, maxy = bounds
    
    da = read_geotiff(filepath, chunks=chunks)
    
    # Select by coordinates - note y is typically decreasing
    da = da.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),  # y typically decreasing
    )
    
    return da


def write_geotiff(
    data: xr.DataArray | xr.Dataset | NDArray,
    filepath: str | Path,
    crs: str | None = None,
    transform: Any | None = None,
    nodata: float | None = None,
    dtype: np.dtype | None = None,
    compress: str = "lzw",
) -> None:
    """
    Write data to a GeoTIFF file.

    Parameters
    ----------
    data : xr.DataArray, xr.Dataset, or ndarray
        Data to write. If ndarray, crs and transform are required.
    filepath : str or Path
        Output file path.
    crs : str, optional
        Coordinate reference system (e.g., "EPSG:32637").
    transform : affine.Affine, optional
        Affine transform for the raster.
    nodata : float, optional
        NoData value.
    dtype : dtype, optional
        Output data type.
    compress : str
        Compression method.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, np.ndarray):
        raise ValueError(
            "Raw ndarray requires explicit CRS and transform. "
            "Consider using xarray DataArray with rio accessor."
        )

    # Ensure data is computed if it's a dask array
    if hasattr(data.data, "compute"):
        data = data.compute()

    # Set CRS if provided
    if crs is not None:
        data = data.rio.write_crs(crs)

    # Set nodata if provided
    if nodata is not None:
        data = data.rio.write_nodata(nodata)

    # Write to file
    data.rio.to_raster(
        filepath,
        dtype=dtype,
        compress=compress,
    )


def read_envi(filepath: str | Path, chunks: dict[str, int] | None = None) -> xr.DataArray:
    """
    Read an ENVI format file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ENVI data file (not the .hdr file).
    chunks : dict, optional
        Chunk sizes for dask arrays.

    Returns
    -------
    xr.DataArray
        The raster data.

    Notes
    -----
    Replaces MATLAB enviread function. The ENVI header file (.hdr) is
    automatically located.
    """
    filepath = Path(filepath)
    hdr_path = filepath.with_suffix(".hdr")
    if not hdr_path.exists():
        hdr_path = Path(str(filepath) + ".hdr")

    if not hdr_path.exists():
        raise FileNotFoundError(f"ENVI header not found for: {filepath}")

    # Parse header
    header = _parse_envi_header(hdr_path)

    # Determine data type
    dtype = _envi_dtype(header["data type"])

    # Read binary data
    shape = (header["lines"], header["samples"], header.get("bands", 1))
    interleave = header.get("interleave", "bsq").lower()

    if chunks is not None:
        # Lazy load with dask
        data = da.from_delayed(
            _read_envi_binary(filepath, shape, dtype, interleave, header.get("header offset", 0)),
            shape=shape[:2] if shape[2] == 1 else shape,
            dtype=dtype,
        )
    else:
        # Eager load
        data = _read_envi_binary_eager(
            filepath, shape, dtype, interleave, header.get("header offset", 0)
        )

    # Create coordinates from map info
    x, y = _envi_coordinates(header)

    # Create DataArray
    if shape[2] == 1:
        da_result = xr.DataArray(
            data if len(data.shape) == 2 else data[:, :, 0],
            dims=["y", "x"],
            coords={"y": y, "x": x},
        )
    else:
        da_result = xr.DataArray(
            data,
            dims=["y", "x", "band"],
            coords={"y": y, "x": x, "band": np.arange(shape[2])},
        )

    return da_result


def _parse_envi_header(hdr_path: Path) -> dict[str, Any]:
    """Parse an ENVI header file."""
    header = {}
    with open(hdr_path) as f:
        content = f.read()

    # Handle multi-line values in braces
    pattern = r"(\w+)\s*=\s*(\{[^}]+\}|[^\n]+)"
    for match in re.finditer(pattern, content, re.IGNORECASE):
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()

        # Remove braces and parse
        if value.startswith("{"):
            value = value[1:-1].strip()

        # Try to convert to number
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        header[key] = value

    return header


def _envi_dtype(type_code: int) -> np.dtype:
    """Convert ENVI data type code to numpy dtype."""
    type_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        6: np.complex64,
        9: np.complex128,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    return np.dtype(type_map.get(type_code, np.float32))


def _envi_coordinates(header: dict[str, Any]) -> tuple[NDArray, NDArray]:
    """Extract x, y coordinates from ENVI header map info."""
    samples = header["samples"]
    lines = header["lines"]

    if "map_info" in header:
        # Parse map info string
        parts = [p.strip() for p in str(header["map_info"]).split(",")]
        if len(parts) >= 7:
            mapx = float(parts[3])
            mapy = float(parts[4])
            dx = float(parts[5])
            dy = float(parts[6])
        else:
            mapx, mapy, dx, dy = 0, lines - 1, 1, 1
    else:
        mapx, mapy, dx, dy = 0, lines - 1, 1, 1

    x = mapx + np.arange(samples) * dx
    y = mapy - np.arange(lines) * dy

    return x, y


def _read_envi_binary_eager(
    filepath: Path,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    interleave: str,
    offset: int,
) -> NDArray:
    """Read ENVI binary data eagerly."""
    lines, samples, bands = shape

    with open(filepath, "rb") as f:
        f.seek(offset)
        n_elements = lines * samples * bands
        data = np.fromfile(f, dtype=dtype, count=n_elements)

    # Reshape based on interleave
    if interleave == "bsq":
        data = data.reshape((bands, lines, samples))
        data = np.transpose(data, (1, 2, 0))
    elif interleave == "bil":
        data = data.reshape((lines, bands, samples))
        data = np.transpose(data, (0, 2, 1))
    elif interleave == "bip":
        data = data.reshape((lines, samples, bands))

    return data


def _read_envi_binary(
    filepath: Path,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    interleave: str,
    offset: int,
):
    """Lazy reader for ENVI binary data (returns delayed object)."""
    import dask

    @dask.delayed
    def _read():
        return _read_envi_binary_eager(filepath, shape, dtype, interleave, offset)

    return _read()


def get_image_boundary(filepath: str | Path) -> tuple[NDArray, tuple[float, ...], str]:
    """
    Get boundary polygon and metadata from a raster file.

    Parameters
    ----------
    filepath : str or Path
        Path to the raster file.

    Returns
    -------
    tuple
        (boundary_xy, range, projection)
        - boundary_xy: Nx2 array of [x, y] boundary coordinates
        - range: (minx, maxx, miny, maxy)
        - projection: EPSG code string

    Notes
    -----
    Replaces MATLAB imagebd function.
    """
    filepath = Path(filepath)

    # Use gdalinfo to get metadata
    result = subprocess.run(
        ["gdalinfo", str(filepath)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"gdalinfo failed for {filepath}: {result.stderr}")

    output = result.stdout

    # Parse corner coordinates
    corners = {}
    for corner, name in [
        ("Upper Left", "ul"),
        ("Lower Left", "ll"),
        ("Upper Right", "ur"),
        ("Lower Right", "lr"),
    ]:
        pattern = rf"{corner}\s*\(\s*([\d.-]+),\s*([\d.-]+)\)"
        match = re.search(pattern, output)
        if match:
            corners[name] = (float(match.group(1)), float(match.group(2)))

    if len(corners) != 4:
        raise ValueError(f"Could not parse all corners from {filepath}")

    # Build boundary polygon (closed)
    boundary = np.array([
        corners["ll"],
        corners["lr"],
        corners["ur"],
        corners["ul"],
        corners["ll"],  # close the polygon
    ])

    # Calculate range
    x_coords = boundary[:, 0]
    y_coords = boundary[:, 1]
    bounds = (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max())

    # Extract EPSG code
    epsg_match = re.search(r'ID\["EPSG",(\d+)', output)
    if epsg_match:
        projection = f"epsg:{epsg_match.group(1)}"
    else:
        projection = "unknown"

    return boundary, bounds, projection


def reproject_bounds(
    bounds: tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float, float, float]:
    """
    Reproject bounding box coordinates.

    Parameters
    ----------
    bounds : tuple
        (minx, maxx, miny, maxy) in source CRS.
    src_crs : str
        Source CRS (e.g., "epsg:32637").
    dst_crs : str
        Destination CRS.

    Returns
    -------
    tuple
        Reprojected bounds in destination CRS.
    """
    from pyproj import Transformer

    minx, maxx, miny, maxy = bounds
    
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    # Transform corner points
    corners_x = [minx, maxx, maxx, minx]
    corners_y = [miny, miny, maxy, maxy]
    
    new_x, new_y = transformer.transform(corners_x, corners_y)
    
    return (min(new_x), max(new_x), min(new_y), max(new_y))
