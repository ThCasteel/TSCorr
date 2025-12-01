"""
Geospatial utility functions.

Provides coordinate transformations, projection handling, and
polygon operations for geospatial data processing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def xy_to_latlon(
    x: NDArray | float,
    y: NDArray | float,
    crs: str,
) -> tuple[NDArray | float, NDArray | float]:
    """
    Convert projected coordinates to geographic coordinates.

    Parameters
    ----------
    x : ndarray or float
        X coordinates (easting) in meters.
    y : ndarray or float
        Y coordinates (northing) in meters.
    crs : str
        Source coordinate reference system (e.g., "epsg:32637").

    Returns
    -------
    tuple
        (latitude, longitude) in degrees.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


def latlon_to_xy(
    lat: NDArray | float,
    lon: NDArray | float,
    crs: str,
) -> tuple[NDArray | float, NDArray | float]:
    """
    Convert geographic coordinates to projected coordinates.

    Parameters
    ----------
    lat : ndarray or float
        Latitude in degrees.
    lon : ndarray or float
        Longitude in degrees.
    crs : str
        Target coordinate reference system (e.g., "epsg:32637").

    Returns
    -------
    tuple
        (x, y) coordinates in meters.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def compute_overlap_ratio(
    polygon1: NDArray,
    polygon2: NDArray,
    resolution: float = 100.0,
) -> tuple[float, float, float]:
    """
    Compute overlap ratios between two polygons.

    Parameters
    ----------
    polygon1 : ndarray
        First polygon as Nx2 array of [x, y] coordinates.
    polygon2 : ndarray
        Second polygon as Nx2 array of [x, y] coordinates.
    resolution : float
        Grid resolution for rasterization in meters.

    Returns
    -------
    tuple
        (ratio1, ratio2, mean_ratio)
        - ratio1: Overlap as percentage of polygon1
        - ratio2: Overlap as percentage of polygon2
        - mean_ratio: Mean of the two ratios

    Notes
    -----
    Replaces MATLAB getoverlap function.
    """
    from matplotlib.path import Path as MplPath

    # Get bounding box of both polygons
    all_x = np.concatenate([polygon1[:, 0], polygon2[:, 0]])
    all_y = np.concatenate([polygon1[:, 1], polygon2[:, 1]])

    xmin, xmax = all_x.min(), all_x.max()
    ymin, ymax = all_y.min(), all_y.max()

    # Round to resolution
    xmin = np.floor(xmin / resolution) * resolution
    xmax = np.ceil(xmax / resolution) * resolution
    ymin = np.floor(ymin / resolution) * resolution
    ymax = np.ceil(ymax / resolution) * resolution

    # Create grid
    x = np.arange(xmin, xmax, resolution)
    y = np.arange(ymax, ymin, -resolution)  # y decreasing
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Create polygon paths
    path1 = MplPath(polygon1)
    path2 = MplPath(polygon2)

    # Test points
    inside1 = path1.contains_points(points).reshape(xx.shape)
    inside2 = path2.contains_points(points).reshape(xx.shape)

    # Compute overlap
    overlap = inside1 & inside2

    n1 = inside1.sum()
    n2 = inside2.sum()
    n_overlap = overlap.sum()

    ratio1 = (n_overlap / n1 * 100) if n1 > 0 else 0
    ratio2 = (n_overlap / n2 * 100) if n2 > 0 else 0
    mean_ratio = (ratio1 + ratio2) / 2

    return ratio1, ratio2, mean_ratio


def polygon_to_mask(
    polygon: NDArray,
    x: NDArray,
    y: NDArray,
) -> NDArray:
    """
    Convert polygon to raster mask.

    Parameters
    ----------
    polygon : ndarray
        Polygon as Nx2 array of [x, y] coordinates.
    x : ndarray
        X coordinates of output grid.
    y : ndarray
        Y coordinates of output grid.

    Returns
    -------
    ndarray
        Boolean mask where True is inside the polygon.
    """
    from matplotlib.path import Path as MplPath

    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    path = MplPath(polygon)
    mask = path.contains_points(points).reshape(xx.shape)

    return mask


def snap_to_grid(
    value: float,
    resolution: float,
    mode: str = "round",
) -> float:
    """
    Snap a value to the nearest grid point.

    Parameters
    ----------
    value : float
        Value to snap.
    resolution : float
        Grid resolution.
    mode : str
        Snapping mode: "round", "floor", or "ceil".

    Returns
    -------
    float
        Snapped value.
    """
    if mode == "round":
        return round(value / resolution) * resolution
    elif mode == "floor":
        return np.floor(value / resolution) * resolution
    elif mode == "ceil":
        return np.ceil(value / resolution) * resolution
    else:
        raise ValueError(f"Unknown snap mode: {mode}")


def create_output_grid(
    bounds: tuple[float, float, float, float],
    resolution: float,
) -> tuple[NDArray, NDArray]:
    """
    Create output coordinate arrays for a given extent and resolution.

    Parameters
    ----------
    bounds : tuple
        (minx, maxx, miny, maxy) in map coordinates.
    resolution : float
        Grid resolution in meters.

    Returns
    -------
    tuple
        (x, y) coordinate arrays.
    """
    minx, maxx, miny, maxy = bounds

    # Snap to grid
    minx = snap_to_grid(minx, resolution, "floor")
    maxx = snap_to_grid(maxx, resolution, "ceil")
    miny = snap_to_grid(miny, resolution, "floor")
    maxy = snap_to_grid(maxy, resolution, "ceil")

    x = np.arange(minx, maxx + resolution, resolution)
    y = np.arange(maxy, miny - resolution, -resolution)  # y decreasing

    return x, y


def find_images_in_region(
    image_boundaries: list[tuple[NDArray, tuple, str]],
    target_bounds: tuple[float, float, float, float],
    target_crs: str,
    min_overlap: float = 0.0,
) -> list[int]:
    """
    Find images that overlap with a target region.

    Parameters
    ----------
    image_boundaries : list
        List of (boundary_polygon, bounds, crs) tuples for each image.
    target_bounds : tuple
        Target region bounds (minx, maxx, miny, maxy).
    target_crs : str
        Target coordinate reference system.
    min_overlap : float
        Minimum overlap percentage required.

    Returns
    -------
    list
        Indices of images that overlap the target region.
    """
    minx, maxx, miny, maxy = target_bounds
    target_polygon = np.array([
        [minx, miny],
        [maxx, miny],
        [maxx, maxy],
        [minx, maxy],
        [minx, miny],
    ])

    matching_indices = []

    for i, (boundary, bounds, crs) in enumerate(image_boundaries):
        # Reproject if needed
        if crs.lower() != target_crs.lower():
            # Convert boundary to target CRS
            lat, lon = xy_to_latlon(boundary[:, 0], boundary[:, 1], crs)
            x, y = latlon_to_xy(lat, lon, target_crs)
            boundary = np.column_stack([x, y])

        # Check for overlap
        ratio1, ratio2, mean_ratio = compute_overlap_ratio(
            target_polygon, boundary
        )

        if ratio1 >= min_overlap:
            matching_indices.append(i)

    return matching_indices


def date_from_filename(filename: str) -> str | None:
    """
    Extract date string from various filename formats.

    Parameters
    ----------
    filename : str
        Filename to parse.

    Returns
    -------
    str or None
        Date string in YYYYMMDDHHMMSS format, or None if not found.
    """
    import re

    name = Path(filename).stem

    # Try various patterns
    patterns = [
        r"(\d{14})",           # YYYYMMDDHHMMSS
        r"(\d{8})(\d{6})",     # YYYYMMDD_HHMMSS
        r"(\d{8})",            # YYYYMMDD
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return groups[0] + groups[1]
            else:
                date = groups[0]
                if len(date) == 8:
                    return date + "000000"
                return date

    return None


def standardize_filename(
    filepath: str | Path,
    image_type: int,
) -> str:
    """
    Generate standardized output filename from input path.

    Parameters
    ----------
    filepath : str or Path
        Input file path.
    image_type : int
        Image type code (1=Landsat8, 2=ASTER, etc.)

    Returns
    -------
    str
        Standardized filename.

    Notes
    -----
    Follows the naming convention:
    YYYYMMDDHHMMSS_SENSOR_BAND_sub.TIF
    """
    filepath = Path(filepath)
    name = filepath.stem

    # Extract date
    date_str = date_from_filename(name) or "00000000000000"

    # Determine sensor code
    sensor_map = {
        1: "LC08",  # Landsat 8
        2: "AST0",  # ASTER
        3: "PLAN",  # Planet
        4: "PGC0",  # PGC
        5: "LE07",  # Landsat 7
        6: "SUB0",  # Subset
        8: "WV00",  # WorldView
    }
    sensor = sensor_map.get(image_type, "UNKN")

    # Default band
    band = "B8" if image_type in [1, 5] else "B1"

    return f"{date_str}_{sensor}_{band}_sub.TIF"
