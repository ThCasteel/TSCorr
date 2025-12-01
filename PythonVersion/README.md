# TSCorr - Time-Series image Correlation Software

A Python implementation of the TSCorr toolbox for detecting and measuring ground deformation (landslides, earthquakes, volcanic activity) from satellite imagery.

## Features

- **Multi-sensor support**: Landsat 7/8, ASTER, Planet, WorldView, and PGC imagery
- **Memory-efficient processing**: Lazy loading via Dask for large datasets
- **Unified configuration**: Single YAML file for all parameters
- **Multiple analysis modes**: Landslide velocity, earthquake displacement, volcanic deformation
- **Time-series models**: Linear, constant, and step-function fitting

## Installation

```bash
# Clone the repository
git clone https://github.com/tscorr/tscorr.git
cd tscorr

# Install with pip (recommended)
pip install -e .

# Or with conda environment
conda env create -f environment.yml
conda activate tscorr
pip install -e .
```

### Dependencies

Core dependencies:
- Python >= 3.10
- NumPy, SciPy
- xarray, Dask
- rioxarray, rasterio
- pyproj
- PyYAML

External tools (for correlation):
- GDAL command-line tools
- COSI-Corr Plus (optional)

## Quick Start

### 1. Create Configuration File

Copy and edit the example configuration:

```bash
cp config.yaml my_config.yaml
```

Edit `my_config.yaml` to set:
- `paths.image_dir`: Directory containing your satellite images
- `projection.epsg`: Target coordinate system (e.g., "epsg:32637")
- `processing.earthquake_mode`: `true` for earthquakes, `false` for landslides

### 2. Run Processing

```python
from tscorr import TSCorrWorkflow

# Initialize workflow
workflow = TSCorrWorkflow("my_config.yaml")

# Run complete pipeline
workflow.run()

# Or run steps individually:
workflow.discover_images()
workflow.prepare_images(bounds=(minx, maxx, miny, maxy))
workflow.load_displacements()
workflow.coregister()
result = workflow.estimate_velocity()
workflow.save_results()
```

### 3. Command Line Interface

```bash
# Run full workflow
tscorr -c my_config.yaml

# With specific bounds
tscorr -c my_config.yaml --bounds 430000 450000 6770000 6790000

# Skip correlation if results exist
tscorr -c my_config.yaml --skip-correlation
```

## Configuration Reference

All configuration is centralized in a single YAML file. Key sections:

### Paths
```yaml
paths:
  image_dir: "~/data/imagery/"      # Input images
  output_dir: "./outputs/"          # Results
  corr_result_dir: "./results/"     # Correlation outputs
```

### Image Settings
```yaml
image:
  type: 1                           # 1=Landsat8, 2=ASTER, 3=Planet, 8=WorldView
  resolution: 15                    # Image resolution in meters
```

### Processing
```yaml
processing:
  tool: 4                           # 4=COSI-Corr Plus
  earthquake_mode: false            # true for earthquakes
  algorithm: 2                      # 1=linear, 2=constant, 3=step+linear
  output_resolution: 60             # Output resolution in meters
```

### Temporal Filtering
```yaml
temporal:
  min_date: "2020/01/01"           # Start date
  max_date: null                    # End date (null = no limit)
  season_start_month: 1             # Snow-free season start
  season_end_month: 12              # Snow-free season end
```

### Dask Configuration
```yaml
dask:
  scheduler: "threads"              # "threads", "processes", or "synchronous"
  num_workers: 4                    # Number of parallel workers
  memory_limit: "8GB"               # Memory per worker
  chunk_size:                       # Array chunk sizes
    x: 1024
    y: 1024
```

## Module Overview

```
tscorr/
├── config.py          # Configuration loading and validation
├── workflow.py        # Main processing pipeline
├── io/
│   └── raster.py      # GeoTIFF/ENVI I/O with Dask support
├── processing/
│   ├── displacement.py    # Displacement map loading/filtering
│   ├── velocity.py        # Time-series velocity estimation
│   └── coregistration.py  # Offset estimation and removal
└── utils/
    └── geospatial.py  # Coordinate transforms, polygon ops
```

## API Examples

### Reading Raster Data
```python
from tscorr.io import read_geotiff, read_geotiff_subset

# Lazy load with Dask
data = read_geotiff("image.tif", chunks={"x": 1024, "y": 1024})

# Load spatial subset
subset = read_geotiff_subset("image.tif", bounds=(minx, maxx, miny, maxy))
```

### Displacement Processing
```python
from tscorr.processing import DisplacementStack, FitModel

# Load displacement stack
stack = DisplacementStack(config)
stack.load_from_filelist("corrfilelist.txt")

# Estimate velocity
from tscorr.processing import estimate_velocity_dask
dx_stack, dy_stack, times = stack.stack()
result = estimate_velocity_dask(dx_stack, dy_stack, model=FitModel.CONSTANT)

# Access results
print(result.rate_magnitude.mean().values)  # Mean rate
result.rate_x.rio.to_raster("rate_x.tif")   # Save to file
```

### Coordinate Transformations
```python
from tscorr.utils import xy_to_latlon, latlon_to_xy

# Project to geographic
lat, lon = xy_to_latlon(x, y, "epsg:32637")

# Project to UTM
x, y = latlon_to_xy(lat, lon, "epsg:32637")
```

## Migration from MATLAB

This Python implementation replaces the original MATLAB TSCorr toolbox. Key differences:

| MATLAB | Python |
|--------|--------|
| `constant.m` | `config.yaml` |
| `mat0.mat` | `image_catalog.pkl` |
| `readGeotiff()` | `tscorr.io.read_geotiff()` |
| `enviread()` | `tscorr.io.read_envi()` |
| Workspace variables | Explicit function arguments |
| Scripts with `addpath` | Installable package |
| Manual loops | Vectorized operations |
| Single-threaded | Parallel via Dask |

### Numerical Validation

The Python implementation maintains numerical compatibility with MATLAB:
- Float64 tolerance: 1e-8 (absolute)
- Float32 tolerance: 1e-5 (absolute)
- Geophysical fields: 1e-6 (relative)

## Reference

If you use TSCorr, please cite:

> Dai, C., Higman, B., Lynett, P.J., Jacquemart, M., Howat, I.M., et al. (2020).
> Detection and assessment of a large and potentially tsunamigenic periglacial
> landslide in Barry Arm, Alaska. *Geophysical Research Letters*, 47(22), e2020GL089800.
> https://doi.org/10.1029/2020GL089800

## License

MIT License - see LICENSE file for details.
