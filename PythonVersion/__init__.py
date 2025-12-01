"""
TSCorr - Time-Series image Correlation software.

A Python implementation of the TSCorr MATLAB toolbox for detecting
and measuring ground deformation from satellite imagery.

Features
--------
- Multi-sensor support (Landsat, ASTER, Planet, WorldView)
- Lazy loading via Dask for memory-efficient processing
- YAML-based configuration
- Time-series analysis for velocity/displacement estimation

Example
-------
>>> from tscorr import TSCorrWorkflow, load_config
>>> 
>>> # Load configuration
>>> config = load_config("config.yaml")
>>> 
>>> # Run workflow
>>> workflow = TSCorrWorkflow("config.yaml")
>>> workflow.run()

Reference
---------
Dai, C., Higman, B., Lynett, P.J., et al. (2020). Detection and assessment
of a large and potentially tsunamigenic periglacial landslide in Barry Arm,
Alaska. Geophysical Research Letters, 47(22), e2020GL089800.
"""

from .config import TSCorrConfig, load_config
from .workflow import TSCorrWorkflow

__version__ = "0.1.0"
__author__ = "TSCorr Contributors"

__all__ = [
    "TSCorrConfig",
    "load_config",
    "TSCorrWorkflow",
    "__version__",
]
