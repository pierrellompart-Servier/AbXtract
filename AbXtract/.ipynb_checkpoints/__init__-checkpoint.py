"""
Antibody Descriptors - Comprehensive antibody analysis toolkit.

This package provides tools for calculating sequence and structure-based
descriptors for antibodies and nanobodies, including developability metrics,
physicochemical properties, and structural features.
"""

from .core import (
    AntibodyDescriptorCalculator,
    Config,
    load_config
)
from .core.main import AntibodyDescriptorCalculator

from .__version__ import __version__

# Make key classes available at package level
__all__ = [
    'AntibodyDescriptorCalculator',
    'Config',
    'load_config',
    '__version__'
]
