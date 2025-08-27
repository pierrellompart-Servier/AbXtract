"""
Core module for antibody descriptor calculations.
"""

from .main import AntibodyDescriptorCalculator
from .config import Config, load_config

__all__ = [
    'AntibodyDescriptorCalculator',
    'Config',
    'load_config'
]