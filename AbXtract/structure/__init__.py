"""
Structure-based descriptor calculation modules for antibodies.

This package provides tools for calculating various structural properties
of antibodies including solvent accessibility, charge distribution,
secondary structure, and molecular interactions.
"""

from .sasa import SASACalculator, SAPCalculator
from .charge import ChargeAnalyzer
from .dssp import DSSPAnalyzer
from .propka import PropkaAnalyzer
from .arpeggio import ArpeggioAnalyzer

__all__ = [
    'SASACalculator',
    'SAPCalculator', 
    'ChargeAnalyzer',
    'DSSPAnalyzer',
    'PropkaAnalyzer',
    'ArpeggioAnalyzer'
]