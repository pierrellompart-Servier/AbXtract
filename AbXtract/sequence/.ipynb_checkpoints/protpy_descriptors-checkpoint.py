"""
ProtPy-based descriptor calculation module.

This module provides comprehensive protein descriptor calculations using the protpy library,
including amino acid composition, dipeptide composition, autocorrelation features, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Try to import protpy, handle if not installed
try:
    import protpy
    PROTPY_AVAILABLE = True
except ImportError:
    PROTPY_AVAILABLE = False
    logger.warning("protpy not installed. ProtPy descriptors will not be available.")


class ProtPyDescriptorCalculator:
    """
    Calculator for ProtPy-based protein descriptors.
    
    ProtPy provides a comprehensive set of protein descriptors including:
    - Amino acid composition
    - Dipeptide composition
    - Autocorrelation features (Moreau-Broto, Moran, Geary)
    - Conjoint triad features
    - Quasi-sequence order
    - Pseudo amino acid composition
    
    Parameters
    ----------
    lag : int
        Lag value for autocorrelation calculations (default: 30)
    weight : float
        Weight parameter for pseudo amino acid composition (default: 0.05)
    distance_matrix : str
        Distance matrix type for autocorrelation (default: 'schneider-wrede')
    
    Examples
    --------
    >>> calc = ProtPyDescriptorCalculator()
    >>> descriptors = calc.calculate("MVLQTQVFISLLLWISGAYG")
    >>> print(f"Number of descriptors: {len(descriptors)}")
    """
    
    def __init__(self, 
                 lag: int = 30,
                 weight: float = 0.05,
                 distance_matrix: str = 'schneider-wrede'):
        """Initialize ProtPy descriptor calculator."""
        self.lag = lag
        self.weight = weight
        self.distance_matrix = distance_matrix
        
        if not PROTPY_AVAILABLE:
            logger.warning("ProtPy is not available. Install with: pip install protpy")
    
    def calculate(self, sequence: str) -> Dict[str, float]:
        """
        Calculate all ProtPy descriptors for a single sequence.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence
            
        Returns
        -------
        dict
            Dictionary of descriptor names and values
        """
        if not PROTPY_AVAILABLE:
            logger.warning("ProtPy not available, returning empty descriptors")
            return {}
        
        sequence = sequence.upper()
        descriptors = {}
        
        try:
            # Amino acid composition
            aa_comp = protpy.amino_acid_composition(sequence)
            for aa, value in aa_comp.items():
                descriptors[f'AAC_{aa}'] = value
            
            # Dipeptide composition
            dipep_comp = protpy.dipeptide_composition(sequence)
            for dipep, value in dipep_comp.items():
                descriptors[f'DPC_{dipep}'] = value
            
            # Moreau-Broto autocorrelation
            try:
                moreaubroto = protpy.moreaubroto_autocorrelation(sequence, lag=self.lag)
                for key, value in moreaubroto.items():
                    descriptors[f'MoreauBroto_{key}'] = value
            except Exception as e:
                logger.debug(f"Moreau-Broto calculation failed: {e}")
            
            # Moran autocorrelation
            try:
                moran = protpy.moran_autocorrelation(sequence, lag=self.lag)
                for key, value in moran.items():
                    descriptors[f'Moran_{key}'] = value
            except Exception as e:
                logger.debug(f"Moran calculation failed: {e}")
            
            # Geary autocorrelation
            try:
                geary = protpy.geary_autocorrelation(sequence, lag=self.lag)
                for key, value in geary.items():
                    descriptors[f'Geary_{key}'] = value
            except Exception as e:
                logger.debug(f"Geary calculation failed: {e}")
            
            # Conjoint triad
            try:
                conjoint = protpy.conjoint_triad(sequence)
                for key, value in conjoint.items():
                    descriptors[f'ConjointTriad_{key}'] = value
            except Exception as e:
                logger.debug(f"Conjoint triad calculation failed: {e}")
            
            # Quasi-sequence order
            try:
                qso = protpy.quasi_sequence_order(sequence, lag=self.lag, weight=self.weight)
                for key, value in qso.items():
                    descriptors[f'QSO_{key}'] = value
            except Exception as e:
                logger.debug(f"Quasi-sequence order calculation failed: {e}")
            
            # Pseudo amino acid composition
            try:
                pseaac = protpy.pseudo_amino_acid_composition(sequence, lamda=self.lag, weight=self.weight)
                for key, value in pseaac.items():
                    descriptors[f'PseAAC_{key}'] = value
            except Exception as e:
                logger.debug(f"Pseudo AAC calculation failed: {e}")
            
            # Sequence order coupling numbers
            try:
                socn = protpy.sequence_order_coupling_number(sequence, lag=self.lag)
                for key, value in socn.items():
                    descriptors[f'SOCN_{key}'] = value
            except Exception as e:
                logger.debug(f"SOCN calculation failed: {e}")
            
        except Exception as e:
            logger.error(f"ProtPy descriptor calculation failed: {e}")
        
        return descriptors
    
    def calculate_for_chains(self,
                           heavy_sequence: Optional[str] = None,
                           light_sequence: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate descriptors for antibody chains.
        
        Parameters
        ----------
        heavy_sequence : str, optional
            Heavy chain sequence
        light_sequence : str, optional
            Light chain sequence
            
        Returns
        -------
        dict
            Dictionary with descriptors for each chain and combined
        """
        results = {}
        
        if heavy_sequence:
            heavy_desc = self.calculate(heavy_sequence)
            for key, value in heavy_desc.items():
                results[f'Heavy_{key}'] = value
        
        if light_sequence:
            light_desc = self.calculate(light_sequence)
            for key, value in light_desc.items():
                results[f'Light_{key}'] = value
        
        # Combined sequence descriptors
        if heavy_sequence and light_sequence:
            combined_seq = heavy_sequence + light_sequence
            combined_desc = self.calculate(combined_seq)
            for key, value in combined_desc.items():
                results[f'Combined_{key}'] = value
        
        return results
    
    def calculate_summary_statistics(self, sequence: str) -> Dict[str, float]:
        """
        Calculate summary statistics of ProtPy descriptors.
        
        Instead of returning all descriptors, returns summary statistics
        to reduce dimensionality.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence
            
        Returns
        -------
        dict
            Summary statistics of descriptors
        """
        full_descriptors = self.calculate(sequence)
        
        if not full_descriptors:
            return {}
        
        summary = {}
        
        # Group descriptors by type
        descriptor_groups = {
            'AAC': [],
            'DPC': [],
            'MoreauBroto': [],
            'Moran': [],
            'Geary': [],
            'ConjointTriad': [],
            'QSO': [],
            'PseAAC': [],
            'SOCN': []
        }
        
        # Categorize descriptors
        for key, value in full_descriptors.items():
            for group_name in descriptor_groups:
                if key.startswith(group_name):
                    descriptor_groups[group_name].append(value)
                    break
        
        # Calculate statistics for each group
        for group_name, values in descriptor_groups.items():
            if values:
                summary[f'{group_name}_mean'] = np.mean(values)
                summary[f'{group_name}_std'] = np.std(values)
                summary[f'{group_name}_max'] = np.max(values)
                summary[f'{group_name}_min'] = np.min(values)
        
        return summary
    
    def calculate_for_dataframe(self,
                              df: pd.DataFrame,
                              sequence_column: str = 'sequence',
                              use_summary: bool = True) -> pd.DataFrame:
        """
        Calculate descriptors for sequences in a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sequences
        sequence_column : str
            Name of column containing sequences
        use_summary : bool
            If True, calculate summary statistics instead of all descriptors
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added descriptor columns
        """
        result_df = df.copy()
        
        for idx, row in df.iterrows():
            seq = row[sequence_column]
            
            if use_summary:
                descriptors = self.calculate_summary_statistics(seq)
            else:
                descriptors = self.calculate(seq)
            
            for key, value in descriptors.items():
                result_df.at[idx, key] = value
        
        return result_df
    
    def get_descriptor_names(self, sequence_example: str = "ACDEFGHIKLMNPQRSTVWY") -> List[str]:
        """
        Get list of all descriptor names that will be calculated.
        
        Parameters
        ----------
        sequence_example : str
            Example sequence to determine descriptor names
            
        Returns
        -------
        list
            List of descriptor names
        """
        descriptors = self.calculate(sequence_example)
        return list(descriptors.keys())
    
    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate if sequence is suitable for ProtPy calculations.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence
            
        Returns
        -------
        bool
            True if sequence is valid
        """
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        sequence = sequence.upper()
        
        # Check for invalid characters
        if not all(aa in valid_amino_acids for aa in sequence):
            return False
        
        # Check minimum length (some descriptors need minimum length)
        if len(sequence) < 3:
            return False
        
        return True