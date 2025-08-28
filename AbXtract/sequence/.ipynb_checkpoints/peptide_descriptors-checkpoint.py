"""
Comprehensive peptide descriptor calculation module.

This module calculates 102+ peptide descriptors including:
- Atchley factors
- BLOSUM indices  
- Cruciani properties
- Kidera factors
- Z-scales
- And many more physicochemical properties

Uses the peptides.py library for comprehensive feature extraction.

References
----------
.. [1] https://peptides.readthedocs.io/
.. [2] Osorio et al. (2015) "Peptides: A package for data mining of antimicrobial peptides"
"""

import pandas as pd
import numpy as np
import peptides
from typing import List, Dict, Union, Optional
import logging
import protpy

logger = logging.getLogger(__name__)


class PeptideDescriptorCalculator:
    """
    Calculator for comprehensive peptide descriptors.
    
    This class provides methods to calculate 102+ descriptors covering
    various physicochemical, structural, and compositional properties.
    
    Attributes
    ----------
    results : pd.DataFrame
        DataFrame containing calculated descriptors
    descriptor_mapping : dict
        Mapping of descriptor codes to readable names
        
    Examples
    --------
    >>> calc = PeptideDescriptorCalculator()
    >>> results = calc.calculate_all("MVLQTQVFISLLLWISGAYG")
    >>> print(results.columns)
    """
    
    def __init__(self, pH: float = 7.4, window: int = 5, angle: int = 100):
        """Initialize the calculator.

        Parameters
        ----------
        pH : float
            pH for charge calculations (default: 7.4)
        window : int
            Window size for hydrophobic moment (default: 5)
        angle : int
            Angle for hydrophobic moment (default: 100)
        """
        self.results = None
        self.descriptor_mapping = self._get_descriptor_mapping()
        self.pH = pH
        self.window = window
        self.angle = angle
        
        
    def _get_descriptor_mapping(self) -> Dict[str, str]:
        """Get mapping of descriptor codes to readable names."""
        return {
            # Atchley Factors
            'AF1': 'AF1_Polarity_Accessibility_Hydrophobicity',
            'AF2': 'AF2_Secondary_Structure_Propensity',
            'AF3': 'AF3_Molecular_Size',
            'AF4': 'AF4_Codon_Composition',
            'AF5': 'AF5_Electrostatic_Charge',
            
            # BLOSUM Indices
            'BLOSUM1': 'BLOSUM1_Helix_Index',
            'BLOSUM2': 'BLOSUM2_Sheet_Index',
            'BLOSUM3': 'BLOSUM3_Turn_Index',
            'BLOSUM4': 'BLOSUM4_Hydrophobicity_Index',
            'BLOSUM5': 'BLOSUM5_Electronic_Index',
            'BLOSUM6': 'BLOSUM6_Aliphatic_Index',
            'BLOSUM7': 'BLOSUM7_Aromatic_Index',
            'BLOSUM8': 'BLOSUM8_Polar_Index',
            'BLOSUM9': 'BLOSUM9_Charged_Index',
            'BLOSUM10': 'BLOSUM10_Small_Index',
            
            # Cruciani Properties
            'PP1': 'PP1_Polarity',
            'PP2': 'PP2_Hydrophobicity',
            'PP3': 'PP3_H_Bonding',
            
            # FASGAI Vectors
            'F1': 'F1_Hydrophobicity_FASGAI',
            'F2': 'F2_Alpha_Turn_Propensity',
            'F3': 'F3_Bulky_Properties',
            'F4': 'F4_Compositional_Characteristic',
            'F5': 'F5_Local_Flexibility',
            'F6': 'F6_Electronic_Properties',
            
            # Kidera Factors
            'KF1': 'KF1_Helix_Bend_Preference',
            'KF2': 'KF2_Side_Chain_Size',
            'KF3': 'KF3_Extended_Structure_Preference',
            'KF4': 'KF4_Hydrophobicity',
            'KF5': 'KF5_Double_Bend_Preference',
            'KF6': 'KF6_Partial_Specific_Volume',
            'KF7': 'KF7_Flat_Extended_Preference',
            'KF8': 'KF8_Alpha_Region_Occurrence',
            'KF9': 'KF9_pK_C',
            'KF10': 'KF10_Surrounding_Hydrophobicity',
            
            # Z-Scales
            'Z1': 'Z1_Lipophilicity',
            'Z2': 'Z2_Steric_Bulk_Polarizability',
            'Z3': 'Z3_Polarity_Charge',
            'Z4': 'Z4_Electronegativity_Heat_Formation',
            'Z5': 'Z5_Electrophilicity_Hardness'
        }
    
    def calculate_all(self, 
                     heavy_sequence: Optional[str] = None,
                     light_sequence: Optional[str] = None,
                     pH: Optional[float] = None,
                     window: Optional[int] = None,
                     angle: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate all peptide descriptors for antibody sequences.
        
        Parameters
        ----------
        heavy_sequence : str, optional
            Heavy chain sequence
        light_sequence : str, optional
            Light chain sequence
        pH : float
            pH for charge calculations
        window : int
            Window size for hydrophobic moment
        angle : int
            Angle for hydrophobic moment
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all calculated descriptors
        """

        # Use instance defaults if not provided
        pH = pH if pH is not None else self.pH
        window = window if window is not None else self.window
        angle = angle if angle is not None else self.angle

        results = []
        
        if heavy_sequence:
            heavy_desc = self._calculate_single_sequence(
                heavy_sequence, "Heavy", pH, window, angle
            )
            results.append(heavy_desc)
        
        if light_sequence:
            light_desc = self._calculate_single_sequence(
                light_sequence, "Light", pH, window, angle
            )
            results.append(light_desc)
        
        if not results:
            raise ValueError("At least one sequence must be provided")
        
        self.results = pd.DataFrame(results)
        
        # post-process: handle "propty*" columns
        for col in self.results.columns:
            if col.startswith("propty"):
                self.results[col] = self.results[col].apply(
                    lambda x: float(str(x).split()[1]) if isinstance(x, str) and len(x.split()) > 1 else x
                )

        for col in self.results.columns:
            if col.startswith("Peptide_propty"):
                self.results[col] = self.results[col].apply(
                    lambda x: float(str(x).split()[1]) if isinstance(x, str) and len(x.split()) > 1 else x
                )
        return self.results
    
    def _calculate_single_sequence(self, 
                                  sequence: str,
                                  chain_type: str,
                                  pH: float,
                                  window: int,
                                  angle: int) -> Dict:
        """Calculate descriptors for a single sequence."""
        sequence = sequence.upper()
        p = peptides.Peptide(sequence)
        
        descriptors = {
            'chain': chain_type,
            'sequence': sequence,
            'length': len(sequence)
        }
        
        # Basic global descriptors
        descriptors.update(self._calculate_basic_descriptors(p, pH))
        
        # Complex descriptors (102 features)
        descriptors.update(self._calculate_complex_descriptors(p))
        
        # Amino acid composition
        descriptors.update(self._calculate_aa_composition(sequence))
        
        # Protpy descriptors
        descriptors.update(self._calculate_protpy_descriptors(sequence))
        
        # Hydrophobic moment
        descriptors[f'hydrophobic_moment_w{window}_a{angle}'] = self._safe_calc(
            lambda: p.hydrophobic_moment(window=window, angle=angle)
        )
        
        return descriptors
    
    def _calculate_basic_descriptors(self, p: peptides.Peptide, pH: float) -> Dict:
        """Calculate basic peptide descriptors."""
        return {
            'aliphatic_index': self._safe_calc(p.aliphatic_index),
            'boman': self._safe_calc(p.boman),
            f'charge_pH{pH}': self._safe_calc(lambda: p.charge(pH=pH)),
            'isoelectric_point': self._safe_calc(p.isoelectric_point),
            'instability_index': self._safe_calc(p.instability_index),
            'molecular_weight': self._safe_calc(p.molecular_weight),
            'molar_extinction_reduced': self._safe_calc(
                lambda: p.molar_extinction_coefficient()[0]
            ),
            'molar_extinction_cystines': self._safe_calc(
                lambda: p.molar_extinction_coefficient()[1]
            ),
            # 'gravy': self._safe_calc(p.gravy),
            # 'aromaticity': self._safe_calc(p.aromaticity)
        }
    
    def _calculate_complex_descriptors(self, p: peptides.Peptide) -> Dict:
        """Calculate 102 complex peptide descriptors."""
        complex_desc = {}
        
        try:
            raw_descriptors = p.descriptors()
            
            # Map to readable names
            for original_name, value in raw_descriptors.items():
                readable_name = self.descriptor_mapping.get(original_name, original_name)
                complex_desc[readable_name] = value
        except Exception as e:
            logger.warning(f"Failed to calculate complex descriptors: {e}")
            # Return NaN for all expected descriptors
            for readable_name in self.descriptor_mapping.values():
                complex_desc[readable_name] = np.nan
        
        return complex_desc
    
    def _calculate_aa_composition(self, sequence: str) -> Dict:
        """Calculate amino acid composition."""
        aa_comp = {}
        
        # Standard amino acids
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Counts
        for aa in amino_acids:
            aa_comp[f'count_{aa}'] = sequence.count(aa)
        
        # Frequencies
        seq_len = len(sequence)
        if seq_len > 0:
            for aa in amino_acids:
                aa_comp[f'frequency_{aa}'] = sequence.count(aa) / seq_len
        else:
            for aa in amino_acids:
                aa_comp[f'frequency_{aa}'] = 0
        
        return aa_comp

    def _calculate_protpy_descriptors(self, sequence: str) -> Dict:
        """Calculate protpy-specific descriptors with ALL features (not reduced)."""
        protpy_desc = {}
        
        def extract_value(val):
            """Extract scalar value from protpy output (handles Series, arrays, etc.)."""
            try:
                # Handle pandas Series/DataFrame
                if hasattr(val, 'iloc'):
                    if len(val) == 1:
                        return float(val.iloc[0])
                    elif len(val) > 1:
                        # For Series with multiple values, take the first
                        return float(val.iloc[0])
                    else:
                        return np.nan
                # Handle numpy arrays
                elif isinstance(val, np.ndarray):
                    if val.size == 1:
                        return float(val.item())
                    elif val.size > 1:
                        return float(val[0])
                    else:
                        return np.nan
                # Handle lists/tuples
                elif isinstance(val, (list, tuple)):
                    if len(val) >= 1:
                        return float(val[0])
                    else:
                        return np.nan
                # Handle scalar values
                else:
                    return float(val)
            except (ValueError, TypeError, AttributeError, IndexError):
                return np.nan
        
        
        try:
            # Amino acid composition - keep all features
            aa_comp = protpy.amino_acid_composition(sequence)
            for k, v in aa_comp.items():
                protpy_desc[f'protpy_aa_{k}'] = extract_value(v)

            # Dipeptide composition - keep all features (not just summary stats)
            dipep_comp = protpy.dipeptide_composition(sequence)
            for k, v in dipep_comp.items():
                protpy_desc[f'protpy_dipep_{k}'] = extract_value(v)

            # Autocorrelation features - keep all individual features
            try:
                moreaubroto = protpy.moreaubroto_autocorrelation(sequence)
                for k, v in moreaubroto.items():
                    protpy_desc[f'protpy_moreaubroto_{k}'] = extract_value(v)
            except:
                # If it fails, we can still add NaN placeholders for expected features
                # Moreaubroto typically has features for different properties and lags
                for i in range(1, 31):  # Usually 30 features
                    protpy_desc[f'protpy_moreaubroto_{i}'] = np.nan

            try:
                moran = protpy.moran_autocorrelation(sequence)
                for k, v in moran.items():
                    protpy_desc[f'protpy_moran_{k}'] = extract_value(v)
            except:
                # Moran typically has features for different properties and lags
                for i in range(1, 31):  # Usually 30 features
                    protpy_desc[f'protpy_moran_{i}'] = np.nan

            try:
                geary = protpy.geary_autocorrelation(sequence)
                for k, v in geary.items():
                    protpy_desc[f'protpy_geary_{k}'] = extract_value(v)
            except:
                # Geary typically has features for different properties and lags
                for i in range(1, 31):  # Usually 30 features
                    protpy_desc[f'protpy_geary_{i}'] = np.nan

            # Conjoint triad - keep all features (not just mean)
            try:
                conjoint = protpy.conjoint_triad(sequence)
                for k, v in conjoint.items():
                    protpy_desc[f'protpy_conjoint_{k}'] = extract_value(v)
            except:
                # Conjoint triad typically has 343 features (7^3 combinations)
                logger.warning(f"Failed to calculate conjoint triad for sequence")
                # You can add placeholder features if needed

            # Quasi-sequence order - keep all features (not just mean)
            try:
                qso = protpy.quasi_sequence_order(sequence)
                for k, v in qso.items():
                    protpy_desc[f'protpy_qso_{k}'] = extract_value(v)
            except:
                logger.warning(f"Failed to calculate quasi-sequence order for sequence")
                # You can add placeholder features if needed

            # Additional protpy features you might want to include:

            # Tripeptide composition
            try:
                tripep_comp = protpy.tripeptide_composition(sequence)
                for k, v in tripep_comp.items():
                    protpy_desc[f'protpy_tripep_{k}'] = extract_value(v)
            except:
                logger.warning(f"Failed to calculate tripeptide composition for sequence")

            # Composition, Transition, Distribution (CTD)
            # try:
            #     ctd = protpy.ctd(sequence)
            #     for k, v in ctd.items():
            #         protpy_desc[f'protpy_ctd_{k}'] = v
            # except:
            #     logger.warning(f"Failed to calculate CTD for sequence")

            # Pseudo amino acid composition
            try:
                paac = protpy.pseudo_amino_acid_composition(sequence)
                for k, v in paac.items():
                    protpy_desc[f'protpy_paac_{k}'] = extract_value(v)
            except:
                logger.warning(f"Failed to calculate pseudo amino acid composition for sequence")

            # Amphiphilic pseudo amino acid composition
            try:
                apaac = protpy.amphiphilic_pseudo_amino_acid_composition(sequence)
                for k, v in apaac.items():
                    protpy_desc[f'protpy_apaac_{k}'] = extract_value(v)
            except:
                logger.warning(f"Failed to calculate amphiphilic pseudo amino acid composition for sequence")

        except Exception as e:
            logger.warning(f"Failed to calculate protpy descriptors: {e}")

        return protpy_desc
    
    def _safe_calc(self, func):
        """Safely calculate a descriptor, returning NaN on failure."""
        try:
            return func()
        except:
            return np.nan
    
    def get_profiles(self, 
                    sequence: str,
                    window: int = 1) -> Dict[str, np.ndarray]:
        """
        Calculate per-residue profiles for a sequence.
        
        Parameters
        ----------
        sequence : str
            Input sequence
        window : int
            Window size for profile calculations
            
        Returns
        -------
        dict
            Dictionary of profile arrays
        """
        p = peptides.Peptide(sequence)
        profiles = {}
        
        # Different hydrophobicity scales
        scales = {
            'charge_sign': 'sign',
            'hydrophobicity_kd': 'KyteDoolitle',
            'hydrophobicity_hw': 'HoppWoods',
            'hydrophobicity_cornette': 'Cornette',
            'hydrophobicity_eisenberg': 'Eisenberg',
            'hydrophobicity_rose': 'Rose',
            'hydrophobicity_janin': 'Janin',
            'hydrophobicity_engelman': 'Engelman'
        }
        
        for profile_name, scale_name in scales.items():
            try:
                # Get the scale from peptides tables
                if 'charge' in profile_name:
                    scale = getattr(peptides.tables.CHARGE, scale_name, {})
                else:
                    scale = getattr(peptides.tables.HYDROPHOBICITY, scale_name, {})
                
                if scale:
                    profiles[profile_name] = p.profile(scale, window=window)
            except:
                profiles[profile_name] = np.array([])
        
        return profiles
    
    def analyze_regions(self,
                       sequence: str,
                       regions: List[tuple[int, int]],
                       region_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze specific regions of a sequence.
        
        Parameters
        ----------
        sequence : str
            Input sequence
        regions : list of tuples
            List of (start, end) positions for regions
        region_names : list of str, optional
            Names for the regions
            
        Returns
        -------
        pd.DataFrame
            Descriptors calculated for each region
        """
        if not region_names:
            region_names = [f"Region_{i+1}" for i in range(len(regions))]
        
        results = []
        for (start, end), name in zip(regions, region_names):
            region_seq = sequence[start:end+1]
            desc = self._calculate_single_sequence(
                region_seq, name, pH=7.4, window=5, angle=100
            )
            desc['region_start'] = start
            desc['region_end'] = end
            results.append(desc)
        
        return pd.DataFrame(results)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        if self.results is None:
            raise ValueError("No results available. Run calculate_all() first.")
        return self.results
    
    def save_results(self, output_file: str, format: str = "csv"):
        """
        Save results to file.
        
        Parameters
        ----------
        output_file : str
            Output file path
        format : str
            Output format ('csv', 'json', 'excel')
        """
        if self.results is None:
            raise ValueError("No results to save. Run calculate_all() first.")
        
        if format == "csv":
            self.results.to_csv(output_file, index=False)
        elif format == "json":
            self.results.to_json(output_file, orient='records', indent=2)
        elif format == "excel":
            self.results.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {output_file}")
    
    def get_descriptor_categories(self) -> Dict[str, List[str]]:
        """
        Get descriptors organized by category.
        
        Returns
        -------
        dict
            Dictionary with categories as keys and descriptor lists as values
        """
        categories = {
            'Basic Properties': [
                'molecular_weight', 'length', 'isoelectric_point'
            ],
            'Charge': [
                col for col in self.descriptor_mapping.values() 
                if 'Charge' in col or 'Electro' in col
            ],
            'Hydrophobicity': [
                col for col in self.descriptor_mapping.values()
                if 'Hydrophob' in col or 'Lipophil' in col
            ],
            'Structure': [
                col for col in self.descriptor_mapping.values()
                if 'Helix' in col or 'Sheet' in col or 'Turn' in col
            ],
            'Size': [
                col for col in self.descriptor_mapping.values()
                if 'Size' in col or 'Bulk' in col or 'Volume' in col
            ],
            'Composition': [
                col for col in self.descriptor_mapping.values()
                if 'Composition' in col or 'Aromatic' in col
            ]
        }
        
        return categories