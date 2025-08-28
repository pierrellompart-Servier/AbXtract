"""
Bashour physicochemical descriptors for antibody characterization.

This module calculates various physicochemical properties including:
- Molecular weight and composition
- Charge properties and isoelectric point
- Hydrophobicity and hydrophobic moment
- Extinction coefficients
- Stability indices
- Amino acid content categories

Based on the peptides.py library and adapted for antibody-specific analysis.

References
----------
.. [1] Bashour et al. "Comprehensive physicochemical characterization of antibodies"
.. [2] https://github.com/althonos/peptides.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import logging
import peptides
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import logging
import peptides

class BashourDescriptorCalculator:
    """
    Calculator for Bashour physicochemical descriptors.
    
    These descriptors provide comprehensive characterization of protein
    sequences including charge, hydrophobicity, stability, and composition.
    
    Attributes
    ----------
    results : pd.DataFrame
        DataFrame containing calculated descriptors
        
    Examples
    --------
    >>> calc = BashourDescriptorCalculator()
    >>> results = calc.calculate(["MVLQTQVFISLLLWISGAYG"])
    >>> print(results['Molecular Weight'].values)
    """
    



    def __init__(self):
        """Initialize the calculator."""
        self.results = None
        self.descriptor_info = self._get_descriptor_info()
    
    def _get_descriptor_info(self) -> Dict:
        """Get information about each descriptor."""
        return {
            'Molecular Weight': {
                'description': 'Total molecular weight of the sequence',
                'units': 'Da',
                'category': 'Size'
            },
            'Seq Length': {
                'description': 'Number of amino acids',
                'units': 'residues',
                'category': 'Size'
            },
            'Average Residue Weight': {
                'description': 'Average weight per amino acid',
                'units': 'Da',
                'category': 'Size'
            },
            'pI': {
                'description': 'Isoelectric point',
                'units': 'pH',
                'category': 'Charge'
            },
            'Instability Index': {
                'description': 'Guruprasad instability index (>40 indicates instability)',
                'units': None,
                'category': 'Stability'
            },
            'Aliphatic Index': {
                'description': 'Relative volume of aliphatic side chains',
                'units': None,
                'category': 'Composition'
            },
            'Hydrophobicity': {
                'description': 'Grand average hydropathy (GRAVY)',
                'units': None,
                'category': 'Hydrophobicity'
            },
            'Hydrophobic moment': {
                'description': 'Amphiphilicity measure',
                'units': None,
                'category': 'Hydrophobicity'
            }
        }
    
    def _validate_sequence(self, seq: str) -> str:
        """
        Validate and clean sequence.
        
        Parameters
        ----------
        seq : str
            Input sequence
            
        Returns
        -------
        str
            Cleaned sequence
            
        Raises
        ------
        ValueError
            If sequence is invalid
        """
        if not seq or len(seq) == 0:
            raise ValueError("Empty sequence provided")
        
        # Convert to uppercase and remove whitespace
        seq = seq.upper().strip()
        
        # Remove any non-amino acid characters
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        cleaned_seq = ''.join([aa for aa in seq if aa in valid_aa])
        
        if len(cleaned_seq) == 0:
            raise ValueError("No valid amino acids found in sequence")
        
        if len(cleaned_seq) < 3:
            logger.warning(f"Very short sequence: {cleaned_seq}")
        
        return cleaned_seq
    
    def calculate(self, 
                  sequences: Union[List[str], Dict[str, str], str],
                  dataset_name: str = 'Analysis',
                  calculate_charge_profile: bool = True,
                  pH_range: tuple = (1, 14)) -> pd.DataFrame:
        """
        Calculate Bashour descriptors for sequences.
        
        Parameters
        ----------
        sequences : list, dict, or str
            Sequences to analyze
        dataset_name : str
            Name for the dataset
        calculate_charge_profile : bool
            Whether to calculate charge at all pH values
        pH_range : tuple
            pH range for charge calculation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with calculated descriptors
        """
        # Prepare input data
        df = self._prepare_data(sequences, dataset_name)
        
        logger.info(f"Calculating descriptors for {len(df)} sequences")
        
        # Calculate each descriptor category
        self._calculate_size_descriptors(df)
        self._calculate_charge_descriptors(df, calculate_charge_profile, pH_range)
        self._calculate_photochemical_descriptors(df)
        self._calculate_stability_descriptors(df)
        self._calculate_hydrophobicity_descriptors(df)
        self._calculate_composition_descriptors(df)
        
        self.results = df
        return df
    
    def _prepare_data(self, 
                      sequences: Union[List[str], Dict[str, str], str],
                      dataset_name: str) -> pd.DataFrame:
        """Prepare input data as DataFrame."""
        df = pd.DataFrame()
        
        if isinstance(sequences, str):
            df['SeqID'] = ['Seq_1']
            df['Seq'] = [sequences.upper()]
        elif isinstance(sequences, dict):
            df['SeqID'] = list(sequences.keys())
            df['Seq'] = [s.upper() for s in sequences.values()]
        elif isinstance(sequences, list):
            df['SeqID'] = [f'Seq_{i+1}' for i in range(len(sequences))]
            df['Seq'] = [s.upper() for s in sequences]
        else:
            raise ValueError("sequences must be a string, list, or dictionary")
        
        df['Dataset'] = dataset_name
        return df
    
    def _calculate_size_descriptors(self, df: pd.DataFrame):
        """Calculate size-related descriptors."""
        df['Molecular Weight'] = df['Seq'].apply(self._get_mw_safe)
        df['Seq Length'] = df['Seq'].apply(len)
        df['Average Residue Weight'] = df.apply(
            lambda row: row['Molecular Weight'] / row['Seq Length'] 
            if row['Seq Length'] > 0 and pd.notna(row['Molecular Weight'])
            else np.nan, axis=1
        )
    
    def _calculate_charge_descriptors(self, df: pd.DataFrame,
                                     calculate_profile: bool,
                                     pH_range: tuple):
        """Calculate charge-related descriptors."""
        df['pI'] = df['Seq'].apply(self._get_pI_safe)
        
        if calculate_profile:
            df['Charges (all pH values)'] = df['Seq'].apply(
                lambda x: self._get_all_charges_safe(x, pH_range)
            )
            
            # Expand charge columns for individual pH values
            for pH in range(pH_range[0], pH_range[1] + 1):
                df[f'Charge_pH_{pH}'] = df['Charges (all pH values)'].apply(
                    lambda x: x.get(pH, np.nan) if isinstance(x, dict) else np.nan
                )
    
    def _calculate_photochemical_descriptors(self, df: pd.DataFrame):
        """Calculate photochemical descriptors."""
        df['Molecular Extinction Coefficient'] = df['Seq'].apply(
            self._get_mol_extcoef_safe
        )
        df['Molecular Extinction Coefficient (Cysteine bridges)'] = df['Seq'].apply(
            self._get_mol_extcoef_cysteine_bridges_safe
        )
        df['% Molecular Extinction Coefficient'] = df.apply(
            lambda row: (row['Molecular Extinction Coefficient'] * 10) / row['Molecular Weight'] 
            if pd.notna(row['Molecular Extinction Coefficient']) and pd.notna(row['Molecular Weight']) and row['Molecular Weight'] > 0
            else np.nan, axis=1
        )
        df['% Molecular Extinction Coefficient (Cysteine bridges)'] = df.apply(
            lambda row: (row['Molecular Extinction Coefficient (Cysteine bridges)'] * 10) / row['Molecular Weight'] 
            if pd.notna(row['Molecular Extinction Coefficient (Cysteine bridges)']) and pd.notna(row['Molecular Weight']) and row['Molecular Weight'] > 0
            else np.nan, axis=1
        )
    
    def _calculate_stability_descriptors(self, df: pd.DataFrame):
        """Calculate stability-related descriptors."""
        df['Instability Index'] = df['Seq'].apply(self._get_instaindex_safe)
        df['Aliphatic Index'] = df['Seq'].apply(self._get_aliphindex_safe)
    
    def _calculate_hydrophobicity_descriptors(self, df: pd.DataFrame):
        """Calculate hydrophobicity-related descriptors."""
        df['Hydrophobicity'] = df['Seq'].apply(self._get_hydrophobicity_safe)
        df['Hydrophobic moment'] = df['Seq'].apply(self._get_hmom_safe)
    
    def _calculate_composition_descriptors(self, df: pd.DataFrame):
        """Calculate amino acid composition descriptors."""
        df['Aromatic content'] = df['Seq'].apply(self._get_aromaticity_safe)
        df['Tiny content'] = df['Seq'].apply(self._get_tiny_safe)
        df['Small content'] = df['Seq'].apply(self._get_small_safe)
        df['Aliphatic content'] = df['Seq'].apply(self._get_aliphatic_safe)
        df['Nonpolar content'] = df['Seq'].apply(self._get_nonpolar_safe)
        df['Polar content'] = df['Seq'].apply(self._get_polar_safe)
        df['Basic content'] = df['Seq'].apply(self._get_basic_safe)
        df['Acidic content'] = df['Seq'].apply(self._get_acidic_safe)
    
    # Safe calculation methods with better error handling
    def _get_mw_safe(self, seq: str) -> float:
        """Calculate molecular weight safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            
            # Try with peptides library first
            try:
                pep = peptides.Peptide(cleaned_seq)
                return pep.molecular_weight()
            except Exception as e:
                logger.warning(f"peptides.Peptide failed for MW: {e}")
                
                # Fallback to BioPython
                try:
                    analysis = ProteinAnalysis(cleaned_seq)
                    return analysis.molecular_weight()
                except Exception as e2:
                    logger.warning(f"ProteinAnalysis failed for MW: {e2}")
                    
                    # Manual calculation as last resort
                    return self._calculate_mw_manual(cleaned_seq)
                    
        except Exception as e:
            logger.error(f"All MW calculations failed: {e}")
            return np.nan
    
    def _calculate_mw_manual(self, seq: str) -> float:
        """Manual molecular weight calculation."""
        # Average amino acid molecular weights (approximate)
        aa_weights = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
            'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.17,
            'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
            'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        
        total_weight = 0
        for aa in seq:
            if aa in aa_weights:
                total_weight += aa_weights[aa]
        
        # Subtract water for peptide bonds
        if len(seq) > 1:
            total_weight -= (len(seq) - 1) * 18.015
        
        return total_weight
    
    def _get_pI_safe(self, seq: str) -> float:
        """Calculate isoelectric point safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            
            # Try with peptides library first
            try:
                pep = peptides.Peptide(cleaned_seq)
                return pep.isoelectric_point(pKscale="Lehninger")
            except Exception as e:
                logger.warning(f"peptides.Peptide failed for pI: {e}")
                
                # Fallback to BioPython
                try:
                    analysis = ProteinAnalysis(cleaned_seq)
                    return analysis.isoelectric_point()
                except Exception as e2:
                    logger.warning(f"ProteinAnalysis failed for pI: {e2}")
                    return np.nan
                    
        except Exception as e:
            logger.error(f"All pI calculations failed: {e}")
            return np.nan
    
    def _get_all_charges_safe(self, seq: str, pH_range: tuple) -> Dict[int, float]:
        """Calculate charge at all pH values safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            all_charges = {}
            
            # Try with peptides library first
            try:
                pep = peptides.Peptide(cleaned_seq)
                for pH in range(pH_range[0], pH_range[1] + 1):
                    try:
                        charge = pep.charge(pH=pH, pKscale='Lehninger')
                        all_charges[pH] = charge
                    except:
                        all_charges[pH] = np.nan
                return all_charges
            except Exception as e:
                logger.warning(f"peptides.Peptide failed for charges: {e}")
                
                # Fallback to BioPython
                try:
                    analysis = ProteinAnalysis(cleaned_seq)
                    for pH in range(pH_range[0], pH_range[1] + 1):
                        try:
                            charge = analysis.charge_at_pH(pH)
                            all_charges[pH] = charge
                        except:
                            all_charges[pH] = np.nan
                    return all_charges
                except Exception as e2:
                    logger.warning(f"ProteinAnalysis failed for charges: {e2}")
                    return {pH: np.nan for pH in range(pH_range[0], pH_range[1] + 1)}
                    
        except Exception as e:
            logger.error(f"All charge calculations failed: {e}")
            return {pH: np.nan for pH in range(pH_range[0], pH_range[1] + 1)}
    
    def _get_mol_extcoef_safe(self, seq: str) -> float:
        """Calculate molecular extinction coefficient safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            
            # Count aromatic residues manually (most reliable)
            n_tyr = cleaned_seq.count('Y')
            n_trp = cleaned_seq.count('W')
            n_cys = cleaned_seq.count('C')
            
            # Calculate with reduced cysteines
            if n_cys % 2 == 0:
                return (n_tyr * 1490) + (n_trp * 5500) + ((n_cys // 2) * 125)
            else:
                return (n_tyr * 1490) + (n_trp * 5500) + (((n_cys - 1) // 2) * 125)
                
        except Exception as e:
            logger.error(f"Extinction coefficient calculation failed: {e}")
            return np.nan
    
    def _get_mol_extcoef_cysteine_bridges_safe(self, seq: str) -> float:
        """Calculate extinction coefficient from cysteine bridges safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            n_cys = cleaned_seq.count('C')
            
            if n_cys % 2 == 0:
                return (n_cys // 2) * 125
            else:
                return ((n_cys - 1) // 2) * 125
                
        except Exception as e:
            logger.error(f"Cysteine extinction coefficient calculation failed: {e}")
            return np.nan
    
    def _get_instaindex_safe(self, seq: str) -> float:
        """Calculate instability index safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            
            if len(cleaned_seq) < 3:
                return np.nan
            
            # Try with peptides library first
            try:
                pep = peptides.Peptide(cleaned_seq)
                return pep.instability_index()
            except Exception as e:
                logger.warning(f"peptides.Peptide failed for instability: {e}")
                
                # Fallback to BioPython
                try:
                    analysis = ProteinAnalysis(cleaned_seq)
                    return analysis.instability_index()
                except Exception as e2:
                    logger.warning(f"ProteinAnalysis failed for instability: {e2}")
                    return np.nan
                    
        except Exception as e:
            logger.error(f"Instability index calculation failed: {e}")
            return np.nan
    
    def _get_aliphindex_safe(self, seq: str) -> float:
        """Calculate aliphatic index safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            
            # Try with peptides library first
            try:
                pep = peptides.Peptide(cleaned_seq)
                return pep.aliphatic_index()
            except Exception as e:
                logger.warning(f"peptides.Peptide failed for aliphatic index: {e}")
                
                # Fallback to BioPython
                try:
                    analysis = ProteinAnalysis(cleaned_seq)
                    return analysis.aliphatic_index()
                except Exception as e2:
                    logger.warning(f"ProteinAnalysis failed for aliphatic index: {e2}")
                    return np.nan
                    
        except Exception as e:
            logger.error(f"Aliphatic index calculation failed: {e}")
            return np.nan
    
    def _get_hydrophobicity_safe(self, seq: str) -> float:
        """Calculate hydrophobicity (GRAVY) safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            
            # Try with peptides library first (using GRAVY scale)
            try:
                pep = peptides.Peptide(cleaned_seq)
                return pep.hydrophobicity(scale='KyteDoolittle')  # This is GRAVY
            except Exception as e:
                logger.warning(f"peptides.Peptide failed for hydrophobicity: {e}")
                
                # Fallback to BioPython
                try:
                    analysis = ProteinAnalysis(cleaned_seq)
                    return analysis.gravy()
                except Exception as e2:
                    logger.warning(f"ProteinAnalysis failed for hydrophobicity: {e2}")
                    return np.nan
                    
        except Exception as e:
            logger.error(f"Hydrophobicity calculation failed: {e}")
            return np.nan
    
    def _get_hmom_safe(self, seq: str) -> float:
        """Calculate hydrophobic moment safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            
            try:
                pep = peptides.Peptide(cleaned_seq)
                return pep.hydrophobic_moment(angle=160, window=10)
            except Exception as e:
                logger.warning(f"Hydrophobic moment calculation failed: {e}")
                return np.nan
                
        except Exception as e:
            logger.error(f"Hydrophobic moment calculation failed: {e}")
            return np.nan
    
    def _get_aromaticity_safe(self, seq: str) -> float:
        """Calculate aromatic content safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            if len(cleaned_seq) == 0:
                return 0
            return ((cleaned_seq.count('F') + cleaned_seq.count('H') + 
                    cleaned_seq.count('W') + cleaned_seq.count('Y')) / len(cleaned_seq)) * 100
        except Exception as e:
            logger.error(f"Aromaticity calculation failed: {e}")
            return np.nan
    
    def _get_tiny_safe(self, seq: str) -> float:
        """Calculate tiny amino acid content safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            if len(cleaned_seq) == 0:
                return 0
            return ((cleaned_seq.count('A') + cleaned_seq.count('C') + cleaned_seq.count('G') +
                    cleaned_seq.count('S') + cleaned_seq.count('T')) / len(cleaned_seq)) * 100
        except Exception as e:
            logger.error(f"Tiny content calculation failed: {e}")
            return np.nan
    
    def _get_small_safe(self, seq: str) -> float:
        """Calculate small amino acid content safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            if len(cleaned_seq) == 0:
                return 0
            return ((cleaned_seq.count('A') + cleaned_seq.count('C') + cleaned_seq.count('D') +
                    cleaned_seq.count('G') + cleaned_seq.count('N') + cleaned_seq.count('P') +
                    cleaned_seq.count('S') + cleaned_seq.count('T') + cleaned_seq.count('V')) / len(cleaned_seq)) * 100
        except Exception as e:
            logger.error(f"Small content calculation failed: {e}")
            return np.nan
    
    def _get_aliphatic_safe(self, seq: str) -> float:
        """Calculate aliphatic content safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            if len(cleaned_seq) == 0:
                return 0
            return ((cleaned_seq.count('A') + cleaned_seq.count('I') + 
                    cleaned_seq.count('L') + cleaned_seq.count('V')) / len(cleaned_seq)) * 100
        except Exception as e:
            logger.error(f"Aliphatic content calculation failed: {e}")
            return np.nan
    
    def _get_nonpolar_safe(self, seq: str) -> float:
        """Calculate nonpolar content safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            if len(cleaned_seq) == 0:
                return 0
            return ((cleaned_seq.count('A') + cleaned_seq.count('C') + cleaned_seq.count('F') +
                    cleaned_seq.count('G') + cleaned_seq.count('I') + cleaned_seq.count('L') +
                    cleaned_seq.count('M') + cleaned_seq.count('P') + cleaned_seq.count('V') +
                    cleaned_seq.count('W') + cleaned_seq.count('Y')) / len(cleaned_seq)) * 100
        except Exception as e:
            logger.error(f"Nonpolar content calculation failed: {e}")
            return np.nan
    
    def _get_polar_safe(self, seq: str) -> float:
        """Calculate polar content safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            if len(cleaned_seq) == 0:
                return 0
            return ((cleaned_seq.count('D') + cleaned_seq.count('E') + cleaned_seq.count('H') +
                    cleaned_seq.count('K') + cleaned_seq.count('N') + cleaned_seq.count('Q') +
                    cleaned_seq.count('R') + cleaned_seq.count('S') + cleaned_seq.count('T')) / len(cleaned_seq)) * 100
        except Exception as e:
            logger.error(f"Polar content calculation failed: {e}")
            return np.nan
    
    def _get_basic_safe(self, seq: str) -> float:
        """Calculate basic content safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            if len(cleaned_seq) == 0:
                return 0
            return ((cleaned_seq.count('H') + cleaned_seq.count('K') + 
                    cleaned_seq.count('R')) / len(cleaned_seq)) * 100
        except Exception as e:
            logger.error(f"Basic content calculation failed: {e}")
            return np.nan
    
    def _get_acidic_safe(self, seq: str) -> float:
        """Calculate acidic content safely."""
        try:
            cleaned_seq = self._validate_sequence(seq)
            if len(cleaned_seq) == 0:
                return 0
            return ((cleaned_seq.count('D') + cleaned_seq.count('E')) / len(cleaned_seq)) * 100
        except Exception as e:
            logger.error(f"Acidic content calculation failed: {e}")
            return np.nan
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        if self.results is None:
            raise ValueError("No results available. Run calculate() first.")
        return self.results
    
    def save_results(self, output_file: str, format: str = "csv"):
        """Save results to file."""
        if self.results is None:
            raise ValueError("No results to save. Run calculate() first.")
        
        if format == "csv":
            self.results.to_csv(output_file, index=False)
        elif format == "json":
            self.results.to_json(output_file, orient='records', indent=2)
        elif format == "excel":
            self.results.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {output_file}")
        
        
        
class ExtendedBashourCalculator:
    """
    Extended Bashour descriptor calculator with all original metrics.
    """
    
    def __init__(self):
        """Initialize extended Bashour calculator."""
        pass
    
    def calculate_all_charges(self, sequence: str) -> Dict[int, float]:
        """
        Calculate charge at all pH values from 1 to 14.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence
            
        Returns
        -------
        dict
            Dictionary mapping pH to charge
        """
        try:
            analysed_seq = ProteinAnalysis(sequence)
            charges = {}
            
            for pH in range(1, 15):
                try:
                    charge = analysed_seq.charge_at_pH(pH)
                    charges[pH] = charge
                except Exception:
                    charges[pH] = 0.0
            
            return charges
        except Exception as e:
            logger.warning(f"Failed to calculate charges: {e}")
            return {pH: 0.0 for pH in range(1, 15)}
    
    def calculate_hydrophobic_moment(self, sequence: str, 
                                    window: int = 11, 
                                    angle: int = 100) -> float:
        """
        Calculate hydrophobic moment.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence
        window : int
            Window size for calculation
        angle : int
            Angle for moment calculation (100 for alpha helix, 160 for beta sheet)
            
        Returns
        -------
        float
            Maximum hydrophobic moment
        """
        # Kyte-Doolittle hydrophobicity scale
        hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        max_moment = 0.0
        
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i+window]
            
            # Calculate moment for this window
            sum_sin = 0.0
            sum_cos = 0.0
            
            for j, aa in enumerate(window_seq):
                if aa in hydrophobicity:
                    h = hydrophobicity[aa]
                    angle_rad = np.deg2rad(j * angle)
                    sum_sin += h * np.sin(angle_rad)
                    sum_cos += h * np.cos(angle_rad)
            
            moment = np.sqrt(sum_sin**2 + sum_cos**2) / window
            max_moment = max(max_moment, moment)
        
        return max_moment
    
    def calculate_percent_extinction_coefficients(self, sequence: str) -> Dict[str, float]:
        """
        Calculate percent molecular extinction coefficients.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence
            
        Returns
        -------
        dict
            Dictionary with percent extinction coefficients
        """
        try:
            analysed_seq = ProteinAnalysis(sequence)
            
            # Get molecular weight
            mw = analysed_seq.molecular_weight()
            
            # Get extinction coefficients
            ext_reduced, ext_oxidized = analysed_seq.molar_extinction_coefficient()
            
            # Calculate percent extinction coefficients (per mg/ml)
            results = {
                'percent_mol_extcoef': (ext_reduced / mw) * 10 if mw > 0 else 0,
                'percent_mol_extcoef_cys_bridges': (ext_oxidized / mw) * 10 if mw > 0 else 0
            }
            
            return results
        except Exception as e:
            logger.warning(f"Failed to calculate extinction coefficients: {e}")
            return {'percent_mol_extcoef': 0, 'percent_mol_extcoef_cys_bridges': 0}
    
    def calculate_extended_descriptors(self, sequence: str) -> Dict[str, float]:
        """
        Calculate all extended Bashour descriptors.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence
            
        Returns
        -------
        dict
            All extended descriptors
        """
        results = {}
        
        try:
            analysed_seq = ProteinAnalysis(sequence)
            
            # Basic properties
            results['molecular_weight'] = analysed_seq.molecular_weight()
            results['seq_length'] = len(sequence)
            results['average_residue_weight'] = results['molecular_weight'] / len(sequence) if len(sequence) > 0 else 0
            
            # Charge properties
            charges = self.calculate_all_charges(sequence)
            results['charges_all_pH'] = charges
            
            # Individual pH charges
            for pH in range(1, 15):
                results[f'charge_pH_{pH}'] = charges.get(pH, 0)
            
            # pI
            results['pI'] = analysed_seq.isoelectric_point()
            
            # Extinction coefficients
            ext_reduced, ext_oxidized = analysed_seq.molar_extinction_coefficient()
            results['mol_extinction_coef'] = ext_reduced
            results['mol_extinction_coef_cys_bridges'] = ext_oxidized
            
            # Percent extinction coefficients
            percent_ext = self.calculate_percent_extinction_coefficients(sequence)
            results.update(percent_ext)
            
            # Stability indices
            results['instability_index'] = analysed_seq.instability_index()
            results['aliphatic_index'] = analysed_seq.aliphatic_index()
            results['gravy'] = analysed_seq.gravy()  # Hydrophobicity
            
            # Hydrophobic moment
            results['hydrophobic_moment'] = self.calculate_hydrophobic_moment(sequence)
            
            # Amino acid percentages
            aa_percent = analysed_seq.get_amino_acids_percent()
            
            # Content calculations
            results['aromaticity'] = analysed_seq.aromaticity()
            
            # Group contents
            tiny = ['A', 'C', 'G', 'S', 'T']
            small = ['A', 'C', 'D', 'G', 'N', 'P', 'S', 'T', 'V']
            aliphatic = ['A', 'I', 'L', 'V']
            aromatic = ['F', 'W', 'Y']
            nonpolar = ['A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W', 'Y']
            polar = ['D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Z']
            charged = ['D', 'E', 'H', 'K', 'R']
            basic = ['H', 'K', 'R']
            acidic = ['D', 'E']
            
            results['tiny_content'] = sum(aa_percent.get(aa, 0) for aa in tiny)
            results['small_content'] = sum(aa_percent.get(aa, 0) for aa in small)
            results['aliphatic_content'] = sum(aa_percent.get(aa, 0) for aa in aliphatic)
            results['aromatic_content'] = sum(aa_percent.get(aa, 0) for aa in aromatic)
            results['nonpolar_content'] = sum(aa_percent.get(aa, 0) for aa in nonpolar)
            results['polar_content'] = sum(aa_percent.get(aa, 0) for aa in polar)
            results['charged_content'] = sum(aa_percent.get(aa, 0) for aa in charged)
            results['basic_content'] = sum(aa_percent.get(aa, 0) for aa in basic)
            results['acidic_content'] = sum(aa_percent.get(aa, 0) for aa in acidic)
            
            # Secondary structure propensities
            helix_formers = ['A', 'E', 'L', 'M']
            sheet_formers = ['V', 'I', 'Y', 'W', 'F']
            turn_formers = ['G', 'N', 'P', 'S', 'D']
            
            results['helix_propensity'] = sum(aa_percent.get(aa, 0) for aa in helix_formers)
            results['sheet_propensity'] = sum(aa_percent.get(aa, 0) for aa in sheet_formers)
            results['turn_propensity'] = sum(aa_percent.get(aa, 0) for aa in turn_formers)
            
        except Exception as e:
            logger.error(f"Failed to calculate extended descriptors: {e}")
        
        return results
    
    def calculate_for_dataframe(self, df: pd.DataFrame, 
                               sequence_column: str = 'Seq') -> pd.DataFrame:
        """
        Calculate extended descriptors for sequences in DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sequences
        sequence_column : str
            Name of column containing sequences
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added descriptor columns
        """
        result_df = df.copy()
        
        # Calculate for each sequence
        for idx, row in df.iterrows():
            seq = row[sequence_column]
            descriptors = self.calculate_extended_descriptors(seq)
            
            # Add to dataframe
            for key, value in descriptors.items():
                if key != 'charges_all_pH':  # Skip the full dict
                    result_df.at[idx, key] = value
        
        return result_df