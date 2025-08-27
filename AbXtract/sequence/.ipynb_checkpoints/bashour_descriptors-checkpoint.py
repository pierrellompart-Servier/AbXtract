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

import peptides
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)


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
            },
            'Aromatic content': {
                'description': 'Percentage of aromatic amino acids (F,H,W,Y)',
                'units': '%',
                'category': 'Composition'
            },
            'Tiny content': {
                'description': 'Percentage of tiny amino acids (A,C,G,S,T)',
                'units': '%',
                'category': 'Composition'
            },
            'Small content': {
                'description': 'Percentage of small amino acids (A,C,D,G,N,P,S,T,V)',
                'units': '%',
                'category': 'Composition'
            },
            'Aliphatic content': {
                'description': 'Percentage of aliphatic amino acids (A,I,L,V)',
                'units': '%',
                'category': 'Composition'
            },
            'Nonpolar content': {
                'description': 'Percentage of nonpolar amino acids',
                'units': '%',
                'category': 'Composition'
            },
            'Polar content': {
                'description': 'Percentage of polar amino acids',
                'units': '%',
                'category': 'Composition'
            },
            'Basic content': {
                'description': 'Percentage of basic amino acids (H,K,R)',
                'units': '%',
                'category': 'Composition'
            },
            'Acidic content': {
                'description': 'Percentage of acidic amino acids (D,E)',
                'units': '%',
                'category': 'Composition'
            }
        }
    
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
            Sequences to analyze. Can be:
            - List of sequences
            - Dictionary of {name: sequence}
            - Single sequence string
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
        df['Molecular Weight'] = df['Seq'].apply(self._get_mw)
        df['Seq Length'] = df['Seq'].apply(len)
        df['Average Residue Weight'] = df['Seq'].apply(self._get_avresweight)
    
    def _calculate_charge_descriptors(self, df: pd.DataFrame,
                                     calculate_profile: bool,
                                     pH_range: tuple):
        """Calculate charge-related descriptors."""
        df['pI'] = df['Seq'].apply(self._get_pI)
        
        if calculate_profile:
            df['Charges (all pH values)'] = df['Seq'].apply(
                lambda x: self._get_all_charges(x, pH_range)
            )
            
            # Expand charge columns for individual pH values
            for pH in range(pH_range[0], pH_range[1] + 1):
                df[f'Charge_pH_{pH}'] = df['Charges (all pH values)'].apply(
                    lambda x: x.get(pH, np.nan)
                )
    
    def _calculate_photochemical_descriptors(self, df: pd.DataFrame):
        """Calculate photochemical descriptors."""
        df['Molecular Extinction Coefficient'] = df['Seq'].apply(
            self._get_mol_extcoef
        )
        df['Molecular Extinction Coefficient (Cysteine bridges)'] = df['Seq'].apply(
            self._get_mol_extcoef_cysteine_bridges
        )
        df['% Molecular Extinction Coefficient'] = df['Seq'].apply(
            self._get_percent_mol_extcoef
        )
        df['% Molecular Extinction Coefficient (Cysteine bridges)'] = df['Seq'].apply(
            self._get_percent_mol_extcoef_cysteine_bridges
        )
    
    def _calculate_stability_descriptors(self, df: pd.DataFrame):
        """Calculate stability-related descriptors."""
        df['Instability Index'] = df['Seq'].apply(self._get_instaindex)
        df['Aliphatic Index'] = df['Seq'].apply(self._get_aliphindex)
    
    def _calculate_hydrophobicity_descriptors(self, df: pd.DataFrame):
        """Calculate hydrophobicity-related descriptors."""
        df['Hydrophobicity'] = df['Seq'].apply(self._get_hydrophobicity)
        df['Hydrophobic moment'] = df['Seq'].apply(self._get_hmom)
    
    def _calculate_composition_descriptors(self, df: pd.DataFrame):
        """Calculate amino acid composition descriptors."""
        df['Aromatic content'] = df['Seq'].apply(self._get_aromaticity)
        df['Tiny content'] = df['Seq'].apply(self._get_tiny)
        df['Small content'] = df['Seq'].apply(self._get_small)
        df['Aliphatic content'] = df['Seq'].apply(self._get_aliphatic)
        df['Nonpolar content'] = df['Seq'].apply(self._get_nonpolar)
        df['Polar content'] = df['Seq'].apply(self._get_polar)
        df['Basic content'] = df['Seq'].apply(self._get_basic)
        df['Acidic content'] = df['Seq'].apply(self._get_acidic)
    
    # Individual descriptor calculation methods
    def _get_mw(self, seq: str) -> float:
        """Calculate molecular weight."""
        pep = peptides.Peptide(seq)
        return pep.molecular_weight()
    
    def _get_avresweight(self, seq: str) -> float:
        """Calculate average residue weight."""
        pep = peptides.Peptide(seq)
        return pep.molecular_weight() / len(seq) if len(seq) > 0 else 0
    
    def _get_all_charges(self, seq: str, pH_range: tuple) -> Dict[int, float]:
        """Calculate charge at all pH values."""
        all_charges = {}
        pep = peptides.Peptide(seq)
        
        for pH in range(pH_range[0], pH_range[1] + 1):
            try:
                charge = pep.charge(pH=pH, pKscale='Lehninger')
                all_charges[pH] = charge
            except:
                all_charges[pH] = np.nan
        
        return all_charges
    
    def _get_pI(self, seq: str) -> float:
        """Calculate isoelectric point."""
        try:
            pep = peptides.Peptide(seq)
            return pep.isoelectric_point(pKscale="Lehninger")
        except:
            return np.nan
    
    def _get_mol_extcoef(self, seq: str) -> float:
        """Calculate molecular extinction coefficient."""
        # Count aromatic residues
        n_tyr = seq.count('Y')
        n_trp = seq.count('W')
        n_cys = seq.count('C')
        
        # Calculate with reduced cysteines
        if n_cys % 2 == 0:
            return (n_tyr * 1490) + (n_trp * 5500) + ((n_cys // 2) * 125)
        else:
            return (n_tyr * 1490) + (n_trp * 5500) + (((n_cys - 1) // 2) * 125)
    
    def _get_mol_extcoef_cysteine_bridges(self, seq: str) -> float:
        """Calculate extinction coefficient from cysteine bridges only."""
        n_cys = seq.count('C')
        if n_cys % 2 == 0:
            return (n_cys // 2) * 125
        else:
            return ((n_cys - 1) // 2) * 125
    
    def _get_percent_mol_extcoef(self, seq: str) -> float:
        """Calculate percent molecular extinction coefficient."""
        mol_extcoef = self._get_mol_extcoef(seq)
        mw = self._get_mw(seq)
        return (mol_extcoef * 10) / mw if mw > 0 else 0
    
    def _get_percent_mol_extcoef_cysteine_bridges(self, seq: str) -> float:
        """Calculate percent extinction coefficient from cysteine bridges."""
        mol_extcoef_cys = self._get_mol_extcoef_cysteine_bridges(seq)
        mw = self._get_mw(seq)
        return (mol_extcoef_cys * 10) / mw if mw > 0 else 0
    
    def _get_instaindex(self, seq: str) -> float:
        """Calculate instability index."""
        if len(seq) < 3:
            return np.nan
        try:
            pep = peptides.Peptide(seq)
            return pep.instability_index()
        except:
            return np.nan
    
    def _get_aliphindex(self, seq: str) -> float:
        """Calculate aliphatic index."""
        try:
            pep = peptides.Peptide(seq)
            return pep.aliphatic_index()
        except:
            return np.nan
    
    def _get_hydrophobicity(self, seq: str) -> float:
        """Calculate hydrophobicity (GRAVY)."""
        try:
            pep = peptides.Peptide(seq)
            return pep.hydrophobicity(scale='Eisenberg')
        except:
            return np.nan
    
    def _get_hmom(self, seq: str) -> float:
        """Calculate hydrophobic moment."""
        try:
            pep = peptides.Peptide(seq)
            return pep.hydrophobic_moment(angle=160, window=10)
        except:
            return np.nan
    
    def _get_aromaticity(self, seq: str) -> float:
        """Calculate aromatic content (F,H,W,Y)."""
        if len(seq) == 0:
            return 0
        return ((seq.count('F') + seq.count('H') + 
                seq.count('W') + seq.count('Y')) / len(seq)) * 100
    
    def _get_tiny(self, seq: str) -> float:
        """Calculate tiny amino acid content (A,C,G,S,T)."""
        if len(seq) == 0:
            return 0
        return ((seq.count('A') + seq.count('C') + seq.count('G') +
                seq.count('S') + seq.count('T')) / len(seq)) * 100
    
    def _get_small(self, seq: str) -> float:
        """Calculate small amino acid content (A,C,D,G,N,P,S,T,V)."""
        if len(seq) == 0:
            return 0
        return ((seq.count('A') + seq.count('C') + seq.count('D') +
                seq.count('G') + seq.count('N') + seq.count('P') +
                seq.count('S') + seq.count('T') + seq.count('V')) / len(seq)) * 100
    
    def _get_aliphatic(self, seq: str) -> float:
        """Calculate aliphatic content (A,I,L,V)."""
        if len(seq) == 0:
            return 0
        return ((seq.count('A') + seq.count('I') + 
                seq.count('L') + seq.count('V')) / len(seq)) * 100
    
    def _get_nonpolar(self, seq: str) -> float:
        """Calculate nonpolar content (A,C,F,G,I,L,M,P,V,W,Y)."""
        if len(seq) == 0:
            return 0
        return ((seq.count('A') + seq.count('C') + seq.count('F') +
                seq.count('G') + seq.count('I') + seq.count('L') +
                seq.count('M') + seq.count('P') + seq.count('V') +
                seq.count('W') + seq.count('Y')) / len(seq)) * 100
    
    def _get_polar(self, seq: str) -> float:
        """Calculate polar content (D,E,H,K,N,Q,R,S,T)."""
        if len(seq) == 0:
            return 0
        return ((seq.count('D') + seq.count('E') + seq.count('H') +
                seq.count('K') + seq.count('N') + seq.count('Q') +
                seq.count('R') + seq.count('S') + seq.count('T')) / len(seq)) * 100
    
    def _get_basic(self, seq: str) -> float:
        """Calculate basic content (H,K,R)."""
        if len(seq) == 0:
            return 0
        return ((seq.count('H') + seq.count('K') + 
                seq.count('R')) / len(seq)) * 100
    
    def _get_acidic(self, seq: str) -> float:
        """Calculate acidic content (D,E)."""
        if len(seq) == 0:
            return 0
        return ((seq.count('D') + seq.count('E')) / len(seq)) * 100
    
    def get_descriptor_info(self, descriptor_name: str) -> Dict:
        """
        Get information about a specific descriptor.
        
        Parameters
        ----------
        descriptor_name : str
            Name of the descriptor
            
        Returns
        -------
        dict
            Information about the descriptor
        """
        return self.descriptor_info.get(descriptor_name, {})
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        if self.results is None:
            raise ValueError("No results available. Run calculate() first.")
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