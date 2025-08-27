"""
PROPKA analysis module for pKa predictions and pH-dependent properties.

This module runs PROPKA3 to calculate:
- pKa values of titratable residues
- pH-dependent charge distribution
- Folding free energy
- Protein stability at different pH values
"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PropkaAnalyzer:
    """
    Analyzer for pKa predictions and pH-dependent properties using PROPKA3.
    
    PROPKA predicts pKa values of ionizable groups based on:
    - Desolvation effects
    - Hydrogen bonding
    - Charge-charge interactions
    
    Parameters
    ----------
    propka_path : str
        Path to PROPKA3 executable (default: 'propka3')
    pH_range : tuple
        pH range for calculations (default: (0, 14))
        
    References
    ----------
    .. [1] Olsson et al. (2011) "PROPKA3: Consistent Treatment of Internal 
           and Surface Residues in Empirical pKa Predictions" 
           J Chem Theory Comput 7(2):525-37
           
    Examples
    --------
    >>> analyzer = PropkaAnalyzer()
    >>> results = analyzer.run("antibody.pdb")
    >>> print(f"Isoelectric point: {results['pI_folded']:.1f}")
    """
    
    def __init__(self, propka_path: str = 'propka3', pH_range: Tuple[float, float] = (0, 14)):
        self.propka_path = propka_path
        self.pH_range = pH_range
        self._check_propka_installation()
        
        # Standard pKa values
        self.standard_pka = {
            'ASP': 3.80, 'GLU': 4.25, 'HIS': 6.00,
            'CYS': 9.00, 'TYR': 10.00, 'LYS': 10.50,
            'ARG': 12.50, 'N+': 8.00, 'C-': 3.20
        }
    
    def _check_propka_installation(self):
        """Check if PROPKA is installed."""
        try:
            result = subprocess.run(
                [self.propka_path, '--help'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning("PROPKA not found. Some features will be unavailable.")
        except FileNotFoundError:
            logger.warning(f"PROPKA not found at {self.propka_path}")
    
    def run(self, pdb_file: Union[str, Path]) -> Dict:
        """
        Run PROPKA analysis on PDB structure.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        dict
            Dictionary containing:
            - pI_folded: Isoelectric point of folded protein
            - pI_unfolded: Isoelectric point of unfolded protein
            - residue_pka: pKa values for all titratable residues
            - charge_pH7: Net charge at pH 7
            - stability_pH7: Folding free energy at pH 7
            - pH_stability_profile: Stability across pH range
            - titratable_residues: Count of titratable residues
        """
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Run PROPKA
            pka_file = tmpdir / f"{pdb_file.stem}.pka"
            
            try:
                cmd = [self.propka_path, '-q', str(pdb_file), '-o', str(tmpdir / pdb_file.stem)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"PROPKA failed: {result.stderr}")
                    return self._empty_results()
                
                # Parse output
                if not pka_file.exists():
                    logger.error(f"PROPKA output not found: {pka_file}")
                    return self._empty_results()
                
                return self._parse_pka_file(pka_file)
                
            except Exception as e:
                logger.error(f"PROPKA analysis failed: {e}")
                return self._empty_results()
    
    def _parse_pka_file(self, pka_file: Path) -> Dict:
        """Parse PROPKA output file."""
        with open(pka_file, 'r') as f:
            content = f.read()
        
        results = {}
        
        # Extract pI values
        pi_match = re.search(
            r'The pI is\s+([\d.]+)\s+\(folded\)\s+and\s+([\d.]+)\s+\(unfolded\)',
            content
        )
        if pi_match:
            results['pI_folded'] = float(pi_match.group(1))
            results['pI_unfolded'] = float(pi_match.group(2))
        else:
            results['pI_folded'] = np.nan
            results['pI_unfolded'] = np.nan
        
        # Extract residue pKa values
        residue_pka = {}
        desolvation_effects = {}
        
        # Parse SUMMARY section
        summary_match = re.search(
            r'SUMMARY OF THIS PREDICTION.*?\n(.*?)(?:\n-+|\Z)',
            content, re.DOTALL
        )
        
        if summary_match:
            for line in summary_match.group(1).strip().split('\n'):
                match = re.match(r'\s*([A-Z+-]+)\s+(\d+)\s+([A-Z])\s+([\d.]+)', line)
                if match:
                    res_type = match.group(1)
                    res_num = int(match.group(2))
                    chain = match.group(3)
                    pka = float(match.group(4))
                    
                    res_id = f"{res_type}_{res_num}_{chain}"
                    residue_pka[res_id] = pka
                    
                    # Calculate pKa shift from standard
                    if res_type in self.standard_pka:
                        shift = pka - self.standard_pka[res_type]
                        desolvation_effects[res_id] = shift
        
        results['residue_pka'] = residue_pka
        results['pka_shifts'] = desolvation_effects
        results['titratable_residues'] = len(residue_pka)
        
        # Extract charge at different pH values
        charge_data = self._extract_charge_profile(content)
        results.update(charge_data)
        
        # Extract free energy profile
        energy_data = self._extract_energy_profile(content)
        results.update(energy_data)
        
        # Calculate additional metrics
        results.update(self._calculate_stability_metrics(residue_pka, desolvation_effects))
        
        return results
    
    def _extract_charge_profile(self, content: str) -> Dict:
        """Extract charge vs pH profile."""
        charge_match = re.search(
            r'Protein charge of folded and unfolded.*?\n\s*pH\s+unfolded\s+folded\n((?:\s*[\d.]+\s+[\d.-]+\s+[\d.-]+\n)+)',
            content, re.DOTALL
        )
        
        charge_data = {
            'charge_pH7_folded': 0,
            'charge_pH7_unfolded': 0,
            'charge_profile': {}
        }
        
        if charge_match:
            for line in charge_match.group(1).strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 3:
                    pH = float(parts[0])
                    unfolded = float(parts[1])
                    folded = float(parts[2])
                    
                    charge_data['charge_profile'][pH] = {
                        'folded': folded,
                        'unfolded': unfolded
                    }
                    
                    if abs(pH - 7.0) < 0.01:
                        charge_data['charge_pH7_folded'] = folded
                        charge_data['charge_pH7_unfolded'] = unfolded
        
        return charge_data
    
    def _extract_energy_profile(self, content: str) -> Dict:
        """Extract folding free energy vs pH profile."""
        energy_match = re.search(
            r'Free energy of\s+folding.*?\n((?:\s*[\d.]+\s+[\d.-]+\n)+)',
            content, re.DOTALL
        )
        
        energy_data = {
            'stability_pH7': 0,
            'stability_profile': {},
            'optimal_pH': 7.0,
            'max_stability': 0
        }
        
        if energy_match:
            energies = {}
            for line in energy_match.group(1).strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 2:
                    pH = float(parts[0])
                    energy = float(parts[1])
                    energies[pH] = energy
                    
                    if abs(pH - 7.0) < 0.01:
                        energy_data['stability_pH7'] = energy
            
            energy_data['stability_profile'] = energies
            
            if energies:
                # Find optimal pH (minimum energy)
                optimal_pH = min(energies, key=energies.get)
                energy_data['optimal_pH'] = optimal_pH
                energy_data['max_stability'] = energies[optimal_pH]
        
        return energy_data
    
    def _calculate_stability_metrics(
        self,
        residue_pka: Dict,
        pka_shifts: Dict
    ) -> Dict:
        """Calculate additional stability metrics."""
        metrics = {}
        
        # Count unusual pKa values (shifted > 2 units)
        unusual_pka = sum(1 for shift in pka_shifts.values() if abs(shift) > 2)
        metrics['unusual_pka_count'] = unusual_pka
        
        # Calculate mean pKa shift
        if pka_shifts:
            metrics['mean_pka_shift'] = np.mean(list(pka_shifts.values()))
            metrics['max_pka_shift'] = max(pka_shifts.values(), key=abs)
        else:
            metrics['mean_pka_shift'] = 0
            metrics['max_pka_shift'] = 0
        
        # Count buried titratable residues (large pKa shifts often indicate burial)
        buried_titratable = sum(1 for shift in pka_shifts.values() if abs(shift) > 1.5)
        metrics['buried_titratable'] = buried_titratable
        
        return metrics
    
    def calculate_developability_index(
        self,
        sap_score: float,
        charge_pH7: float,
        beta: float = 0.0815
    ) -> float:
        """
        Calculate Developability Index (DI).
        
        DI = SAP_score - β × (net_charge)²
        
        Lower DI values indicate better developability.
        
        Parameters
        ----------
        sap_score : float
            Spatial aggregation propensity score
        charge_pH7 : float
            Net charge at pH 7
        beta : float
            Weighting factor (default: 0.0815)
            
        Returns
        -------
        float
            Developability Index
            
        References
        ----------
        .. [1] Jain et al. (2017) "Biophysical properties of the clinical-stage 
               antibody landscape" PNAS 114(5):944-949
        """
        return sap_score - beta * (charge_pH7 ** 2)
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'pI_folded': np.nan,
            'pI_unfolded': np.nan,
            'residue_pka': {},
            'pka_shifts': {},
            'titratable_residues': 0,
            'charge_pH7_folded': 0,
            'charge_pH7_unfolded': 0,
            'charge_profile': {},
            'stability_pH7': 0,
            'stability_profile': {},
            'optimal_pH': 7.0,
            'max_stability': 0,
            'unusual_pka_count': 0,
            'mean_pka_shift': 0,
            'max_pka_shift': 0,
            'buried_titratable': 0
        }