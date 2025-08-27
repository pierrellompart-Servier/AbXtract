"""
DSSP secondary structure analysis module.

This module runs DSSP to calculate secondary structure assignments,
solvent accessibility, and backbone torsion angles.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import three_to_one

logger = logging.getLogger(__name__)


class DSSPAnalyzer:
    """
    Analyzer for secondary structure using DSSP.
    
    DSSP (Define Secondary Structure of Proteins) calculates:
    - Secondary structure assignments (helix, sheet, coil)
    - Solvent accessibility
    - Backbone phi/psi angles
    - Hydrogen bonding patterns
    
    Parameters
    ----------
    dssp_path : str
        Path to DSSP executable (default: 'dssp')
        
    References
    ----------
    .. [1] Kabsch W, Sander C (1983) "Dictionary of protein secondary 
           structure" Biopolymers 22(12):2577-637
           
    Examples
    --------
    >>> analyzer = DSSPAnalyzer()
    >>> results = analyzer.run("antibody.pdb")
    >>> print(f"Alpha helix content: {results['helix_content']:.1%}")
    """
    
    def __init__(self, dssp_path: str = 'dssp'):
        self.dssp_path = dssp_path
        self._check_dssp_installation()
        
        # DSSP secondary structure codes
        self.ss_codes = {
            'H': 'Alpha helix',
            'B': 'Beta bridge',
            'E': 'Extended strand',
            'G': '3-10 helix',
            'I': 'Pi helix',
            'T': 'Turn',
            'S': 'Bend',
            '-': 'Coil'
        }
        
        # Simplified categories
        self.ss_categories = {
            'helix': {'H', 'G', 'I'},
            'sheet': {'B', 'E'},
            'turn': {'T', 'S'},
            'coil': {'-', ' '}
        }
    
    def _check_dssp_installation(self):
        """Check if DSSP is installed and accessible."""
        try:
            result = subprocess.run(
                [self.dssp_path, '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning("DSSP not found. Some features will be unavailable.")
        except FileNotFoundError:
            logger.warning(f"DSSP not found at {self.dssp_path}")
    
    def run(self, pdb_file: Union[str, Path], chain_type: str = '') -> Dict:
        """
        Run DSSP analysis on PDB structure.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
        chain_type : str
            Type of chains ('antibody', 'nanobody', 'tcr')
            
        Returns
        -------
        dict
            Dictionary containing:
            - secondary_structure: Per-residue SS assignments
            - ss_composition: Fraction of each SS type
            - helix_content: Fraction in helix
            - sheet_content: Fraction in sheet
            - coil_content: Fraction in coil
            - relative_sasa: Per-residue relative SASA
            - phi_psi: Backbone torsion angles
        """
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', str(pdb_file))
        
        try:
            # Run DSSP
            model = structure[0]
            dssp = DSSP(model, str(pdb_file), dssp=self.dssp_path)
            
            # Parse DSSP output
            results = self._parse_dssp_output(dssp, model)
            
            # Calculate composition
            results['ss_composition'] = self._calculate_ss_composition(
                results['secondary_structure']
            )
            
            # Calculate content percentages
            ss_list = list(results['secondary_structure'].values())
            total = len(ss_list)
            
            if total > 0:
                helix_count = sum(1 for ss in ss_list if ss in self.ss_categories['helix'])
                sheet_count = sum(1 for ss in ss_list if ss in self.ss_categories['sheet'])
                coil_count = sum(1 for ss in ss_list if ss in self.ss_categories['coil'])
                
                results['helix_content'] = helix_count / total
                results['sheet_content'] = sheet_count / total  
                results['coil_content'] = coil_count / total
            else:
                results['helix_content'] = 0
                results['sheet_content'] = 0
                results['coil_content'] = 0
            
            # Calculate additional metrics
            results.update(self._calculate_additional_metrics(dssp))
            
            return results
            
        except Exception as e:
            logger.error(f"DSSP analysis failed: {e}")
            return self._empty_results()
    
    def _parse_dssp_output(self, dssp, model) -> Dict:
        """Parse DSSP results into structured format."""
        secondary_structure = {}
        relative_sasa = {}
        phi_psi = {}
        
        for key in dssp.property_dict:
            chain_id = key[0]
            res_id = key[1]
            
            # Get DSSP data
            dssp_data = dssp[key]
            ss = dssp_data[2]
            rsa = dssp_data[3]
            phi = dssp_data[4]
            psi = dssp_data[5]
            
            # Store results
            res_key = f"{chain_id}_{res_id[1]}"
            secondary_structure[res_key] = ss
            relative_sasa[res_key] = rsa
            phi_psi[res_key] = (phi, psi)
        
        return {
            'secondary_structure': secondary_structure,
            'relative_sasa': relative_sasa,
            'phi_psi': phi_psi
        }
    
    def _calculate_ss_composition(self, ss_assignments: Dict) -> Dict:
        """Calculate composition of secondary structure types."""
        composition = {ss: 0 for ss in self.ss_codes.keys()}
        total = len(ss_assignments)
        
        if total == 0:
            return composition
        
        for ss in ss_assignments.values():
            if ss in composition:
                composition[ss] += 1
        
        # Convert to fractions
        for ss in composition:
            composition[ss] = composition[ss] / total
        
        return composition
    
    def _calculate_additional_metrics(self, dssp) -> Dict:
        """Calculate additional structural metrics."""
        metrics = {}
        
        # Extract all values
        all_rsa = []
        all_phi = []
        all_psi = []
        
        for key, data in dssp.property_dict.items():
            all_rsa.append(data[3])
            all_phi.append(data[4])
            all_psi.append(data[5])
        
        # Calculate statistics
        metrics['mean_relative_sasa'] = np.mean(all_rsa) if all_rsa else 0
        metrics['std_relative_sasa'] = np.std(all_rsa) if all_rsa else 0
        metrics['buried_fraction'] = sum(1 for rsa in all_rsa if rsa < 0.075) / len(all_rsa) if all_rsa else 0
        
        # Ramachandran statistics
        favored, allowed, outliers = self._analyze_ramachandran(all_phi, all_psi)
        metrics['ramachandran_favored'] = favored
        metrics['ramachandran_allowed'] = allowed
        metrics['ramachandran_outliers'] = outliers
        
        return metrics
    
    def _analyze_ramachandran(self, phi_angles: list, psi_angles: list) -> Tuple[float, float, float]:
        """
        Analyze Ramachandran plot statistics.
        
        Returns fractions of residues in favored, allowed, and outlier regions.
        """
        if not phi_angles or not psi_angles:
            return 0, 0, 0
        
        favored = 0
        allowed = 0
        outlier = 0
        
        for phi, psi in zip(phi_angles, psi_angles):
            if phi == 360.0 or psi == 360.0:  # Missing values
                continue
            
            # Simplified Ramachandran regions
            # Beta sheet region
            if -180 <= phi <= -45 and 90 <= psi <= 180:
                favored += 1
            # Alpha helix region
            elif -120 <= phi <= -45 and -60 <= psi <= 30:
                favored += 1
            # Left-handed helix region
            elif 45 <= phi <= 120 and -30 <= psi <= 60:
                allowed += 1
            # Beta turn region
            elif -180 <= phi <= -45 and -60 <= psi <= 90:
                allowed += 1
            else:
                outlier += 1
        
        total = favored + allowed + outlier
        if total == 0:
            return 0, 0, 0
        
        return favored/total, allowed/total, outlier/total
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'secondary_structure': {},
            'ss_composition': {ss: 0 for ss in self.ss_codes},
            'helix_content': 0,
            'sheet_content': 0,
            'coil_content': 0,
            'relative_sasa': {},
            'phi_psi': {},
            'mean_relative_sasa': 0,
            'std_relative_sasa': 0,
            'buried_fraction': 0,
            'ramachandran_favored': 0,
            'ramachandran_allowed': 0,
            'ramachandran_outliers': 0
        }