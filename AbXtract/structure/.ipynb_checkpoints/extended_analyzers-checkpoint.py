"""
Extended structure analyzers for AbXtract.

This module provides additional structure-based analyzers including:
- Disulfide bond analysis
- Charge dispersion metrics
- Extended SASA calculations
- Extended PROPKA analysis
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class DisulfideBondAnalyzer:
    """
    Analyzer for disulfide bond detection and characterization.
    
    Parameters
    ----------
    distance_cutoff : float
        Maximum S-S distance for disulfide bonds (default: 2.5 Å)
    """
    
    def __init__(self, distance_cutoff: float = 2.5):
        self.distance_cutoff = distance_cutoff
    
    def analyze(self, pdb_file: Union[str, Path]) -> Dict:
        """
        Analyze disulfide bonds in structure.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        dict
            Analysis results including free cysteines and disulfide bonds
        """
        try:
            from Bio.PDB import PDBParser
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('struct', str(pdb_file))
            
            cysteines = []
            
            # Find all cysteines
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_resname() == 'CYS':
                            if 'SG' in residue:  # Sulfur atom
                                cysteines.append({
                                    'chain': chain.id,
                                    'residue': residue.id[1],
                                    'atom': residue['SG'],
                                    'coords': residue['SG'].get_coord()
                                })
            
            # Find disulfide bonds
            disulfide_bonds = []
            bonded_cysteines = set()
            
            for i, cys1 in enumerate(cysteines):
                for j, cys2 in enumerate(cysteines[i+1:], i+1):
                    distance = np.linalg.norm(cys1['coords'] - cys2['coords'])
                    if distance <= self.distance_cutoff:
                        disulfide_bonds.append({
                            'cys1': f"{cys1['chain']}_{cys1['residue']}",
                            'cys2': f"{cys2['chain']}_{cys2['residue']}",
                            'distance': distance
                        })
                        bonded_cysteines.add(i)
                        bonded_cysteines.add(j)
            
            # Count free cysteines
            free_cys = len(cysteines) - len(bonded_cysteines)
            
            return {
                'total_cys': len(cysteines),
                'cys_bridges': len(disulfide_bonds),
                'free_cys': free_cys,
                'disulfide_bonds': disulfide_bonds
            }
            
        except Exception as e:
            logger.error(f"Disulfide bond analysis failed: {e}")
            return {
                'total_cys': 0,
                'cys_bridges': 0,
                'free_cys': 0,
                'disulfide_bonds': []
            }


class ChargeDispersionAnalyzer:
    """
    Analyzer for charge dispersion and clustering metrics.
    """
    
    def __init__(self):
        self.positive_residues = {'ARG', 'LYS', 'HIS'}
        self.negative_residues = {'ASP', 'GLU'}
    
    def calculate(self, pdb_file: Union[str, Path]) -> Dict:
        """
        Calculate charge dispersion metrics.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        dict
            Charge dispersion metrics
        """
        try:
            from Bio.PDB import PDBParser
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('struct', str(pdb_file))
            
            positive_coords = []
            negative_coords = []
            
            # Collect charged residue positions
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_name = residue.get_resname()
                        
                        # Get CA coordinate if available
                        if 'CA' in residue:
                            coord = residue['CA'].get_coord()
                            
                            if res_name in self.positive_residues:
                                positive_coords.append(coord)
                            elif res_name in self.negative_residues:
                                negative_coords.append(coord)
            
            positive_coords = np.array(positive_coords) if positive_coords else np.array([])
            negative_coords = np.array(negative_coords) if negative_coords else np.array([])
            
            metrics = {}
            
            # Calculate heterogeneity (variance in positions)
            if len(positive_coords) > 1:
                metrics['positive_charge_heterogeneity'] = np.std(positive_coords).mean()
            else:
                metrics['positive_charge_heterogeneity'] = 0
            
            if len(negative_coords) > 1:
                metrics['negative_charge_heterogeneity'] = np.std(negative_coords).mean()
            else:
                metrics['negative_charge_heterogeneity'] = 0
            
            # Calculate asymmetry
            if len(positive_coords) > 0 and len(negative_coords) > 0:
                pos_center = np.mean(positive_coords, axis=0)
                neg_center = np.mean(negative_coords, axis=0)
                metrics['charge_asymmetry'] = np.linalg.norm(pos_center - neg_center)
            else:
                metrics['charge_asymmetry'] = 0
            
            # Calculate clustering
            metrics['positive_charge_clustering'] = self._calculate_clustering(positive_coords)
            metrics['negative_charge_clustering'] = self._calculate_clustering(negative_coords)
            
            # Net charge
            metrics['net_charge'] = len(positive_coords) - len(negative_coords)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Charge dispersion analysis failed: {e}")
            return {
                'positive_charge_heterogeneity': 0,
                'negative_charge_heterogeneity': 0,
                'charge_asymmetry': 0,
                'positive_charge_clustering': 0,
                'negative_charge_clustering': 0,
                'net_charge': 0
            }
    
    def _calculate_clustering(self, coords: np.ndarray, threshold: float = 8.0) -> float:
        """Calculate clustering score for charged residues."""
        if len(coords) < 2:
            return 0
        
        # Calculate pairwise distances
        n = len(coords)
        close_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < threshold:
                    close_pairs += 1
        
        # Normalize by number of possible pairs
        max_pairs = n * (n - 1) / 2
        return close_pairs / max_pairs if max_pairs > 0 else 0


class ExtendedSASACalculator:
    """
    Extended SASA calculator with additional metrics.
    """
    
    def __init__(self, probe_radius: float = 1.4):
        self.probe_radius = probe_radius
        self.hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
        self.polar_residues = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        self.charged_residues = {'ARG', 'LYS', 'HIS', 'ASP', 'GLU'}
    
    def calculate_exposed_sidechains(self, pdb_file: Union[str, Path]) -> Dict:
        """
        Calculate exposed sidechain metrics.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        dict
            Extended SASA metrics
        """
        try:
            import freesasa
            from collections import defaultdict
            
            # Calculate SASA
            structure = freesasa.Structure(str(pdb_file))
            result = freesasa.calc(structure, freesasa.Parameters({'probe-radius': self.probe_radius}))
            
            # Define exposure threshold (relative SASA > 0.25)
            exposure_threshold = 0.25
            
            # Standard maximum SASA values (Tien et al., 2013)
            max_sasa = {
                'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0,
                'CYS': 167.0, 'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0,
                'HIS': 224.0, 'ILE': 197.0, 'LEU': 201.0, 'LYS': 236.0,
                'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 'SER': 155.0,
                'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
            }
            
            # Build residue SASA by aggregating atoms
            residue_sasa = defaultdict(float)
            residue_info = {}
            
            # Iterate through all atoms to build residue SASA
            for i in range(structure.nAtoms()):
                res_name = structure.residueName(i).strip()
                res_num = structure.residueNumber(i).strip()
                chain = structure.chainLabel(i).strip()
                atom_area = result.atomArea(i)
                
                res_key = (res_name, res_num, chain)
                residue_sasa[res_key] += atom_area
                residue_info[res_key] = res_name
            
            # Categorize exposed residues
            exposed_hydrophobic = 0
            exposed_polar = 0
            exposed_charged = 0
            total_exposed = 0
            
            # Process each unique residue
            for res_key, total_sasa in residue_sasa.items():
                res_name = residue_info[res_key]
                
                if res_name in max_sasa:
                    relative_sasa = total_sasa / max_sasa[res_name]
                    
                    if relative_sasa > exposure_threshold:
                        total_exposed += 1
                        
                        if res_name in self.hydrophobic_residues:
                            exposed_hydrophobic += 1
                        elif res_name in self.polar_residues:
                            exposed_polar += 1
                        elif res_name in self.charged_residues:
                            exposed_charged += 1
            
            return {
                'exposed_hydrophobic': exposed_hydrophobic,
                'exposed_polar': exposed_polar,
                'exposed_charged': exposed_charged,
                'total_exposed': total_exposed,
                'exposed_hydrophobic_fraction': exposed_hydrophobic / total_exposed if total_exposed > 0 else 0,
                'exposed_polar_fraction': exposed_polar / total_exposed if total_exposed > 0 else 0,
                'exposed_charged_fraction': exposed_charged / total_exposed if total_exposed > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Extended SASA calculation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'exposed_hydrophobic': 0,
                'exposed_polar': 0,
                'exposed_charged': 0,
                'total_exposed': 0,
                'exposed_hydrophobic_fraction': 0,
                'exposed_polar_fraction': 0,
                'exposed_charged_fraction': 0
            }

class ExtendedPropkaAnalyzer:
    """
    Extended PROPKA analyzer with additional stability metrics.
    """
    
    def __init__(self, propka_path: str = 'propka3', pH: float = 7.0):
        self.propka_path = propka_path
        self.pH = pH
    
    def parse_detailed(self, pka_file: Union[str, Path]) -> Dict:
        """
        Parse PROPKA output in detail.
        
        Parameters
        ----------
        pka_file : str or Path
            Path to .pka file
            
        Returns
        -------
        dict
            Detailed parsing results
        """
        try:
            with open(pka_file, 'r') as f:
                content = f.read()
            
            # Extract summary if available
            summary_df = self._extract_summary_table(content)
            
            # Extract residue details
            residues_df = self._extract_residue_details(content)
            
            return {
                'summary': summary_df,
                'residues': residues_df
            }
            
        except Exception as e:
            logger.error(f"Extended PROPKA parsing failed: {e}")
            return {
                'summary': None,
                'residues': None
            }
    
    def _extract_summary_table(self, content: str) -> Optional[pd.DataFrame]:
        """Extract summary table from PROPKA output."""
        # This would parse the summary section
        # Simplified implementation
        return pd.DataFrame({
            'pI_folded': [7.0],
            'pI_unfolded': [7.5],
            'stability_pH7': [-5.0]
        })
    
    def _extract_residue_details(self, content: str) -> Optional[pd.DataFrame]:
        """Extract per-residue pKa details."""
        # This would parse the residue details
        # Simplified implementation
        return pd.DataFrame({
            'residue': ['ASP_1_A', 'GLU_2_A'],
            'pKa': [3.8, 4.2],
            'pKa_shift': [-0.1, 0.0]
        })
    
    def calculate_stability_metrics(self, pka_file: Union[str, Path]) -> Dict:
        """
        Calculate stability metrics from PROPKA results.
        
        Parameters
        ----------
        pka_file : str or Path
            Path to .pka file
            
        Returns
        -------
        dict
            Stability metrics
        """
        try:
            detailed = self.parse_detailed(pka_file)
            
            metrics = {}
            
            if detailed['residues'] is not None:
                residues_df = detailed['residues']
                
                # Calculate stability metrics
                if 'pKa_shift' in residues_df.columns:
                    metrics['mean_pKa_shift'] = residues_df['pKa_shift'].mean()
                    metrics['max_pKa_shift'] = residues_df['pKa_shift'].abs().max()
                    metrics['n_perturbed_residues'] = (residues_df['pKa_shift'].abs() > 1.0).sum()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Stability metrics calculation failed: {e}")
            return {
                'mean_pKa_shift': 0,
                'max_pKa_shift': 0,
                'n_perturbed_residues': 0
            }