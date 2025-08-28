"""
Solvent Accessible Surface Area (SASA) and Spatial Aggregation Propensity (SAP) calculations.

This module calculates:
- SASA: Total and per-residue solvent accessible surface area
- SAP: Spatial aggregation propensity score based on exposed hydrophobic surface
"""

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import freesasa
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)

# Hydrophobicity scale from Black & Mould (1991)
# DOI: 10.1016/0003-2697(91)90045-u
HYDROPHOBICITY_SCALE = {
    'ALA': 0.616, 'CYS': 0.680, 'ASP': 0.028, 'GLU': 0.043,
    'PHE': 1.000, 'GLY': 0.501, 'HIS': 0.165, 'ILE': 0.943,
    'LYS': 0.283, 'LEU': 0.943, 'MET': 0.738, 'ASN': 0.236,
    'PRO': 0.711, 'GLN': 0.251, 'ARG': 0.000, 'SER': 0.359,
    'THR': 0.450, 'VAL': 0.825, 'TRP': 0.878, 'TYR': 0.880
}

# Maximum SASA values for residues (Wilke: Tien et al. 2013)
# DOI: 10.1371/journal.pone.0080635
SASA_MAX = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
    "CYS": 167.0, "GLN": 225.0, "GLU": 223.0, "GLY": 104.0,
    "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
    "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
    "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0
}


class SASACalculator:
    """
    Calculator for Solvent Accessible Surface Area (SASA).
    
    SASA quantifies the surface area of a protein that is accessible to solvent.
    Higher SASA values indicate more exposed surface area.
    
    Parameters
    ----------
    probe_radius : float
        Radius of the solvent probe in Angstroms (default: 1.4)
    n_points : int
        Number of points for SASA calculation (default: 100)
    
    Attributes
    ----------
    backbone_atoms : set
        Set of backbone atom names
        
    Examples
    --------
    >>> calc = SASACalculator()
    >>> results = calc.calculate("antibody.pdb")
    >>> print(f"Total SASA: {results['total_sasa']:.2f} Å²")
    """
    
    def __init__(self, probe_radius: float = 1.4, n_points: int = 100, cutoff: float = 0.075):
        self.probe_radius = probe_radius
        self.n_points = n_points
        self.cutoff = cutoff
        self.backbone_atoms = {'C', 'N', 'CA', 'O'}
    
    def calculate(self, pdb_file: Union[str, Path]) -> Dict:
        """
        Calculate SASA for a protein structure.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        dict
            Dictionary containing:
            - total_sasa: Total SASA in Å²
            - total_sasa_sidechain: Total sidechain SASA
            - residue_sasa: Per-residue SASA values
            - relative_sasa: Per-residue relative SASA (0-1)
            - buried_residues: List of buried residues (relative SASA < 0.075)
        """
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        try:
            # Calculate SASA using FreeSASA
            structure = freesasa.Structure(str(pdb_file))
            result = freesasa.calc(structure)
            
            # Parse results
            residue_sasa = defaultdict(dict)
            total_sasa = result.totalArea()
            
            # Get per-atom SASA
            for atom_idx in range(structure.nAtoms()):
                res_name = structure.residueName(atom_idx).strip()
                res_num = structure.residueNumber(atom_idx).strip()
                chain = structure.chainLabel(atom_idx).strip()
                atom_name = structure.atomName(atom_idx).strip()
                area = result.atomArea(atom_idx)
                
                res_id = f"{res_name}_{res_num}_{chain}"
                residue_sasa[res_id][atom_name] = area
            
            # Calculate sidechain SASA
            sidechain_sasa = {}
            total_sidechain = 0
            
            for res_id, atoms in residue_sasa.items():
                sidechain_area = sum(
                    area for atom, area in atoms.items() 
                    if atom not in self.backbone_atoms
                )
                sidechain_sasa[res_id] = sidechain_area
                total_sidechain += sidechain_area
            
            # Calculate relative SASA
            relative_sasa = {}
            buried_residues = []
            
            parser = PDBParser(QUIET=True)
            bio_structure = parser.get_structure('struct', str(pdb_file))
            
            for model in bio_structure:
                for chain in model:
                    for residue in chain:
                        res_name = residue.resname
                        res_num = residue.id[1]
                        chain_id = chain.id
                        res_id = f"{res_name}_{res_num}_{chain_id}"
                        
                        if res_id in residue_sasa and res_name in SASA_MAX:
                            total_res_sasa = sum(residue_sasa[res_id].values())
                            rel_sasa = total_res_sasa / SASA_MAX[res_name]
                            relative_sasa[res_id] = rel_sasa
                            
                            if rel_sasa < self.cutoff:  # Buried threshold
                                buried_residues.append(res_id)

            
            return {
                'total_sasa': total_sasa,
                'total_sasa_sidechain': total_sidechain,
                'residue_sasa': dict(residue_sasa),
                'sidechain_sasa': sidechain_sasa,
                'relative_sasa': relative_sasa,
                'buried_residues': buried_residues,
                'n_buried': len(buried_residues)
            }
            
        except Exception as e:
            logger.error(f"SASA calculation failed: {e}")
            return {
                'total_sasa': np.nan,
                'total_sasa_sidechain': np.nan,
                'residue_sasa': {},
                'sidechain_sasa': {},
                'relative_sasa': {},
                'buried_residues': [],
                'n_buried': 0
            }
    
    def calculate_exposed_reference_sasa(self, tripeptide_dir: Union[str, Path]) -> Dict[str, float]:
        """
        Calculate reference SASA values from tripeptide structures.
        
        Parameters
        ----------
        tripeptide_dir : str or Path
            Directory containing tripeptide PDB files (A_X_A.pdb format)
            
        Returns
        -------
        dict
            Dictionary of amino acid to exposed sidechain SASA
        """
        tripeptide_dir = Path(tripeptide_dir)
        exposed_sasas = {}
        
        for aa in HYDROPHOBICITY_SCALE.keys():
            pdb_file = tripeptide_dir / f"A_{aa[:1]}_A.pdb"
            if not pdb_file.exists():
                logger.warning(f"Tripeptide file not found: {pdb_file}")
                continue
            
            try:
                structure = freesasa.Structure(str(pdb_file))
                result = freesasa.calc(structure)
                
                # Get middle residue (position 2) SASA
                sidechain_area = 0
                for atom_idx in range(structure.nAtoms()):
                    if structure.residueNumber(atom_idx).strip() == '2':
                        atom_name = structure.atomName(atom_idx).strip()
                        if atom_name not in self.backbone_atoms:
                            sidechain_area += result.atomArea(atom_idx)
                
                exposed_sasas[aa] = sidechain_area
                
            except Exception as e:
                logger.warning(f"Failed to calculate SASA for {aa}: {e}")
                exposed_sasas[aa] = 0
        
        return exposed_sasas


class SAPCalculator:
    """
    Calculator for Spatial Aggregation Propensity (SAP).
    
    SAP score quantifies aggregation propensity based on exposed hydrophobic
    surface area. Higher SAP scores indicate higher aggregation risk.
    
    References
    ----------
    .. [1] Chennamsetty et al. (2009) "Design of therapeutic proteins with 
           enhanced stability" PNAS 106(29):11937-42
    
    Parameters
    ----------
    sasa_calculator : SASACalculator, optional
        SASA calculator instance
        
    Examples
    --------
    >>> calc = SAPCalculator()
    >>> sap_score = calc.calculate("antibody.pdb", exposed_ref_sasa)
    >>> print(f"SAP score: {sap_score:.2f}")
    """
    
    def __init__(self, sasa_calculator: Optional[SASACalculator] = None):
        self.sasa_calc = sasa_calculator or SASACalculator()
        self.hydrophobicity = HYDROPHOBICITY_SCALE
    
    def calculate(
        self, 
        pdb_file: Union[str, Path],
        exposed_reference_sasa: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Calculate SAP score for a protein structure.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
        exposed_reference_sasa : dict, optional
            Reference SASA values for exposed amino acids
            If None, will use default values
            
        Returns
        -------
        dict
            Dictionary containing:
            - sap_score: Overall SAP score
            - residue_sap: Per-residue SAP values
            - high_sap_residues: Residues with SAP > threshold
        """
        pdb_file = Path(pdb_file)
        
        # Get SASA data
        sasa_data = self.sasa_calc.calculate(pdb_file)
        
        if exposed_reference_sasa is None:
            # Use default reference values
            exposed_reference_sasa = self._get_default_reference_sasa()
        
        # Calculate per-residue SAP
        residue_sap = {}
        
        for res_id, sidechain_sasa in sasa_data['sidechain_sasa'].items():
            res_type = res_id.split('_')[0]
            
            if res_type in exposed_reference_sasa and res_type in self.hydrophobicity:
                try:
                    # SAP = (SASA_residue / SASA_reference) * hydrophobicity
                    ref_sasa = exposed_reference_sasa[res_type]
                    if ref_sasa > 0:
                        sasa_ratio = sidechain_sasa / ref_sasa
                    else:
                        sasa_ratio = 0 if res_type == 'GLY' else sidechain_sasa
                    
                    sap = sasa_ratio * self.hydrophobicity[res_type]
                    residue_sap[res_id] = sap
                    
                except (KeyError, ZeroDivisionError) as e:
                    logger.warning(f"SAP calculation error for {res_id}: {e}")
                    residue_sap[res_id] = 0
        
        # Calculate total SAP score
        sap_score = sum(residue_sap.values())
        
        # Identify high SAP residues (top 10% or SAP > 0.5)
        sap_threshold = np.percentile(list(residue_sap.values()), 90) if residue_sap else 0.5
        high_sap_residues = [
            res_id for res_id, sap in residue_sap.items()
            if sap > max(sap_threshold, 0.5)
        ]
        
        return {
            'sap_score': sap_score,
            'residue_sap': residue_sap,
            'high_sap_residues': high_sap_residues,
            'n_high_sap': len(high_sap_residues),
            'mean_residue_sap': np.mean(list(residue_sap.values())) if residue_sap else 0,
            'max_residue_sap': max(residue_sap.values()) if residue_sap else 0
        }
    
    def _get_default_reference_sasa(self) -> Dict[str, float]:
        """Get default reference SASA values for amino acids."""
        # Default values based on extended tripeptide conformations
        return {
            'ALA': 67.0, 'ARG': 196.0, 'ASN': 113.0, 'ASP': 106.0,
            'CYS': 104.0, 'GLN': 144.0, 'GLU': 138.0, 'GLY': 0.0,
            'HIS': 151.0, 'ILE': 140.0, 'LEU': 137.0, 'LYS': 167.0,
            'MET': 160.0, 'PHE': 175.0, 'PRO': 105.0, 'SER': 73.0,
            'THR': 93.0, 'TRP': 217.0, 'TYR': 187.0, 'VAL': 117.0
        }