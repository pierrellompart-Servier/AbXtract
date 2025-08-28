"""
Charge distribution and heterogeneity analysis for protein structures.

This module calculates spatial distribution of charged residues and
charge heterogeneity metrics important for protein stability and solubility.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import prody as prd
from collections import defaultdict

logger = logging.getLogger(__name__)


class ChargeAnalyzer:
    """
    Analyzer for protein charge distribution and heterogeneity.
    
    Charge distribution affects:
    - Protein solubility
    - Aggregation propensity  
    - Viscosity at high concentrations
    - Electrostatic interactions
    
    Parameters
    ----------
    pH : float
        pH for charge calculations (default: 7.0)
        
    Examples
    --------
    >>> analyzer = ChargeAnalyzer(pH=7.4)
    >>> results = analyzer.calculate("antibody.pdb")
    >>> print(f"Positive charge heterogeneity: {results['positive_heterogeneity']:.2f}")
    """
    
    def __init__(self, pH: float = 7.0, distance_cutoff: float = 4.0):
        self.pH = pH
        self.distance_cutoff = distance_cutoff
        self.positive_residues = {'HIS', 'ARG', 'LYS'}
        self.negative_residues = {'ASP', 'GLU'}
        
        # pKa values for titratable residues
        self.pka_values = {
            'ASP': 3.9, 'GLU': 4.3, 'HIS': 6.0,
            'CYS': 8.3, 'TYR': 10.1, 'LYS': 10.5,
            'ARG': 12.5
        }
    
    def calculate(self, pdb_file: Union[str, Path]) -> Dict:
        """
        Calculate charge distribution metrics.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        dict
            Dictionary containing:
            - net_charge: Net charge at specified pH
            - positive_charge: Total positive charge
            - negative_charge: Total negative charge
            - charge_asymmetry: Asymmetry in charge distribution
            - positive_heterogeneity: Std dev of positive charge distances
            - negative_heterogeneity: Std dev of negative charge distances
            - dipole_moment: Magnitude of dipole moment
        """
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        try:
            # Parse structure
            protein = prd.parsePDB(str(pdb_file), QUIET=True)
            
            # Get center of geometry
            central_atom = prd.pickCentralAtom(protein)
            center = central_atom.getCoords() if central_atom else np.zeros(3)
            
            # Center the protein
            prd.moveAtoms(protein, to=np.zeros(3))
            
            # Collect charged residues
            positive_positions = []
            negative_positions = []
            
            for residue in protein.iterResidues():
                res_name = residue.getResname()
                central_res_atom = prd.pickCentralAtom(residue)
                
                if central_res_atom is None:
                    continue
                
                position = central_res_atom.getCoords()
                
                if res_name in self.positive_residues:
                    charge = self._calculate_charge(res_name, self.pH)
                    if charge > 0:
                        positive_positions.append(position)
                elif res_name in self.negative_residues:
                    charge = self._calculate_charge(res_name, self.pH)
                    if charge < 0:
                        negative_positions.append(position)
            
            # Calculate distances from center
            positive_distances = [
                np.linalg.norm(pos) for pos in positive_positions
            ]
            negative_distances = [
                np.linalg.norm(pos) for pos in negative_positions
            ]
            
            # Calculate heterogeneity (standard deviation of distances)
            positive_heterogeneity = (
                np.std(positive_distances) if positive_distances else 0
            )
            negative_heterogeneity = (
                np.std(negative_distances) if negative_distances else 0
            )
            
            # Calculate charge asymmetry
            charge_asymmetry = self._calculate_asymmetry(
                positive_positions, negative_positions
            )
            
            # Calculate dipole moment
            dipole_moment = self._calculate_dipole_moment(
                positive_positions, negative_positions
            )
            
            # Count charges
            n_positive = len(positive_positions)
            n_negative = len(negative_positions)
            net_charge = n_positive - n_negative
            
            # Calculate charge density (charges per 1000 Å³)
            volume = self._estimate_volume(protein)
            charge_density = abs(net_charge) / (volume / 1000) if volume > 0 else 0
            
            return {
                'net_charge': net_charge,
                'positive_charge': n_positive,
                'negative_charge': n_negative,
                'charge_asymmetry': charge_asymmetry,
                'positive_heterogeneity': positive_heterogeneity,
                'negative_heterogeneity': negative_heterogeneity,
                'dipole_moment': dipole_moment,
                'charge_density': charge_density,
                'positive_patches': self._find_charge_patches(
                    positive_positions, threshold=4.0
                ),
                'negative_patches': self._find_charge_patches(
                    negative_positions, threshold=4.0
                )
            }
            
        except Exception as e:
            logger.error(f"Charge analysis failed: {e}")
            return {
                'net_charge': 0,
                'positive_charge': 0,
                'negative_charge': 0,
                'charge_asymmetry': 0,
                'positive_heterogeneity': 0,
                'negative_heterogeneity': 0,
                'dipole_moment': 0,
                'charge_density': 0,
                'positive_patches': 0,
                'negative_patches': 0
            }
    
    def _calculate_charge(self, residue: str, pH: float) -> float:
        """
        Calculate charge of a residue at given pH.
        
        Uses Henderson-Hasselbalch equation.
        """
        if residue not in self.pka_values:
            return 0
        
        pka = self.pka_values[residue]
        
        if residue in {'ASP', 'GLU', 'CYS', 'TYR'}:
            # Acidic residues
            charge = -1 / (1 + 10**(pka - pH))
        else:
            # Basic residues  
            charge = 1 / (1 + 10**(pH - pka))
        
        return charge
    
    def _calculate_asymmetry(
        self, 
        positive_positions: List[np.ndarray],
        negative_positions: List[np.ndarray]
    ) -> float:
        """Calculate asymmetry in charge distribution."""
        if not positive_positions or not negative_positions:
            return 0
        
        # Calculate center of positive and negative charges
        pos_center = np.mean(positive_positions, axis=0)
        neg_center = np.mean(negative_positions, axis=0)
        
        # Distance between charge centers
        asymmetry = np.linalg.norm(pos_center - neg_center)
        
        return asymmetry
    
    def _calculate_dipole_moment(
        self,
        positive_positions: List[np.ndarray],
        negative_positions: List[np.ndarray]
    ) -> float:
        """Calculate dipole moment magnitude."""
        dipole = np.zeros(3)
        
        for pos in positive_positions:
            dipole += pos
        for pos in negative_positions:
            dipole -= pos
        
        return np.linalg.norm(dipole)
    
    def _find_charge_patches(
        self,
        positions: List[np.ndarray],
        threshold: Optional[float] = None
    ) -> int:
        if threshold is None:
            threshold = self.distance_cutoff
        """
        Find patches of same-charge residues.
        
        A patch is defined as residues within threshold distance.
        """
        if len(positions) < 2:
            return 0
        
        patches = 0
        visited = set()
        
        for i, pos1 in enumerate(positions):
            if i in visited:
                continue
            
            patch = {i}
            for j, pos2 in enumerate(positions):
                if i != j and np.linalg.norm(pos1 - pos2) < threshold:
                    patch.add(j)
            
            if len(patch) >= 3:  # Patch of 3+ residues
                patches += 1
                visited.update(patch)
        
        return patches
    
    def _estimate_volume(self, protein) -> float:
        """Estimate protein volume using convex hull."""
        try:
            coords = protein.getCoords()
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            return hull.volume
        except:
            # Fallback: estimate from radius of gyration
            rg = prd.calcGyradius(protein)
            return (4/3) * np.pi * (rg ** 3)