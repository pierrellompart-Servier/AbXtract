"""
Arpeggio interaction analysis module.

This module runs PDBe Arpeggio to calculate protein interactions including:
- Hydrogen bonds
- Salt bridges  
- Hydrophobic interactions
- Aromatic interactions
- Van der Waals contacts
"""

import json
import logging
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import gemmi
import numpy as np

logger = logging.getLogger(__name__)


class ArpeggioAnalyzer:
    """
    Analyzer for protein interactions using PDBe Arpeggio.
    
    Arpeggio calculates interatomic interactions including:
    - Hydrogen bonds (donor-acceptor)
    - Salt bridges (charge-charge)
    - Hydrophobic contacts
    - π-π stacking
    - Cation-π interactions
    - Van der Waals contacts
    
    Parameters
    ----------
    arpeggio_path : str
        Path to pdbe-arpeggio executable
    n_processes : int
        Number of parallel processes (default: 1)
        
    References
    ----------
    .. [1] Jubb et al. (2017) "Arpeggio: A Web Server for Calculating and 
           Visualising Interatomic Interactions in Protein Structures"
           J Mol Biol 429(3):365-371
           
    Examples
    --------
    >>> analyzer = ArpeggioAnalyzer()
    >>> results = analyzer.run("antibody.pdb")
    >>> print(f"Hydrogen bonds: {results['hydrogen_bonds']}")
    """
    
    def __init__(self, arpeggio_path: str = 'pdbe-arpeggio', n_processes: int = 1):
        self.arpeggio_path = arpeggio_path
        self.n_processes = n_processes
        self._check_arpeggio_installation()
        
        # Interaction types
        self.interaction_types = {
            'hydrogen_bond': 'Hydrogen bond',
            'salt_bridge': 'Salt bridge/ionic',
            'hydrophobic': 'Hydrophobic contact',
            'aromatic': 'Aromatic interaction',
            'cation_pi': 'Cation-π interaction',
            'pi_pi': 'π-π stacking',
            'vdw': 'Van der Waals contact',
            'clash': 'Steric clash'
        }
    
    def _check_arpeggio_installation(self):
        """Check if Arpeggio is installed."""
        try:
            result = subprocess.run(
                [self.arpeggio_path, '-h'],
                capture_output=True,
                text=True
            )
            if result.returncode not in [0, 1]:  # Arpeggio returns 1 for help
                logger.warning("Arpeggio not found. Some features will be unavailable.")
        except FileNotFoundError:
            logger.warning(f"Arpeggio not found at {self.arpeggio_path}")
    
    def run(self, pdb_file: Union[str, Path]) -> Dict:
        """
        Run Arpeggio analysis on PDB structure.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        dict
            Dictionary containing:
            - total_interactions: Total number of interactions
            - interaction_counts: Count by interaction type
            - hydrogen_bonds: Number of H-bonds
            - salt_bridges: Number of salt bridges
            - hydrophobic_contacts: Number of hydrophobic contacts
            - aromatic_interactions: Number of aromatic interactions
            - mean_interaction_distance: Mean distance of interactions
            - interaction_network: Network statistics
            - residue_interactions: Per-residue interaction counts
        """
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            try:
                # Convert PDB to CIF format (required by Arpeggio)
                cif_file = tmpdir / f"{pdb_file.stem}.cif"
                self._convert_to_cif(pdb_file, cif_file)
                
                # Run Arpeggio
                output_prefix = tmpdir / pdb_file.stem
                cmd = [
                    self.arpeggio_path,
                    '-m',  # Include main chain
                    str(cif_file),
                    '-o', str(output_prefix)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Arpeggio failed: {result.stderr}")
                    return self._empty_results()
                
                # Parse JSON output
                json_file = tmpdir / f"{pdb_file.stem}.json"
                if not json_file.exists():
                    logger.error(f"Arpeggio output not found: {json_file}")
                    return self._empty_results()
                
                return self._parse_arpeggio_output(json_file)
                
            except Exception as e:
                logger.error(f"Arpeggio analysis failed: {e}")
                return self._empty_results()
    
    def _convert_to_cif(self, pdb_file: Path, cif_file: Path):
        """Convert PDB to mmCIF format using gemmi."""
        try:
            structure = gemmi.read_structure(str(pdb_file))
            structure.write_cif(str(cif_file))
        except Exception as e:
            logger.error(f"PDB to CIF conversion failed: {e}")
            raise
    
    def _parse_arpeggio_output(self, json_file: Path) -> Dict:
        """Parse Arpeggio JSON output."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            return self._empty_results()
        
        # Count interaction types
        interaction_counts = Counter()
        contact_types = Counter()
        distances = []
        
        # Per-residue interactions
        residue_interactions = defaultdict(lambda: defaultdict(int))
        
        for interaction in data:
            # Get interaction type
            int_type = interaction.get('type', 'unknown')
            interaction_counts[int_type] += 1
            
            # Get contact types
            contacts = interaction.get('contact', [])
            for contact in contacts:
                contact_types[contact] += 1
            
            # Get distance
            distance = interaction.get('distance', 0)
            if distance > 0:
                distances.append(distance)
            
            # Track per-residue
            entities = interaction.get('interacting_entities', '').split('/')
            for entity in entities:
                if entity:
                    residue_interactions[entity][int_type] += 1
        
        # Calculate statistics
        results = {
            'total_interactions': len(data),
            'interaction_counts': dict(interaction_counts),
            'contact_counts': dict(contact_types),
            'mean_interaction_distance': np.mean(distances) if distances else 0,
            'std_interaction_distance': np.std(distances) if distances else 0,
            'min_interaction_distance': min(distances) if distances else 0,
            'max_interaction_distance': max(distances) if distances else 0
        }
        
        # Add specific interaction counts
        results['hydrogen_bonds'] = contact_types.get('hydrogen_bond', 0)
        results['salt_bridges'] = contact_types.get('ionic', 0)
        results['hydrophobic_contacts'] = contact_types.get('hydrophobic', 0)
        results['aromatic_interactions'] = (
            contact_types.get('aromatic', 0) +
            contact_types.get('pi_pi', 0) +
            contact_types.get('cation_pi', 0)
        )
        results['vdw_contacts'] = contact_types.get('vdw', 0)
        results['clashes'] = contact_types.get('clash', 0)
        
        # Network analysis
        results.update(self._analyze_interaction_network(residue_interactions))
        
        # Store per-residue data
        results['residue_interactions'] = dict(residue_interactions)
        
        # Calculate interaction density
        n_residues = len(residue_interactions)
        if n_residues > 0:
            results['interaction_density'] = results['total_interactions'] / n_residues
        else:
            results['interaction_density'] = 0
        
        return results
    
    def _analyze_interaction_network(self, residue_interactions: Dict) -> Dict:
        """Analyze interaction network properties."""
        network_stats = {}
        
        # Count highly connected residues (hubs)
        interaction_counts = {
            res: sum(counts.values())
            for res, counts in residue_interactions.items()
        }
        
        if interaction_counts:
            mean_interactions = np.mean(list(interaction_counts.values()))
            std_interactions = np.std(list(interaction_counts.values()))
            
            # Hubs are residues with interactions > mean + 2*std
            hub_threshold = mean_interactions + 2 * std_interactions
            hubs = [
                res for res, count in interaction_counts.items()
                if count > hub_threshold
            ]
            
            network_stats['n_hubs'] = len(hubs)
            network_stats['hub_residues'] = hubs
            network_stats['mean_degree'] = mean_interactions
            network_stats['std_degree'] = std_interactions
            network_stats['max_degree'] = max(interaction_counts.values())
            network_stats['min_degree'] = min(interaction_counts.values())
        else:
            network_stats = {
                'n_hubs': 0,
                'hub_residues': [],
                'mean_degree': 0,
                'std_degree': 0,
                'max_degree': 0,
                'min_degree': 0
            }
        
        return network_stats
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'total_interactions': 0,
            'interaction_counts': {},
            'contact_counts': {},
            'hydrogen_bonds': 0,
            'salt_bridges': 0,
            'hydrophobic_contacts': 0,
            'aromatic_interactions': 0,
            'vdw_contacts': 0,
            'clashes': 0,
            'mean_interaction_distance': 0,
            'std_interaction_distance': 0,
            'min_interaction_distance': 0,
            'max_interaction_distance': 0,
            'interaction_density': 0,
            'n_hubs': 0,
            'hub_residues': [],
            'mean_degree': 0,
            'std_degree': 0,
            'max_degree': 0,
            'min_degree': 0,
            'residue_interactions': {}
        }