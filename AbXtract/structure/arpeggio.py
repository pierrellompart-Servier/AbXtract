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
    """
    
    def __init__(self, arpeggio_path: str = 'pdbe-arpeggio', temp_dir: Optional[str] = None, n_processes: int = 1):
        self.arpeggio_path = arpeggio_path
        self.temp_dir = Path(temp_dir) if temp_dir else Path.cwd() / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
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
                text=True,
                timeout=5
            )
            if result.returncode not in [0, 1, 2]:  # Arpeggio may return 1 or 2 for help
                logger.warning("Arpeggio not found or not working correctly")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning(f"Arpeggio not found at {self.arpeggio_path}")
    
    def run(self, pdb_file: Union[str, Path]) -> Dict:
        """
        Run Arpeggio analysis on PDB structure.
        """
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            try:
                # Convert PDB to CIF format
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
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    cwd=str(tmpdir)  # Run in temp directory
                )
                
                if result.returncode != 0:
                    logger.error(f"Arpeggio failed: {result.stderr}")
                    return self._empty_results()
                
                # Parse JSON output - Arpeggio creates .json file with the base name
                json_file = output_prefix.with_suffix('.json')
                
                # Check alternative naming patterns
                if not json_file.exists():
                    # Try without path prefix
                    json_file = tmpdir / f"{pdb_file.stem}.json"
                
                if not json_file.exists():
                    # List all json files in tmpdir for debugging
                    json_files = list(tmpdir.glob("*.json"))
                    if json_files:
                        json_file = json_files[0]
                    else:
                        logger.error(f"Arpeggio JSON output not found in {tmpdir}")
                        return self._empty_results()
                
                return self._parse_arpeggio_output(json_file)
                
            except Exception as e:
                logger.error(f"Arpeggio analysis failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return self._empty_results()

    def _convert_to_cif(self, pdb_file: Path, cif_file: Path):
        """Convert PDB to mmCIF format using gemmi."""
        try:
            import gemmi

            # Read PDB structure
            structure = gemmi.read_structure(str(pdb_file))

            # Create mmCIF document and write it
            doc = structure.make_mmcif_document()
            doc.write_file(str(cif_file))

        except ImportError:
            logger.error("gemmi not installed. Install with: pip install gemmi")
            raise
        except AttributeError:
            # Try alternative method for different gemmi versions
            try:
                import gemmi
                structure = gemmi.read_structure(str(pdb_file))

                # Alternative approach using gemmi cif module
                import gemmi.cif as cif
                block = cif.Block('structure')

                # Write using gemmi's built-in PDB to CIF conversion
                with open(str(cif_file), 'w') as f:
                    structure.write_minimal_cif(f)

            except AttributeError:
                # If that doesn't work either, try using command line
                logger.warning("gemmi Python API issues, trying command line conversion")
                result = subprocess.run(
                    ['gemmi', 'convert', str(pdb_file), str(cif_file)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    raise Exception(f"gemmi conversion failed: {result.stderr}")
        except Exception as e:
            logger.error(f"PDB to CIF conversion failed: {e}")
            raise    
    def _parse_arpeggio_output(self, json_file: Path) -> Dict:
        """Parse Arpeggio JSON output."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return self._empty_results()
        
        if not data:
            return self._empty_results()
        
        # Count interaction types
        interaction_counts = Counter()
        contact_types = Counter()
        distances = []
        
        # Per-residue interactions
        residue_interactions = defaultdict(lambda: defaultdict(int))
        
        for interaction in data:
            # Get interaction type - handle different possible keys
            int_type = interaction.get('type', interaction.get('interaction_type', 'unknown'))
            interaction_counts[int_type] += 1
            
            # Get contact types - may be a list or single value
            contacts = interaction.get('contact', interaction.get('contacts', []))
            if isinstance(contacts, str):
                contacts = [contacts]
            for contact in contacts:
                contact_types[contact] += 1
            
            # Get distance
            distance = interaction.get('distance')
            if distance is not None and distance > 0:
                distances.append(float(distance))
            
            # Track per-residue - handle different formats
            entities = interaction.get('interacting_entities', '')
            if not entities:
                # Try alternative keys
                entities = interaction.get('interacting_residues', '')
                if not entities:
                    # Try to construct from atom info
                    atom1 = interaction.get('atom_1', {})
                    atom2 = interaction.get('atom_2', {})
                    if atom1 and atom2:
                        entities = f"{atom1.get('residue', '')}/{atom2.get('residue', '')}"
            
            if isinstance(entities, str):
                entities = entities.split('/')
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
        
        # Map contact types to standard names
        results['hydrogen_bonds'] = sum(
            contact_types.get(k, 0) 
            for k in ['hydrogen', 'hbond', 'hydrogen_bond', 'h_bond']
        )
        results['salt_bridges'] = sum(
            contact_types.get(k, 0) 
            for k in ['ionic', 'salt_bridge', 'saltbridge']
        )
        results['hydrophobic_contacts'] = sum(
            contact_types.get(k, 0) 
            for k in ['hydrophobic', 'hydrophobic_contact']
        )
        results['aromatic_interactions'] = sum(
            contact_types.get(k, 0) 
            for k in ['aromatic', 'pi_pi', 'pi-pi', 'cation_pi', 'cation-pi', 'pi_stacking']
        )
        results['vdw_contacts'] = sum(
            contact_types.get(k, 0) 
            for k in ['vdw', 'van_der_waals', 'vdw_clash']
        )
        results['clashes'] = sum(
            contact_types.get(k, 0) 
            for k in ['clash', 'steric_clash', 'vdw_clash']
        )
        
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
            counts_list = list(interaction_counts.values())
            mean_interactions = np.mean(counts_list)
            std_interactions = np.std(counts_list)
            
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
            network_stats['max_degree'] = max(counts_list)
            network_stats['min_degree'] = min(counts_list)
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