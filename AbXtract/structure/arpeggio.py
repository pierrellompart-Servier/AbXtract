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
import os
import subprocess
from pathlib import Path
from multiprocessing import Pool
import gemmi  # Import gemmi Python module directly
import gemmi
import numpy as np

import os
import subprocess
from pathlib import Path
from multiprocessing import Pool

logger = logging.getLogger(__name__)
import os
import json
from operator import itemgetter
from collections import defaultdict, Counter
import numpy as np
#from multiprocessing import Pool
import pandas as pd

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
        cpus = 23

        # running on hydrogenated structures
        # from greiff paper:
        # Finally, we used Arpeggio (version 1.4.1) to calculate interatomic interactions 
        #   after converting antibody hydrogenated structures to cif format
        structures = [pdb_file]
        output_path = './data/test/arpeggio/'




        pool = Pool(processes=cpus)
        results = pool.map(self.run_arpeggio, structures)
        pool.close()
        pool.join()

        # Summary
        successful = sum(1 for r in results if r)

        path = './data/test/arpeggio/'
        files = [path + f for f in os.listdir(path) if f.endswith('.json')]

        results = []
        for file in files:
            output = self.run_arpu(file)
            results.append(output)

        d_protint = defaultdict(dict)

        for n, (f, r) in enumerate(zip(files, results)):
            #print(n)
            name = f.split('/')[-1].split('.')[0]
            d_protint[name] = r
        df = pd.DataFrame(d_protint.values(), index=d_protint.keys())
        df = df.reset_index()
        df = df.rename(columns={'index':'SeqID'})
        df
        
        return(df)

            

    
    
    
    
    
    def run_arpeggio(self, structure):
        """
        Convert PDB to CIF using gemmi Python API and run Arpeggio
        """
        try:

            # Define file paths
            pdb_file = structure
            cif_file = structure[:-4] + '.cif'

            # Convert from PDB to mmCIF format using gemmi Python API

            # Read PDB structure
            struct = gemmi.read_structure(pdb_file)

            # Write as mmCIF - try different methods based on gemmi version
            try:
                # Try newer gemmi method
                struct.write_cif(cif_file)
            except AttributeError:
                try:
                    # Try alternative method for writing CIF
                    struct.make_mmcif_document().write_file(cif_file)
                except AttributeError:
                    try:
                        # Another alternative for older versions
                        doc = struct.make_mmcif_document()
                        doc.write_file(cif_file)
                    except:
                        # If all else fails, use gemmi cif module directly
                        import gemmi.cif as cif
                        doc = struct.make_mmcif_document()
                        doc.write_file(cif_file)

            # Check if CIF file was created
            if not Path(cif_file).exists():
                return False

            # Run arpeggio on CIF file
            run_cmd = '/opt/conda/envs/propermab/bin/pdbe-arpeggio -m ' + cif_file + ' -o ' + output_path

            result = subprocess.call(run_cmd, shell=True)

            if result == 0:
                return True
            else:
                return False

        except Exception as e:
            return False





    
    
    
    
    
    
    
    def flatten(t):
        return [item for sublist in t for item in sublist]


    def run_arpu(self, f):

        #print(f)
        d = {}
        with open(f) as rf:
            json_d = json.load(rf)

            contacts = Counter(flatten([j['contact'] for j in json_d]))

            interactions = [itemgetter('distance', 'interacting_entities', 'type')(j) for j in json_d]
            interacting_entities = Counter([i[1] for i in interactions])
            types = Counter([i[2] for i in interactions])

            d.update(contacts)
            d.update(interacting_entities)
            d.update(types)
            d.update({'distance': np.mean([i[0] for i in interactions])})

        return d




            
            
            
            
            
            
            
            
            
            

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