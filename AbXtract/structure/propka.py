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
    
    def __init__(self, propka_path: str = 'propka3', pH: float = 7.0, temp_dir: Optional[str] = None, pH_range: Tuple[float, float] = (0, 14)):
        self.propka_path = propka_path
        self.pH = pH
        self.temp_dir = Path(temp_dir) if temp_dir else Path.cwd() / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
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
        Run PROPKA analysis on PDB structure with fixed command arguments.
        """
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Run PROPKA with corrected arguments
            # FIXED: Use correct PROPKA3 command format
            cmd = [self.propka_path, '-q', str(pdb_file)]
            
            # Change to temp directory so output goes there
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=str(tmpdir)  # Run in temp directory
            )
            
            if result.returncode != 0:
                logger.error(f"PROPKA failed: {result.stderr}")
                return self._empty_results()
            
            # Look for output file (PROPKA creates .pka file with same name as input)
            pka_file = tmpdir / f"{pdb_file.stem}.pka"
            if not pka_file.exists():
                # Try alternative naming
                for f in tmpdir.glob("*.pka"):
                    pka_file = f
                    break
                else:
                    logger.error(f"PROPKA output not found in {tmpdir}")
                    return self._empty_results()
            
            return self._parse_pka_file(pka_file)

    
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
        print('_extract_charge_profile')
        charge_data = self._extract_charge_profile(pka_file)
        results.update(charge_data)
        
        # Extract free energy profile
        print('_extract_energy_profile')
        energy_data = self._extract_energy_profile(content)
        results.update(energy_data)
        
        # Calculate additional metrics
        print('_calculate_stability_metrics')
        results.update(self._calculate_stability_metrics(residue_pka, desolvation_effects))
        
        print('parse_propka_file')
        df_pdb, df_residues = self.parse_propka_file(pka_file)
        results.update(df_pdb)

        
        return results, df_residues
    

    
    def _extract_charge_profile(self, file_path: str) -> Dict:
        """Extract charge vs pH profile."""
        """Extract charge vs pH profile with fixed parsing."""
        charge_data = {
            'charge_pH7_folded': 0,
            'charge_pH7_unfolded': 0,
            'charge_profile': {}
        }
        with open(file_path, 'r') as f:
            content = f.read()

        # Look for the charge section with more flexible pattern
        charge_section = re.search(
            r'Protein charge of folded and unfolded state.*?\n.*?pH.*?unfolded.*?folded.*?\n(.*?)(?:\nThe pI|$)',
            content, re.DOTALL
        )
        
        if charge_section:
            for line in charge_section.group(1).strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('The pI'):
                    break
                    
                # Parse lines with pH values
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        pH = float(parts[0])
                        unfolded = float(parts[1])
                        folded = float(parts[2])
                        
                        charge_data['charge_profile'][pH] = {
                            'folded': folded,
                            'unfolded': unfolded
                        }
                        
                        # Store pH 7 values
                        if abs(pH - 7.0) < 0.01:
                            charge_data['charge_pH7_folded'] = folded
                            charge_data['charge_pH7_unfolded'] = unfolded
                    except (ValueError, IndexError):
                        continue
        
        return charge_data

    


    def parse_propka_file(self, file_path):
        """
        Parse a PROPKA .pka file and extract data into two dataframes:
        1. df_pdb: One row per PDB file with columns for each property at each pH
        2. df_residues: Per residue desolvation effects data

        Parameters:
        -----------
        file_path : str or Path
            Path to the .pka file

        Returns:
        --------
        tuple: (df_pdb, df_residues)
            - df_pdb: DataFrame with one row containing all pH-dependent properties
            - df_residues: DataFrame with residue-specific desolvation effects
        """
        with open(file_path, 'r') as f:
            content = f.read()
        # Initialize the PDB data dictionary
        pdb_data = {}

        # Extract pI values (constant across all pH)
        pi_match = re.search(r'The pI is\s+([\d.]+)\s+\(folded\)\s+and\s+([\d.]+)\s+\(unfolded\)', content)
        pi_folded = float(pi_match.group(1)) if pi_match else None
        pi_unfolded = float(pi_match.group(2)) if pi_match else None

        # Parse Free energy data - only at specific pH points (0, 1, 2, etc.)
        free_energy_dict = {}
        free_energy_section = re.search(r'Free energy of\s+folding.*?\n((?:\s*[\d.]+\s+[\d.-]+\n)+)', content, re.DOTALL)
        if free_energy_section:
            for line in free_energy_section.group(1).strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 2:
                    ph = float(parts[0])
                    energy = float(parts[1])
                    free_energy_dict[ph] = energy

        # Parse Protein charge data - available at all 0.1 pH increments
        charge_dict = {'folded': {}, 'unfolded': {}}
        charge_section = re.search(r'Protein charge of folded and unfolded.*?\n\s*pH\s+unfolded\s+folded\n((?:\s*[\d.]+\s+[\d.-]+\s+[\d.-]+\n)+)', content, re.DOTALL)
        if charge_section:
            for line in charge_section.group(1).strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 3:
                    ph = float(parts[0])
                    unfolded = float(parts[1])
                    folded = float(parts[2])
                    charge_dict['unfolded'][ph] = unfolded
                    charge_dict['folded'][ph] = folded

        # Get all pH values from charge data (most complete set)
        all_ph_values = sorted(charge_dict['folded'].keys()) if charge_dict['folded'] else []

        # If no charge data, use free energy pH values
        if not all_ph_values:
            all_ph_values = sorted(free_energy_dict.keys())

        # Build the PDB data row
        for ph in all_ph_values:
            # pI values (constant for all pH)
            pdb_data[f'pI_Folded_pH_{ph:.2f}'] = pi_folded
            pdb_data[f'pI_Unfolded_pH_{ph:.2f}'] = pi_unfolded

            # Protein charge values (available for all pH)
            pdb_data[f'Protein_Charge_Folded_pH_{ph:.2f}'] = charge_dict['folded'].get(ph, None)
            pdb_data[f'Protein_Charge_Unfolded_pH_{ph:.2f}'] = charge_dict['unfolded'].get(ph, None)

            # Free energy values (only available at integer pH values)
            pdb_data[f'Free_Energy_kcal_mol_pH_{ph:.2f}'] = free_energy_dict.get(ph, None)

        # Create PDB dataframe (one row)
        df_pdb = pd.DataFrame([pdb_data])

        # Parse Residue desolvation effects
        residues_data = []

        # Find the desolvation effects section
        desolv_section = re.search(
            r'DESOLVATION\s+EFFECTS.*?\n\s*RESIDUE\s+pKa\s+BURIED\s+REGULAR\s+RE.*?\n-+\n(.*?)(?:\n-+|\nSUMMARY)',
            content, re.DOTALL
        )

        if desolv_section:
            lines = desolv_section.group(1).strip().split('\n')

            current_residue = None
            for line in lines:
                if not line.strip():
                    continue

                # Enhanced pattern to handle various formats
                # Match lines like: ASP  54 H   2.55     0 %    0.35  197   0.00    0
                match = re.match(r'^([A-Z+-]+)\s+(\d+)\s+([A-Z])\s+([\d.]+)\s+(\d+)\s*%\s+([\d.]+)\s+(\d+)', line)
                if match:
                    res_type = match.group(1)
                    res_num = int(match.group(2))
                    chain = match.group(3)
                    pka = float(match.group(4))
                    buried_pct = int(match.group(5))
                    buried = f"{buried_pct} %"
                    regular = float(match.group(6))
                    re_value = int(match.group(7))

                    # Check if we already have this residue (avoid duplicates from continuation lines)
                    exists = False
                    for existing_res in residues_data:
                        if (existing_res['Residue_Type'] == res_type and 
                            existing_res['Residue_Number'] == res_num and 
                            existing_res['Chain'] == chain):
                            exists = True
                            break

                    if not exists:
                        residue_entry = {
                            'Residue_Type': res_type,
                            'Residue_Number': res_num,
                            'Chain': chain,
                            'pKa': pka,
                            'BURIED': buried,
                            'REGULAR': regular,
                            'RE': re_value
                        }
                        residues_data.append(residue_entry)

        # If desolvation parsing didn't work or is incomplete, try the summary section
        summary_section = re.search(r'SUMMARY OF THIS PREDICTION\s*\n.*?Group\s+pKa\s+model-pKa.*?\n(.*?)(?:\n-+|\Z)', 
                                    content, re.DOTALL)

        if summary_section:
            existing_residues = set((r['Residue_Type'], r['Residue_Number'], r['Chain']) 
                                   for r in residues_data)

            for line in summary_section.group(1).strip().split('\n'):
                match = re.match(r'\s*([A-Z+-]+)\s+(\d+)\s+([A-Z])\s+([\d.]+)\s+([\d.]+)', line)
                if match:
                    res_type = match.group(1)
                    res_num = int(match.group(2))
                    chain = match.group(3)
                    pka = float(match.group(4))

                    # Only add if not already in the list
                    if (res_type, res_num, chain) not in existing_residues:
                        # These residues don't have desolvation data
                        residue_entry = {
                            'Residue_Type': res_type,
                            'Residue_Number': res_num,
                            'Chain': chain,
                            'pKa': pka,
                            'BURIED': None,
                            'REGULAR': None,
                            'RE': None
                        }
                        residues_data.append(residue_entry)

        # Create residue dataframe
        df_residues = pd.DataFrame(residues_data) if residues_data else pd.DataFrame()

        # Sort residues by residue number
        if not df_residues.empty:
            df_residues = df_residues.sort_values('Residue_Number').reset_index(drop=True)

            

        rows = []

        with open(file_path, "r") as f_in:
            keep_mode = False
            for r in f_in:
                r_stripped = r.strip()
                if r_stripped.startswith("RESIDUE"):
                    keep_mode = True
                    continue
                if r_stripped.startswith("SUMMARY"):
                    keep_mode = False
                    continue

                if keep_mode and "%" in r:
                    line_split = r.split()
                    rows.append([
                        line_split[0],   # Residue_Type
                        line_split[1],   # Residue_Number
                        line_split[2],   # Chain
                        line_split[4],   # Buried ratio
                        line_split[6],   # Disolvation regular Cst
                        line_split[7],   # Disolvation regular Nb
                        line_split[8],   # Effects RE Cst
                        line_split[9],   # Effects RE Nb
                    ])

        df_res_comp = pd.DataFrame(
            rows,
            columns=[
                "Residue_Type",
                "Residue_Number",
                "Chain",
                "Buried ratio",
                "Disolvation regular Cst",
                "Disolvation regular Nb",
                "Effects RE Cst",
                "Effects RE Nb",
            ]
        )


        for col in df_res_comp.columns.tolist()[3:]:
            df_residues[col] = df_res_comp[col]

        df_residues_ka = df_residues.drop( ["BURIED", "REGULAR", "RE"], axis =1) 



        return df_pdb, df_residues_ka

    
    
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