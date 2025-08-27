"""
PDB file handling and manipulation utilities.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from Bio import PDB
from Bio.PDB import PDBParser as BioPDBParser, PDBIO, Select
import prody as prd
import logging

logger = logging.getLogger(__name__)

# Standard amino acid conversion dictionaries
THREE_TO_ONE = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}

# Hydrogen atom pattern
_HYDROGEN_PATTERN = re.compile("[123 ]*H.*")


class RemoveHydrogensSelect(Select):
    """
    Selection class for removing hydrogen atoms from PDB structures.
    """
    
    def accept_atom(self, atom):
        """Accept only non-hydrogen atoms."""
        return 0 if _HYDROGEN_PATTERN.match(atom.get_id()) else 1


class PDBParser:
    """
    Enhanced PDB parser with various utility functions.
    
    This class provides methods for parsing PDB files and extracting
    various structural information needed for descriptor calculations.
    """
    
    def __init__(self, quiet: bool = True):
        """
        Initialize PDB parser.
        
        Parameters
        ----------
        quiet : bool
            Suppress warnings from Bio.PDB parser
        """
        self.parser = BioPDBParser(QUIET=quiet)
        self.quiet = quiet
    
    def parse(self, pdb_file: Union[str, Path]) -> PDB.Structure:
        """
        Parse PDB file and return structure object.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        Bio.PDB.Structure
            Parsed structure object
        """
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        return self.parser.get_structure(pdb_file.stem, str(pdb_file))
    
    def extract_sequences(self, pdb_file: Union[str, Path]) -> Dict[str, str]:
        """
        Extract sequences from all chains in PDB file.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        dict
            Dictionary mapping chain IDs to sequences
        """
        structure = self.parse(pdb_file)
        sequences = {}
        
        for model in structure:
            for chain in model:
                sequence = []
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Skip heterogens
                        resname = residue.get_resname()
                        if resname in THREE_TO_ONE:
                            sequence.append(THREE_TO_ONE[resname])
                
                if sequence:
                    sequences[chain.get_id()] = ''.join(sequence)
        
        return sequences
    
    def extract_chain_info(self, pdb_file: Union[str, Path]) -> pd.DataFrame:
        """
        Extract detailed information about chains.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
            
        Returns
        -------
        pd.DataFrame
            DataFrame with chain information
        """
        structure = self.parse(pdb_file)
        chain_info = []
        
        for model in structure:
            for chain in model:
                residues = [r for r in chain if r.get_id()[0] == ' ']
                atoms = list(chain.get_atoms())
                
                info = {
                    'model': model.get_id(),
                    'chain': chain.get_id(),
                    'n_residues': len(residues),
                    'n_atoms': len(atoms),
                    'first_residue': residues[0].get_id()[1] if residues else None,
                    'last_residue': residues[-1].get_id()[1] if residues else None
                }
                chain_info.append(info)
        
        return pd.DataFrame(chain_info)
    
    def get_residue_info(
        self, 
        pdb_file: Union[str, Path], 
        chain_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get detailed residue information.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
        chain_id : str, optional
            Specific chain to analyze
            
        Returns
        -------
        pd.DataFrame
            DataFrame with residue information
        """
        structure = self.parse(pdb_file)
        residue_info = []
        
        for model in structure:
            chains = [model[chain_id]] if chain_id else model
            
            for chain in chains:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Standard residues only
                        # Calculate residue properties
                        atoms = list(residue.get_atoms())
                        coords = [atom.get_coord() for atom in atoms]
                        
                        if coords:
                            coords = np.array(coords)
                            center = np.mean(coords, axis=0)
                            
                            # Get CA atom if available
                            ca_coord = None
                            if 'CA' in residue:
                                ca_coord = residue['CA'].get_coord()
                            
                            # Calculate B-factors
                            b_factors = [atom.get_bfactor() for atom in atoms]
                            
                            info = {
                                'chain': chain.get_id(),
                                'residue_num': residue.get_id()[1],
                                'residue_name': residue.get_resname(),
                                'residue_code': THREE_TO_ONE.get(residue.get_resname(), 'X'),
                                'n_atoms': len(atoms),
                                'center_x': center[0],
                                'center_y': center[1],
                                'center_z': center[2],
                                'ca_x': ca_coord[0] if ca_coord is not None else None,
                                'ca_y': ca_coord[1] if ca_coord is not None else None,
                                'ca_z': ca_coord[2] if ca_coord is not None else None,
                                'mean_bfactor': np.mean(b_factors),
                                'max_bfactor': np.max(b_factors),
                                'min_bfactor': np.min(b_factors)
                            }
                            residue_info.append(info)
        
        return pd.DataFrame(residue_info)
    
    def calculate_distance_matrix(
        self, 
        pdb_file: Union[str, Path], 
        chain_id: str,
        use_ca_only: bool = True
    ) -> np.ndarray:
        """
        Calculate distance matrix between residues.
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB file
        chain_id : str
            Chain identifier
        use_ca_only : bool
            Use only CA atoms (True) or residue centers (False)
            
        Returns
        -------
        np.ndarray
            Distance matrix
        """
        structure = self.parse(pdb_file)
        chain = structure[0][chain_id]
        
        coordinates = []
        for residue in chain:
            if residue.get_id()[0] == ' ':
                if use_ca_only and 'CA' in residue:
                    coordinates.append(residue['CA'].get_coord())
                else:
                    atoms = list(residue.get_atoms())
                    if atoms:
                        coords = np.array([atom.get_coord() for atom in atoms])
                        coordinates.append(np.mean(coords, axis=0))
        
        if not coordinates:
            raise ValueError(f"No valid coordinates found for chain {chain_id}")
        
        coordinates = np.array(coordinates)
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix


def extract_sequences_from_pdb(pdb_file: Union[str, Path]) -> Dict[str, str]:
    """
    Extract sequences from PDB file.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to PDB file
        
    Returns
    -------
    dict
        Dictionary mapping chain IDs to sequences
    """
    parser = PDBParser()
    return parser.extract_sequences(pdb_file)


def extract_chains_from_pdb(
    pdb_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> List[Path]:
    """
    Extract individual chains as separate PDB files.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to PDB file
    output_dir : str or Path, optional
        Output directory for chain files
        
    Returns
    -------
    list
        List of paths to created chain files
    """
    pdb_file = Path(pdb_file)
    output_dir = Path(output_dir) if output_dir else pdb_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parser = BioPDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file.stem, str(pdb_file))
    
    output_files = []
    io = PDBIO()
    
    for model in structure:
        for chain in model:
            # Create output filename
            output_file = output_dir / f"{pdb_file.stem}_chain_{chain.get_id()}.pdb"
            
            # Save chain
            io.set_structure(chain)
            io.save(str(output_file))
            output_files.append(output_file)
            
            logger.info(f"Extracted chain {chain.get_id()} to {output_file}")
    
    return output_files


def remove_hydrogens_from_pdb(
    pdb_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None
) -> Path:
    """
    Remove hydrogen atoms from PDB file.
    
    Parameters
    ----------
    pdb_file : str or Path
        Input PDB file
    output_file : str or Path, optional
        Output file path (defaults to input_file_noH.pdb)
        
    Returns
    -------
    Path
        Path to output file
    """
    pdb_file = Path(pdb_file)
    if output_file is None:
        output_file = pdb_file.parent / f"{pdb_file.stem}_noH.pdb"
    else:
        output_file = Path(output_file)
    
    parser = BioPDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file.stem, str(pdb_file))
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_file), RemoveHydrogensSelect())
    
    logger.info(f"Removed hydrogens from {pdb_file}, saved to {output_file}")
    return output_file


def get_residue_coordinates(
    pdb_file: Union[str, Path],
    chain_id: str,
    residue_numbers: Optional[List[int]] = None,
    atom_name: str = 'CA'
) -> pd.DataFrame:
    """
    Get coordinates for specified residues.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to PDB file
    chain_id : str
        Chain identifier
    residue_numbers : list, optional
        Specific residue numbers to extract
    atom_name : str
        Atom name to use for coordinates (default: 'CA')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with residue coordinates
    """
    parser = BioPDBParser(QUIET=True)
    structure = parser.get_structure('struct', str(pdb_file))
    
    coordinates = []
    
    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Standard residues only
                    res_num = residue.get_id()[1]
                    
                    if residue_numbers is None or res_num in residue_numbers:
                        if atom_name in residue:
                            coord = residue[atom_name].get_coord()
                            coordinates.append({
                                'chain': chain_id,
                                'residue_num': res_num,
                                'residue_name': residue.get_resname(),
                                'atom_name': atom_name,
                                'x': coord[0],
                                'y': coord[1],
                                'z': coord[2]
                            })
    
    return pd.DataFrame(coordinates)


def calculate_center_of_mass(
    pdb_file: Union[str, Path],
    chain_id: Optional[str] = None
) -> np.ndarray:
    """
    Calculate center of mass for structure or specific chain.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to PDB file
    chain_id : str, optional
        Specific chain to calculate COM for
        
    Returns
    -------
    np.ndarray
        Center of mass coordinates [x, y, z]
    """
    structure = prd.parsePDB(str(pdb_file), QUIET=True)
    
    if chain_id:
        selection = structure.select(f'chain {chain_id}')
        if selection is None:
            raise ValueError(f"Chain {chain_id} not found in structure")
        return prd.calcCenter(selection)
    else:
        return prd.calcCenter(structure)


def calculate_distance_matrix(
    pdb_file: Union[str, Path],
    chain_id: str,
    selection: str = 'ca'
) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    Calculate distance matrix using ProDy.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to PDB file
    chain_id : str
        Chain identifier
    selection : str
        Atom selection string (default: 'ca' for CA atoms)
        
    Returns
    -------
    tuple
        (distance_matrix, residue_info)
        - distance_matrix: np.ndarray of distances
        - residue_info: list of (residue_number, residue_name) tuples
    """
    structure = prd.parsePDB(str(pdb_file), QUIET=True)
    
    # Select chain and atoms
    selection_str = f'chain {chain_id} and {selection}'
    atoms = structure.select(selection_str)
    
    if atoms is None:
        raise ValueError(f"No atoms found for selection: {selection_str}")
    
    # Calculate distance matrix
    coords = atoms.getCoords()
    n_atoms = len(coords)
    dist_matrix = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # Get residue information
    residue_info = [(atom.getResnum(), atom.getResname()) for atom in atoms]
    
    return dist_matrix, residue_info


def write_pdb(
    structure: Union[PDB.Structure, PDB.Chain],
    output_file: Union[str, Path],
    select: Optional[Select] = None
) -> Path:
    """
    Write structure or chain to PDB file.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure or Bio.PDB.Chain
        Structure or chain to write
    output_file : str or Path
        Output file path
    select : Bio.PDB.Select, optional
        Selection object for filtering
        
    Returns
    -------
    Path
        Path to written file
    """
    output_file = Path(output_file)
    io = PDBIO()
    io.set_structure(structure)
    
    if select:
        io.save(str(output_file), select)
    else:
        io.save(str(output_file))
    
    return output_file