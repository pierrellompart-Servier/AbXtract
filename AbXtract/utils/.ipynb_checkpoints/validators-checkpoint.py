"""
Validation utilities for sequences, files, and parameters.
"""

import re
from pathlib import Path
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)

# Valid amino acid letters
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

# Valid numbering schemes
VALID_NUMBERING_SCHEMES = ['imgt', 'kabat', 'chothia', 'martin', 'wolfguy']

# Valid chain types
VALID_CHAIN_TYPES = {
    'antibody': ['H', 'L', 'K'],
    'nanobody': ['H'],
    'tcr': ['A', 'B', 'G', 'D']
}


def validate_sequence(
    sequence: str,
    allow_x: bool = False,
    allow_gaps: bool = False,
    min_length: int = 10,
    max_length: int = 1000
) -> tuple[bool, str]:
    """
    Validate amino acid sequence.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence
    allow_x : bool
        Allow X (unknown) amino acids
    allow_gaps : bool
        Allow gap characters (-)
    min_length : int
        Minimum sequence length
    max_length : int
        Maximum sequence length
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if not sequence:
        return False, "Empty sequence"
    
    # Convert to uppercase
    sequence = sequence.upper()
    
    # Check length
    if len(sequence) < min_length:
        return False, f"Sequence too short (minimum {min_length} residues)"
    
    if len(sequence) > max_length:
        return False, f"Sequence too long (maximum {max_length} residues)"
    
    # Build valid character set
    valid_chars = VALID_AMINO_ACIDS.copy()
    if allow_x:
        valid_chars.add('X')
    if allow_gaps:
        valid_chars.add('-')
    
    # Check for invalid characters
    invalid_chars = set(sequence) - valid_chars
    if invalid_chars:
        return False, f"Invalid characters found: {', '.join(invalid_chars)}"
    
    # Additional checks
    if not allow_gaps and '-' in sequence:
        return False, "Sequence contains gaps"
    
    # Check for unusual patterns
    if re.search(r'(.)\1{9,}', sequence):  # 10+ consecutive identical residues
        logger.warning("Sequence contains unusual repeat pattern")
    
    return True, ""


def validate_pdb_file(
    pdb_file: Union[str, Path],
    check_structure: bool = True
) -> tuple[bool, str]:
    """
    Validate PDB file.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to PDB file
    check_structure : bool
        Perform structural validation
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    pdb_file = Path(pdb_file)
    
    # Check file exists
    if not pdb_file.exists():
        return False, f"File not found: {pdb_file}"
    
    # Check file extension
    if pdb_file.suffix.lower() not in ['.pdb', '.ent']:
        logger.warning(f"Unusual PDB file extension: {pdb_file.suffix}")
    
    # Check file size
    if pdb_file.stat().st_size == 0:
        return False, "Empty PDB file"
    
    if pdb_file.stat().st_size > 100_000_000:  # 100 MB
        logger.warning("Very large PDB file (>100MB)")
    
    if check_structure:
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('test', str(pdb_file))
            
            # Check for atoms
            n_atoms = len(list(structure.get_atoms()))
            if n_atoms == 0:
                return False, "No atoms found in PDB file"
            
            # Check for chains
            n_chains = len(list(structure.get_chains()))
            if n_chains == 0:
                return False, "No chains found in PDB file"
            
            # Check for residues
            n_residues = len(list(structure.get_residues()))
            if n_residues == 0:
                return False, "No residues found in PDB file"
            
        except Exception as e:
            return False, f"Error parsing PDB file: {str(e)}"
    
    return True, ""


def validate_chain_type(
    chain_type: str,
    molecule_type: str = 'antibody'
) -> tuple[bool, str]:
    """
    Validate chain type.
    
    Parameters
    ----------
    chain_type : str
        Chain type identifier
    molecule_type : str
        Type of molecule (antibody, nanobody, tcr)
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    chain_type = chain_type.upper()
    
    if molecule_type not in VALID_CHAIN_TYPES:
        return False, f"Invalid molecule type: {molecule_type}"
    
    valid_chains = VALID_CHAIN_TYPES[molecule_type]
    
    if chain_type not in valid_chains:
        return False, f"Invalid chain type '{chain_type}' for {molecule_type}. Valid types: {', '.join(valid_chains)}"
    
    return True, ""


def is_valid_amino_acid_sequence(
    sequence: str,
    strict: bool = True
) -> bool:
    """
    Quick check if sequence is valid amino acid sequence.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence
    strict : bool
        If True, only allow standard amino acids
        
    Returns
    -------
    bool
        True if valid
    """
    if not sequence:
        return False
    
    sequence = sequence.upper()
    valid_chars = VALID_AMINO_ACIDS
    
    if not strict:
        valid_chars = VALID_AMINO_ACIDS | {'X', '-', '*'}
    
    return all(char in valid_chars for char in sequence)


def check_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if file exists.
    
    Parameters
    ----------
    file_path : str or Path
        File path
        
    Returns
    -------
    bool
        True if file exists
    """
    return Path(file_path).exists()


def validate_numbering_scheme(scheme: str) -> tuple[bool, str]:
    """
    Validate numbering scheme.
    
    Parameters
    ----------
    scheme : str
        Numbering scheme name
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    scheme = scheme.lower()
    
    if scheme not in VALID_NUMBERING_SCHEMES:
        return False, f"Invalid numbering scheme '{scheme}'. Valid schemes: {', '.join(VALID_NUMBERING_SCHEMES)}"
    
    return True, ""


def validate_ph_value(ph: float) -> tuple[bool, str]:
    """
    Validate pH value.
    
    Parameters
    ----------
    ph : float
        pH value
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if not isinstance(ph, (int, float)):
        return False, "pH must be a number"
    
    if ph < 0 or ph > 14:
        return False, "pH must be between 0 and 14"
    
    return True, ""


def validate_temperature(temp: float, unit: str = 'C') -> tuple[bool, str]:
    """
    Validate temperature value.
    
    Parameters
    ----------
    temp : float
        Temperature value
    unit : str
        Temperature unit (C, K, F)
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if not isinstance(temp, (int, float)):
        return False, "Temperature must be a number"
    
    unit = unit.upper()
    
    if unit == 'C':
        if temp < -273.15:
            return False, "Temperature below absolute zero"
        if temp > 150:
            logger.warning("Unusually high temperature for protein analysis")
    elif unit == 'K':
        if temp < 0:
            return False, "Temperature below absolute zero"
        if temp > 423:
            logger.warning("Unusually high temperature for protein analysis")
    elif unit == 'F':
        if temp < -459.67:
            return False, "Temperature below absolute zero"
        if temp > 302:
            logger.warning("Unusually high temperature for protein analysis")
    else:
        return False, f"Invalid temperature unit: {unit}"
    
    return True, ""