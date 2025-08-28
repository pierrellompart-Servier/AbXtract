"""
Utility functions for antibody descriptor calculations.
"""

from .pdb_utils import (
    PDBParser,
    extract_sequences_from_pdb,
    extract_chains_from_pdb,
    remove_hydrogens_from_pdb,
    get_residue_coordinates,
    calculate_center_of_mass,
    calculate_distance_matrix,
    write_pdb
)

from .file_handlers import (
    FileHandler,
    read_fasta,
    write_fasta,
    read_csv,
    write_csv,
    read_json,
    write_json,
    create_temp_directory,
    cleanup_temp_files
)

from .validators import (
    validate_sequence,
    validate_pdb_file,
    validate_chain_type,
    is_valid_amino_acid_sequence,
    check_file_exists,
    validate_numbering_scheme
)

from .converters import (
    three_to_one_letter,
    one_to_three_letter,
    sequence_to_fasta,
    dataframe_to_dict,
    dict_to_dataframe,
    format_results
)

# Add the missing functions that are imported in core/main.py
def parse_sequence(sequence_input) -> str:
    """Parse sequence from various input formats."""
    from pathlib import Path
    from typing import Union
    from Bio import SeqIO
    
    if isinstance(sequence_input, (str, Path)):
        sequence_input_path = Path(str(sequence_input))
        
        # If it's a file path that exists
        if sequence_input_path.exists():
            if sequence_input_path.suffix.lower() in ['.fasta', '.fa']:
                # Read FASTA file
                records = list(SeqIO.parse(sequence_input_path, 'fasta'))
                if records:
                    return str(records[0].seq).upper()
                else:
                    raise ValueError("No sequences found in FASTA file")
            else:
                # Read as plain text
                with open(sequence_input_path, 'r') as f:
                    sequence = f.read().strip()
                    # Remove whitespace and non-amino acid characters
                    sequence = ''.join(c for c in sequence if c.isalpha())
                    return sequence.upper()
        else:
            # Direct sequence input
            sequence = str(sequence_input)
            # Remove whitespace and keep only amino acids
            sequence = ''.join(c for c in sequence if c.isalpha())
            return sequence.upper()
    else:
        raise TypeError(f"Invalid sequence input type: {type(sequence_input)}")


def parse_pdb(pdb_file):
    """Parse sequences from PDB file (alias to extract_sequences_from_pdb)."""
    return extract_sequences_from_pdb(pdb_file)


def validate_pdb(pdb_file):
    """Validate PDB file (alias to validate_pdb_file with exception raising)."""
    is_valid, error_message = validate_pdb_file(pdb_file)
    if not is_valid:
        raise ValueError(f"Invalid PDB file: {error_message}")
    return True


__all__ = [
    # PDB utilities
    'PDBParser',
    'extract_sequences_from_pdb',
    'extract_chains_from_pdb',
    'remove_hydrogens_from_pdb',
    'get_residue_coordinates',
    'calculate_center_of_mass',
    'calculate_distance_matrix',
    'write_pdb',
    # File handlers
    'FileHandler',
    'read_fasta',
    'write_fasta',
    'read_csv',
    'write_csv',
    'read_json',
    'write_json',
    'create_temp_directory',
    'cleanup_temp_files',
    # Validators
    'validate_sequence',
    'validate_pdb_file',
    'validate_chain_type',
    'is_valid_amino_acid_sequence',
    'check_file_exists',
    'validate_numbering_scheme',
    # Converters
    'three_to_one_letter',
    'one_to_three_letter',
    'sequence_to_fasta',
    'dataframe_to_dict',
    'dict_to_dataframe',
    'format_results',
    # Missing functions that are imported in main.py
    'parse_sequence',
    'parse_pdb',  
    'validate_pdb'
]