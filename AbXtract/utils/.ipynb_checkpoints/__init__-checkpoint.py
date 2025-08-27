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
    'format_results'
]