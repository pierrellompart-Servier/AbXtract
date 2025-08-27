"""
Conversion utilities for various data formats and types.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Amino acid conversion dictionaries
THREE_TO_ONE = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
    # Non-standard amino acids
    'SEC': 'U', 'PYL': 'O',
    # Modified amino acids (map to standard)
    'MSE': 'M',  # Selenomethionine
    'PTR': 'Y',  # Phosphotyrosine
    'SEP': 'S',  # Phosphoserine
    'TPO': 'T',  # Phosphothreonine
}

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items() if len(k) == 3}


def three_to_one_letter(
    three_letter_code: str,
    unknown: str = 'X'
) -> str:
    """
    Convert three-letter amino acid code to one-letter code.
    
    Parameters
    ----------
    three_letter_code : str
        Three-letter amino acid code
    unknown : str
        Character to use for unknown amino acids
        
    Returns
    -------
    str
        One-letter amino acid code
    """
    three_letter_code = three_letter_code.upper()
    return THREE_TO_ONE.get(three_letter_code, unknown)


def one_to_three_letter(
    one_letter_code: str,
    unknown: str = 'UNK'
) -> str:
    """
    Convert one-letter amino acid code to three-letter code.
    
    Parameters
    ----------
    one_letter_code : str
        One-letter amino acid code
    unknown : str
        Code to use for unknown amino acids
        
    Returns
    -------
    str
        Three-letter amino acid code
    """
    one_letter_code = one_letter_code.upper()
    return ONE_TO_THREE.get(one_letter_code, unknown)


def sequence_to_fasta(
    sequence: str,
    seq_id: str = "sequence",
    line_length: int = 60
) -> str:
    """
    Convert sequence to FASTA format string.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence
    seq_id : str
        Sequence identifier
    line_length : int
        Number of characters per line
        
    Returns
    -------
    str
        FASTA formatted string
    """
    # Clean sequence
    sequence = re.sub(r'\s+', '', sequence.upper())
    
    # Format sequence with line breaks
    lines = [f">{seq_id}"]
    for i in range(0, len(sequence), line_length):
        lines.append(sequence[i:i + line_length])
    
    return '\n'.join(lines)


def fasta_to_dict(fasta_string: str) -> Dict[str, str]:
    """
    Parse FASTA format string to dictionary.
    
    Parameters
    ----------
    fasta_string : str
        FASTA formatted string
        
    Returns
    -------
    dict
        Dictionary mapping sequence IDs to sequences
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    for line in fasta_string.strip().split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            current_id = line[1:].split()[0]  # Take first word as ID
            current_seq = []
        else:
            current_seq.append(line)
    
    # Don't forget the last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    
    return sequences


def dataframe_to_dict(
    df: pd.DataFrame,
    orient: str = 'records'
) -> Union[Dict, List[Dict]]:
    """
    Convert DataFrame to dictionary with various orientations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert
    orient : str
        Orientation for conversion ('records', 'dict', 'list', 'index')
        
    Returns
    -------
    dict or list
        Converted data
    """
    # Handle NaN values
    df_clean = df.copy()
    df_clean = df_clean.where(pd.notnull(df_clean), None)
    
    return df_clean.to_dict(orient=orient)


def dict_to_dataframe(
    data: Union[Dict, List[Dict]],
    transpose: bool = False
) -> pd.DataFrame:
    """
    Convert dictionary to DataFrame.
    
    Parameters
    ----------
    data : dict or list of dict
        Data to convert
    transpose : bool
        Whether to transpose the result
        
    Returns
    -------
    pd.DataFrame
        Converted DataFrame
    """
    df = pd.DataFrame(data)
    
    if transpose:
        df = df.T
    
    return df


def format_results(
    results: Dict[str, Any],
    flatten: bool = True,
    prefix_keys: bool = True
) -> Dict[str, Any]:
    """
    Format nested results dictionary for output.
    
    Parameters
    ----------
    results : dict
        Nested results dictionary
    flatten : bool
        Whether to flatten nested structures
    prefix_keys : bool
        Whether to prefix keys with parent keys
        
    Returns
    -------
    dict
        Formatted results
    """
    if not flatten:
        return results
    
    formatted = {}
    
    def _flatten(obj, parent_key=''):
        """Recursively flatten nested dictionary."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}_{key}" if parent_key and prefix_keys else key
                
                if isinstance(value, dict):
                    _flatten(value, new_key)
                elif isinstance(value, (list, tuple)):
                    if len(value) > 0 and isinstance(value[0], (int, float, str)):
                        # Store as string representation for simple lists
                        formatted[new_key] = str(value)
                    else:
                        # Complex list - store length
                        formatted[f"{new_key}_count"] = len(value)
                else:
                    formatted[new_key] = value
        else:
            formatted[parent_key] = obj
    
    _flatten(results)
    return formatted


def numbering_to_string(
    numbering: List[Tuple[Tuple[int, str], str]],
    include_gaps: bool = False
) -> str:
    """
    Convert numbered sequence to string representation.
    
    Parameters
    ----------
    numbering : list
        List of ((position, insertion), amino_acid) tuples
    include_gaps : bool
        Whether to include gap characters
        
    Returns
    -------
    str
        String representation of numbering
    """
    positions = []
    sequence = []
    
    for (pos, ins), aa in numbering:
        if aa != '-' or include_gaps:
            positions.append(f"{pos}{ins.strip()}")
            sequence.append(aa)
    
    return ''.join(sequence)


def string_to_numbering(
    sequence: str,
    start_position: int = 1
) -> List[Tuple[Tuple[int, str], str]]:
    """
    Convert sequence string to simple numbering.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence
    start_position : int
        Starting position number
        
    Returns
    -------
    list
        List of ((position, insertion), amino_acid) tuples
    """
    numbering = []
    
    for i, aa in enumerate(sequence):
        position = start_position + i
        numbering.append(((position, ' '), aa))
    
    return numbering


def merge_dataframes(
    dfs: List[pd.DataFrame],
    on: Optional[Union[str, List[str]]] = None,
    how: str = 'outer',
    suffixes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Merge multiple DataFrames.
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        DataFrames to merge
    on : str or list, optional
        Column(s) to merge on
    how : str
        Merge method ('outer', 'inner', 'left', 'right')
    suffixes : list, optional
        Suffixes for overlapping columns
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    if len(dfs) == 1:
        return dfs[0]
    
    # Generate suffixes if not provided
    if suffixes is None:
        suffixes = [f"_{i}" for i in range(len(dfs))]
    
    # Merge iteratively
    result = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        suffix = ('', suffixes[i]) if i < len(suffixes) else ('_x', '_y')
        
        if on is not None:
            result = result.merge(df, on=on, how=how, suffixes=suffix)
        else:
            # Merge on index
            result = result.merge(
                df, 
                left_index=True, 
                right_index=True, 
                how=how, 
                suffixes=suffix
            )
    
    return result


def normalize_dataframe(
    df: pd.DataFrame,
    method: str = 'minmax',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize DataFrame columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize
    method : str
        Normalization method ('minmax', 'zscore', 'robust')
    columns : list, optional
        Specific columns to normalize
        
    Returns
    -------
    pd.DataFrame
        Normalized DataFrame
    """
    df_norm = df.copy()
    
    if columns is None:
        # Select numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        values = df[col].values
        
        if method == 'minmax':
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            if max_val > min_val:
                df_norm[col] = (values - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)
            if std_val > 0:
                df_norm[col] = (values - mean_val) / std_val
        
        elif method == 'robust':
            median_val = np.nanmedian(values)
            mad_val = np.nanmedian(np.abs(values - median_val))
            if mad_val > 0:
                df_norm[col] = (values - median_val) / (1.4826 * mad_val)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return df_norm


def aggregate_results(
    results_list: List[Dict],
    aggregation: str = 'mean'
) -> Dict:
    """
    Aggregate multiple result dictionaries.
    
    Parameters
    ----------
    results_list : list of dict
        List of result dictionaries
    aggregation : str
        Aggregation method ('mean', 'median', 'min', 'max', 'std')
        
    Returns
    -------
    dict
        Aggregated results
    """
    if not results_list:
        return {}
    
    if len(results_list) == 1:
        return results_list[0]
    
    # Collect values by key
    value_dict = defaultdict(list)
    
    for results in results_list:
        flat_results = format_results(results, flatten=True)
        for key, value in flat_results.items():
            if isinstance(value, (int, float)):
                value_dict[key].append(value)
    
    # Aggregate
    aggregated = {}
    for key, values in value_dict.items():
        if values:
            if aggregation == 'mean':
                aggregated[key] = np.mean(values)
            elif aggregation == 'median':
                aggregated[key] = np.median(values)
            elif aggregation == 'min':
                aggregated[key] = np.min(values)
            elif aggregation == 'max':
                aggregated[key] = np.max(values)
            elif aggregation == 'std':
                aggregated[key] = np.std(values)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return aggregated