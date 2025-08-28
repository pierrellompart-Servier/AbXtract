"""
File handling utilities for various formats.
"""

import json
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logging

logger = logging.getLogger(__name__)


class FileHandler:
    """
    Unified file handler for various formats.
    """
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize file handler.
        
        Parameters
        ----------
        temp_dir : Path, optional
            Custom temporary directory
        """
        self.temp_dir = temp_dir
        self.temp_files = []
    
    def read(
        self, 
        file_path: Union[str, Path], 
        format: Optional[str] = None
    ) -> Union[pd.DataFrame, dict, str]:
        """
        Read file based on format or extension.
        
        Parameters
        ----------
        file_path : str or Path
            Path to file
        format : str, optional
            File format (csv, json, fasta, etc.)
            
        Returns
        -------
        Data in appropriate format
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine format from extension if not provided
        if format is None:
            format = file_path.suffix.lower()[1:]  # Remove the dot
        
        if format in ['csv', 'tsv']:
            delimiter = '\t' if format == 'tsv' else ','
            return pd.read_csv(file_path, delimiter=delimiter)
        elif format == 'json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif format in ['fasta', 'fa']:
            return self._read_fasta(file_path)
        elif format == 'pdb':
            with open(file_path, 'r') as f:
                return f.read()
        elif format in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
        else:
            # Default to text file
            with open(file_path, 'r') as f:
                return f.read()
    
    def write(
        self,
        data: Any,
        file_path: Union[str, Path],
        format: Optional[str] = None
    ) -> Path:
        """
        Write data to file.
        
        Parameters
        ----------
        data : Any
            Data to write
        file_path : str or Path
            Output file path
        format : str, optional
            Output format
            
        Returns
        -------
        Path
            Path to written file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension if not provided
        if format is None:
            format = file_path.suffix.lower()[1:]
        
        if format in ['csv', 'tsv']:
            delimiter = '\t' if format == 'tsv' else ','
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False, sep=delimiter)
            else:
                pd.DataFrame(data).to_csv(file_path, index=False, sep=delimiter)
        elif format == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format in ['fasta', 'fa']:
            self._write_fasta(data, file_path)
        elif format in ['xlsx', 'xls']:
            if isinstance(data, pd.DataFrame):
                data.to_excel(file_path, index=False)
            else:
                pd.DataFrame(data).to_excel(file_path, index=False)
        else:
            # Default to text file
            with open(file_path, 'w') as f:
                f.write(str(data))
        
        logger.info(f"Wrote data to {file_path}")
        return file_path
    
    def _read_fasta(self, file_path: Path) -> Dict[str, str]:
        """Read FASTA file."""
        sequences = {}
        for record in SeqIO.parse(str(file_path), "fasta"):
            sequences[record.id] = str(record.seq)
        return sequences
    
    def _write_fasta(self, sequences: Dict[str, str], file_path: Path):
        """Write FASTA file."""
        records = []
        for seq_id, seq in sequences.items():
            record = SeqRecord(Seq(seq), id=seq_id, description="")
            records.append(record)
        SeqIO.write(records, str(file_path), "fasta")
    
    def create_temp_file(
        self, 
        suffix: str = "", 
        prefix: str = "tmp"
    ) -> Path:
        """
        Create temporary file.
        
        Parameters
        ----------
        suffix : str
            File suffix
        prefix : str
            File prefix
            
        Returns
        -------
        Path
            Path to temporary file
        """
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = tempfile.NamedTemporaryFile(
                suffix=suffix,
                prefix=prefix,
                dir=str(self.temp_dir),
                delete=False
            )
        else:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=suffix,
                prefix=prefix,
                delete=False
            )
        
        temp_path = Path(temp_file.name)
        temp_file.close()
        self.temp_files.append(temp_path)
        
        return temp_path
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
        
        self.temp_files = []
        
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Removed temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Could not remove temporary directory: {e}")


# Convenience functions
def read_fasta(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Read sequences from FASTA file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to FASTA file
        
    Returns
    -------
    dict
        Dictionary mapping sequence IDs to sequences
    """
    sequences = {}
    for record in SeqIO.parse(str(file_path), "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def write_fasta(
    sequences: Union[Dict[str, str], List[tuple[str, str]]],
    file_path: Union[str, Path]
) -> Path:
    """
    Write sequences to FASTA file.
    
    Parameters
    ----------
    sequences : dict or list
        Dictionary {id: sequence} or list of (id, sequence) tuples
    file_path : str or Path
        Output file path
        
    Returns
    -------
    Path
        Path to written file
    """
    file_path = Path(file_path)
    records = []
    
    if isinstance(sequences, dict):
        sequences = sequences.items()
    
    for seq_id, seq in sequences:
        record = SeqRecord(Seq(seq), id=seq_id, description="")
        records.append(record)
    
    SeqIO.write(records, str(file_path), "fasta")
    return file_path


def read_csv(
    file_path: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Read CSV file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to CSV file
    **kwargs
        Additional arguments for pandas.read_csv
        
    Returns
    -------
    pd.DataFrame
        DataFrame from CSV
    """
    return pd.read_csv(file_path, **kwargs)


def write_csv(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    **kwargs
) -> Path:
    """
    Write DataFrame to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write
    file_path : str or Path
        Output file path
    **kwargs
        Additional arguments for DataFrame.to_csv
        
    Returns
    -------
    Path
        Path to written file
    """
    file_path = Path(file_path)
    df.to_csv(file_path, index=False, **kwargs)
    return file_path


def read_json(file_path: Union[str, Path]) -> Dict:
    """
    Read JSON file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to JSON file
        
    Returns
    -------
    dict
        Parsed JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json(
    data: Dict,
    file_path: Union[str, Path],
    indent: int = 2
) -> Path:
    """
    Write data to JSON file.
    
    Parameters
    ----------
    data : dict
        Data to write
    file_path : str or Path
        Output file path
    indent : int
        Indentation level for pretty printing
        
    Returns
    -------
    Path
        Path to written file
    """
    file_path = Path(file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    return file_path


def create_temp_directory(prefix: str = "antibody_desc_") -> Path:
    """
    Create temporary directory.
    
    Parameters
    ----------
    prefix : str
        Directory prefix
        
    Returns
    -------
    Path
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return Path(temp_dir)


def cleanup_temp_files(paths: List[Union[str, Path]]):
    """
    Clean up temporary files and directories.
    
    Parameters
    ----------
    paths : list
        List of paths to remove
    """
    for path in paths:
        path = Path(path)
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                logger.debug(f"Removed: {path}")
        except Exception as e:
            logger.warning(f"Could not remove {path}: {e}")