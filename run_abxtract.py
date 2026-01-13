#!/usr/bin/env python3
"""
AbXtract CLI - Command Line Interface for Antibody Descriptor Calculations

This script provides a CLI for running AbXtract antibody descriptor calculations
with support for parallelization, predefined computation modes, and comprehensive
output generation.

Features:
- Supports both VH+VL antibodies and VHH/nanobodies (leave VL column empty)
- Parallel processing using N-2 CPU cores (configurable)
- Predefined computation modes: b(asic), r(egular), mr, mw(ide), wd(eep)
- Progress tracking with tqdm
- Comprehensive logging with JSON run summary

VHH/Nanobody Mode:
- Automatically detected when sequence_VL column is empty/None
- Sets is_fv=False in config
- Note: calculate_proper is disabled for VHH due to library limitations

Usage:
    python run_abxtract.py -i input.csv -o output_folder -m r
    python run_abxtract.py --input data.csv --output results/ --mode wd --pH 7.0

Author: AbXtract Team
"""

import argparse
import sys
import os
import multiprocessing
import logging
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm

# =============================================================================
# Mode Configurations - Predefined parameter sets
# =============================================================================

MODE_CONFIGS = {
    "b": {  # Basic - fast, minimal calculations
        "calculate_liabilities": False,
        "calculate_bashour": True,
        "calculate_peptide": False,
        "calculate_protpy": False,
        "calculate_sasa": False,
        "calculate_charge": True,
        "calculate_dssp": False,
        "calculate_propka": False,
        "calculate_arpeggio": False,
        "calculate_cdr_properties": True,
        "calculate_proper": False,
    },
    "r": {  # Regular - balanced (DEFAULT)
        "calculate_liabilities": True,
        "calculate_bashour": True,
        "calculate_peptide": False,
        "calculate_protpy": False,
        "calculate_sasa": True,
        "calculate_charge": True,
        "calculate_dssp": True,
        "calculate_propka": True,
        "calculate_arpeggio": True,
        "calculate_cdr_properties": True,
        "calculate_proper": True,
    },
    "mr": {  # Medium Regular - same as r
        "calculate_liabilities": True,
        "calculate_bashour": True,
        "calculate_peptide": False,
        "calculate_protpy": False,
        "calculate_sasa": True,
        "calculate_charge": True,
        "calculate_dssp": True,
        "calculate_propka": True,
        "calculate_arpeggio": True,
        "calculate_cdr_properties": True,
        "calculate_proper": True,
    },
    "mw": {  # Medium Wide - includes protpy
        "calculate_liabilities": True,
        "calculate_bashour": True,
        "calculate_peptide": False,
        "calculate_protpy": True,
        "calculate_sasa": True,
        "calculate_charge": True,
        "calculate_dssp": True,
        "calculate_propka": True,
        "calculate_arpeggio": True,
        "calculate_cdr_properties": True,
        "calculate_proper": True,
    },
    "wd": {  # Wide/Deep - all calculations
        "calculate_liabilities": True,
        "calculate_bashour": True,
        "calculate_peptide": True,
        "calculate_protpy": True,
        "calculate_sasa": True,
        "calculate_charge": True,
        "calculate_dssp": True,
        "calculate_propka": True,
        "calculate_arpeggio": True,
        "calculate_cdr_properties": True,
        "calculate_proper": True,
    },
}

# Valid options for CLI parameters
VALID_NUMBERING_SCHEMES = ["imgt", "kabat", "chothia"]
VALID_CDR_DEFINITIONS = ["imgt", "kabat", "chothia", "north", "contact"]
VALID_HYDROPHOBICITY_SCALES = ["Eisenberg", "KyteDoolittle", "Hopp-Woods", "Cornette", "Rose", "Janin"]


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: Path, verbose: bool = True) -> logging.Logger:
    """Setup logging to both file and console."""
    log_file = output_dir / f"abxtract_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger("AbXtract")
    # logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # File handler - captures everything
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ProcessingResult:
    """Container for processing results of a single antibody."""
    id: str
    success: bool
    df_final: Optional[pd.DataFrame] = None
    df_heavy: Optional[pd.DataFrame] = None
    df_light: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class RunParameters:
    """Container for all run parameters for logging."""
    input_file: str
    output_dir: str
    mode: str
    numbering_scheme: str
    cdr_definition: str
    pH: float
    temperature: float
    hydrophobicity_scale: str
    n_jobs: int
    n_cpus_available: int
    n_antibodies: int
    timestamp: str
    
    # Mode-specific calculation flags
    calculate_liabilities: bool = True
    calculate_bashour: bool = True
    calculate_peptide: bool = False
    calculate_protpy: bool = False
    calculate_sasa: bool = True
    calculate_charge: bool = True
    calculate_dssp: bool = True
    calculate_propka: bool = True
    calculate_arpeggio: bool = True
    calculate_cdr_properties: bool = True
    calculate_proper: bool = True


# =============================================================================
# Core Processing Functions
# =============================================================================

def process_single_antibody(
    row_data: Dict[str, Any],
    config_dict: Dict[str, Any],
    abxtract_path: str,
    base_dir: str = None
) -> ProcessingResult:
    """
    Process a single antibody entry.
    
    This function is designed to be called in parallel workers.
    
    Parameters
    ----------
    row_data : dict
        Dictionary containing 'id', 'vh_sequence', 'vl_sequence', 'pdb_path'
    config_dict : dict
        Configuration dictionary for AbXtract
    abxtract_path : str
        Path to AbXtract package
    base_dir : str, optional
        Base directory for resolving relative PDB paths
        
    Returns
    -------
    ProcessingResult
        Result container with dataframes or error information
    """
    import time
    import os
    import warnings
    
    # Suppress Biopython deprecation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    start_time = time.time()
    
    ab_id = row_data.get('id', 'Unknown')
    heavy_seq = row_data.get('vh_sequence')
    light_seq = row_data.get('vl_sequence')
    pdb_path = row_data.get('pdb_path')
    
    # Change to base_dir for external tools that use relative paths
    original_cwd = os.getcwd()
    if base_dir:
        os.chdir(base_dir)
    
    try:
        # Add AbXtract to path
        if abxtract_path not in sys.path:
            sys.path.insert(0, abxtract_path)
        
        # Import AbXtract modules
        from AbXtract import AntibodyDescriptorCalculator, Config
        from AbXtract.sequence import (
            AntibodyNumbering,
            PeptideDescriptorCalculator
        )
        from AbXtract.utils import validate_sequence, analysis_descriptors
        
        # Handle None/NaN values for light sequence (VHH mode)
        if pd.isna(light_seq) or light_seq == "" or light_seq is None or str(light_seq).lower() == 'none':
            light_seq = None
        
        # Resolve PDB path to absolute path
        if pdb_path:
            pdb_path = str(pdb_path)
            if not os.path.isabs(pdb_path):
                # Try resolving relative to base_dir first
                if base_dir:
                    candidate = os.path.join(base_dir, pdb_path)
                    if os.path.exists(candidate):
                        pdb_path = os.path.abspath(candidate)
                    else:
                        # Try relative to current working directory
                        pdb_path = os.path.abspath(pdb_path)
                else:
                    pdb_path = os.path.abspath(pdb_path)
        
        # Check if PDB file exists
        if not pdb_path or not os.path.exists(pdb_path):
            return ProcessingResult(
                id=ab_id,
                success=False,
                error_message=f"PDB file not found: {pdb_path}",
                processing_time=time.time() - start_time
            )
        
        # Validate sequences
        heavy_valid, heavy_msg = validate_sequence(heavy_seq) if heavy_seq else (False, "No heavy sequence")
        light_valid, light_msg = validate_sequence(light_seq) if light_seq else (True, "No light sequence (VHH mode)")
        
        if not heavy_valid:
            return ProcessingResult(
                id=ab_id,
                success=False,
                error_message=f"Invalid heavy sequence: {heavy_msg}",
                processing_time=time.time() - start_time
            )
        
        # Update config for is_fv based on light sequence presence
        # is_fv should be True for Fv (VH+VL) and False for VHH (VH only)
        config_dict['is_fv'] = light_seq is not None
        
        # Disable calculate_proper for VHH mode as it has issues with None light sequence
        if light_seq is None and config_dict.get('calculate_proper', False):
            config_dict['calculate_proper'] = False
        
        # Initialize components
        config = Config.from_dict(config_dict)
        calc = AntibodyDescriptorCalculator(config=config)
        numbering = AntibodyNumbering(scheme=config.numbering_scheme)
        peptide_calc = PeptideDescriptorCalculator()
        
        # Perform numbering and get CDRs
        heavy_numbered = numbering.number_sequence(heavy_seq, 'H')
        annotated_H, cdrs_H = numbering.get_cdr_sequences(heavy_numbered, 'H')
        heavy_profiles = numbering.get_peptide_profiles(heavy_seq)
        
        annotated_L, cdrs_L = None, None
        light_profiles = None
        if light_seq:
            light_numbered = numbering.number_sequence(light_seq, 'L')
            annotated_L, cdrs_L = numbering.get_cdr_sequences(light_numbered, 'L')
            light_profiles = numbering.get_peptide_profiles(light_seq)
        
        # Calculate peptide descriptors
        peptide_results = peptide_calc.calculate_all(
            heavy_sequence=heavy_seq,
            light_sequence=light_seq
        )
        
        # Calculate sequence descriptors
        sequence_results, liabilities = calc.calculate_sequence_descriptors(
            heavy_sequence=heavy_seq,
            light_sequence=light_seq,
            sequence_id=ab_id
        )
        
        # Calculate structure descriptors
        structure_results_seq, structure_results_comp, df_residues, df_AA, df_Ab = calc.calculate_structure_descriptors(
            heavy_sequence=heavy_seq,
            light_sequence=light_seq,
            pdb_file=pdb_path,
            structure_id=ab_id
        )
        
        # Add residue SASA sum column
        structure_results_seq = analysis_descriptors.add_residue_sasa_sum_column(structure_results_seq)
        
        # Get data for residue-level dataframes
        liabilities_list = liabilities['liabilities'].iloc[0] if not liabilities.empty else []
        structures_data = structure_results_seq.iloc[0] if not structure_results_seq.empty else {}
        
        # Create heavy chain dataframe
        df_heavy_final = None
        if heavy_seq:
            df_heavy_final = analysis_descriptors.create_complete_antibody_dataframe(
                0, df_residues, df_Ab,
                heavy_seq, annotated_H, heavy_profiles,
                structures_data, liabilities_list, 'Heavy', config.numbering_scheme
            )
        
        # Create light chain dataframe
        df_light_final = None
        if light_seq:
            df_light_final = analysis_descriptors.create_complete_antibody_dataframe(
                len(heavy_seq), df_residues, df_Ab,
                light_seq, annotated_L, light_profiles,
                structures_data, liabilities_list, 'Light', config.numbering_scheme
            )
        
        # Combine all results
        df_final = analysis_descriptors.combine_all_results(
            df_AA,
            structure_results_comp,
            sequence_results,
            peptide_results,
            heavy_valid=heavy_valid,
            light_valid=light_valid if light_seq else True,
            cdrs_H=cdrs_H,
            cdrs_L=cdrs_L
        )
        
        # Add identifier
        df_final['Identifier'] = ab_id
        if df_heavy_final is not None:
            df_heavy_final['Identifier'] = ab_id
        if df_light_final is not None:
            df_light_final['Identifier'] = ab_id
        
        return ProcessingResult(
            id=ab_id,
            success=True,
            df_final=df_final,
            df_heavy=df_heavy_final,
            df_light=df_light_final,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        return ProcessingResult(
            id=ab_id,
            success=False,
            error_message=f"{str(e)}\n{traceback.format_exc()}",
            processing_time=time.time() - start_time
        )
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def process_antibody_batch(
    input_df: pd.DataFrame,
    config_dict: Dict[str, Any],
    abxtract_path: str,
    n_jobs: int,
    logger: logging.Logger,
    base_dir: str = None
) -> Tuple[List[ProcessingResult], List[str]]:
    """
    Process all antibodies in parallel.
    
    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe with antibody data
    config_dict : dict
        Configuration dictionary
    abxtract_path : str
        Path to AbXtract package
    n_jobs : int
        Number of parallel jobs
    logger : logging.Logger
        Logger instance
    base_dir : str, optional
        Base directory for resolving relative PDB paths
        
    Returns
    -------
    Tuple[List[ProcessingResult], List[str]]
        List of results and list of failed IDs
    """
    results = []
    failed_ids = []
    
    # Prepare row data for parallel processing
    row_data_list = []
    for _, row in input_df.iterrows():
        row_data_list.append({
            'id': row.get('ID', row.get('id', 'Unknown')),
            'vh_sequence': row.get('sequence_VH', row.get('VH', row.get('heavy_sequence'))),
            'vl_sequence': row.get('sequence_VL', row.get('VL', row.get('light_sequence'))),
            'pdb_path': row.get('pdb_path', row.get('PDB', row.get('structure_path')))
        })
    
    n_antibodies = len(row_data_list)
    # logger.info(f"Processing {n_antibodies} antibodies using {n_jobs} parallel jobs")
    
    if n_jobs == 1:
        # Sequential processing with progress bar
        for row_data in tqdm(row_data_list, desc="Processing antibodies", unit="ab"):
            result = process_single_antibody(row_data, config_dict.copy(), abxtract_path, base_dir)
            results.append(result)
            if not result.success:
                failed_ids.append(result.id)
                logger.warning(f"Failed to process {result.id}: {result.error_message}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_id = {
                executor.submit(
                    process_single_antibody,
                    row_data,
                    config_dict.copy(),
                    abxtract_path,
                    base_dir
                ): row_data['id']
                for row_data in row_data_list
            }
            
            # Collect results with progress bar
            with tqdm(total=n_antibodies, desc="Processing antibodies", unit="ab") as pbar:
                for future in as_completed(future_to_id):
                    ab_id = future_to_id[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if not result.success:
                            failed_ids.append(result.id)
                            logger.warning(f"Failed to process {result.id}: {result.error_message}")
                    except Exception as e:
                        failed_ids.append(ab_id)
                        logger.error(f"Exception processing {ab_id}: {str(e)}")
                        results.append(ProcessingResult(
                            id=ab_id,
                            success=False,
                            error_message=str(e)
                        ))
                    pbar.update(1)
    
    return results, failed_ids


# =============================================================================
# Output Functions
# =============================================================================

def aggregate_results(
    results: List[ProcessingResult],
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate all processing results into final dataframes.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        df_mod, df_fin_heavy, df_fin_light, df_test
    """
    df_finals = []
    df_heavys = []
    df_lights = []
    
    for result in results:
        if result.success:
            if result.df_final is not None:
                # Remove duplicate columns from individual dataframe
                df = result.df_final.copy()
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
                df_finals.append(df)
            if result.df_heavy is not None:
                df = result.df_heavy.copy()
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
                df_heavys.append(df)
            if result.df_light is not None:
                df = result.df_light.copy()
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
                df_lights.append(df)
    
    # Concatenate results with outer join to handle different column structures
    # (VHH vs Fv antibodies have different columns)
    df_test = pd.DataFrame()
    if df_finals:
        try:
            # First try standard concat
            df_test = pd.concat(df_finals, axis=0, ignore_index=True)
        except Exception as e:
            logger.warning(f"Standard concat failed, trying with outer join: {e}")
            try:
                # Use outer join for different column structures
                df_test = pd.concat(df_finals, axis=0, ignore_index=True, join='outer')
            except Exception as e2:
                logger.warning(f"Outer concat also failed, trying row-by-row: {e2}")
                # Last resort: build dataframe row by row
                all_columns = set()
                for df in df_finals:
                    all_columns.update(df.columns.tolist())
                all_columns = sorted(list(all_columns))
                
                rows = []
                for df in df_finals:
                    for _, row in df.iterrows():
                        row_dict = {col: row.get(col, np.nan) for col in all_columns}
                        rows.append(row_dict)
                df_test = pd.DataFrame(rows)
    
    df_fin_heavy = pd.DataFrame()
    if df_heavys:
        try:
            df_fin_heavy = pd.concat(df_heavys, axis=0, ignore_index=True)
        except Exception as e:
            logger.warning(f"Heavy concat failed: {e}")
            try:
                df_fin_heavy = pd.concat(df_heavys, axis=0, ignore_index=True, join='outer')
            except:
                pass
    
    df_fin_light = pd.DataFrame()
    if df_lights:
        try:
            df_fin_light = pd.concat(df_lights, axis=0, ignore_index=True)
        except Exception as e:
            logger.warning(f"Light concat failed: {e}")
            try:
                df_fin_light = pd.concat(df_lights, axis=0, ignore_index=True, join='outer')
            except:
                pass
    
    # Prepare df_mod (processed version of df_test)
    df_mod = pd.DataFrame()
    if not df_test.empty:
        try:
            # Import analysis_descriptors for prepare_object_descriptors
            sys.path.insert(0, os.environ.get('ABXTRACT_PATH', '.'))
            from AbXtract.utils import analysis_descriptors
            df_mod = analysis_descriptors.prepare_object_descriptors(df_test)
            
            # Remove duplicate columns
            df_mod = df_mod.loc[:, ~df_mod.columns.duplicated(keep='first')]
        except Exception as e:
            logger.warning(f"Could not prepare df_mod: {e}")
            df_mod = df_test.copy()
            # Still remove duplicates
            df_mod = df_mod.loc[:, ~df_mod.columns.duplicated(keep='first')]
    
    return df_mod, df_fin_heavy, df_fin_light, df_test


def save_results(
    df_mod: pd.DataFrame,
    df_fin_heavy: pd.DataFrame,
    df_fin_light: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, str]:
    """Save all result dataframes to output directory."""
    output_files = {}
    
    # Save df_mod (standard full descriptors set)
    if not df_mod.empty:
        output_path = output_dir / "descriptors_full.csv"
        df_mod.to_csv(output_path, index=False)
        output_files['descriptors_full'] = str(output_path)
        # logger.info(f"Saved descriptors_full.csv ({df_mod.shape[0]} rows, {df_mod.shape[1]} columns)")
    
    # Save df_fin_heavy (heavy chain residue descriptors)
    if not df_fin_heavy.empty:
        output_path = output_dir / "residues_heavy.csv"
        df_fin_heavy.to_csv(output_path, index=False)
        output_files['residues_heavy'] = str(output_path)
        # logger.info(f"Saved residues_heavy.csv ({df_fin_heavy.shape[0]} rows)")
    
    # Save df_fin_light (light chain residue descriptors)
    if not df_fin_light.empty:
        output_path = output_dir / "residues_light.csv"
        df_fin_light.to_csv(output_path, index=False)
        output_files['residues_light'] = str(output_path)
        # logger.info(f"Saved residues_light.csv ({df_fin_light.shape[0]} rows)")
    
    # Save df_test (pH dependent descriptors / raw combined)
    if not df_test.empty:
        output_path = output_dir / "descriptors_raw.csv"
        df_test.to_csv(output_path, index=False)
        output_files['descriptors_raw'] = str(output_path)
        # logger.info(f"Saved descriptors_raw.csv ({df_test.shape[0]} rows)")
    
    return output_files


def save_run_log(
    params: RunParameters,
    output_files: Dict[str, str],
    failed_ids: List[str],
    total_time: float,
    output_dir: Path,
    logger: logging.Logger
):
    """Save comprehensive run log as JSON."""
    log_data = {
        "run_parameters": asdict(params),
        "output_files": output_files,
        "execution_summary": {
            "total_time_seconds": round(total_time, 2),
            "n_successful": params.n_antibodies - len(failed_ids),
            "n_failed": len(failed_ids),
            "failed_ids": failed_ids
        },
        "mode_settings": MODE_CONFIGS.get(params.mode, {}),
    }
    
    log_path = output_dir / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    # logger.info(f"Run log saved to {log_path}")


# =============================================================================
# CLI Argument Parser
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="""
AbXtract CLI - Antibody Descriptor Calculator

Calculate comprehensive structural and sequence-based descriptors for antibodies
and nanobodies (VHH) from PDB structures.

Input CSV format:
    ID,sequence_VH,sequence_VL,pdb_path
    - ID: Unique identifier for the antibody
    - sequence_VH: Heavy chain variable region sequence
    - sequence_VL: Light chain variable region sequence (leave empty or 'None' for VHH/nanobody)
    - pdb_path: Path to PDB structure file

Computation Modes:
    b  : Basic - fast, minimal calculations (Bashour, charge, CDR properties)
    r  : Regular - balanced (DEFAULT) - most analyses except slow ones
    mr : Medium Regular - same as 'r'
    mw : Medium Wide - adds ProtPy descriptors
    wd : Wide/Deep - all calculations including peptide and ProtPy (slowest)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run with default mode (r)
    python run_abxtract.py -i antibodies.csv -o results/

    # Run with wide/deep mode
    python run_abxtract.py -i antibodies.csv -o results/ -m wd

    # Custom pH and numbering scheme
    python run_abxtract.py -i antibodies.csv -o results/ -m r -p 6.5 --numbering-scheme kabat

    # Force specific number of parallel jobs
    python run_abxtract.py -i antibodies.csv -o results/ --n-jobs 8
    
    # Specify base directory for relative PDB paths
    python run_abxtract.py -i antibodies.csv -o results/ --base-dir /path/to/project
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input CSV file with antibody data'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to output directory for results'
    )
    
    # Mode selection
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['b', 'r', 'mr', 'mw', 'wd'],
        default='r',
        help='Computation mode: b(asic), r(egular), mr, mw(ide), wd(eep). Default: r'
    )
    
    # Numbering and CDR options
    parser.add_argument(
        '--numbering-scheme', '-ns',
        type=str,
        choices=VALID_NUMBERING_SCHEMES,
        default='imgt',
        help='Antibody numbering scheme. Default: imgt'
    )
    parser.add_argument(
        '--cdr-definition', '-cd',
        type=str,
        choices=VALID_CDR_DEFINITIONS,
        default=None,
        help='CDR definition scheme. Default: same as numbering scheme'
    )
    
    # Physical parameters
    parser.add_argument(
        '-p', '--pH',
        type=float,
        default=7.4,
        help='pH value for charge and pKa calculations. Default: 7.4'
    )
    parser.add_argument(
        '-t', '--temperature',
        type=float,
        default=25.0,
        help='Temperature in Celsius. Default: 25.0'
    )
    parser.add_argument(
        '-hs', '--hydrophobicity-scale',
        type=str,
        choices=VALID_HYDROPHOBICITY_SCALES,
        default='Eisenberg',
        help='Hydrophobicity scale to use. Default: Eisenberg'
    )
    
    # Parallelization
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Number of parallel jobs. Default: N_CPUs - 2 (auto-detected)'
    )
    
    # Additional options
    parser.add_argument(
        '--abxtract-path',
        type=str,
        default=None,
        help='Path to AbXtract package. Default: auto-detect or parent of this script'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory for resolving relative PDB paths. Default: directory of input CSV'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output. Default: True'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        default=False,
        help='Suppress verbose output'
    )
    
    return parser


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the CLI."""
    import time
    start_time = time.time()
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    verbose = args.verbose and not args.quiet
    logger = setup_logging(output_dir, verbose)
    # logger.info("=" * 60)
    # logger.info("AbXtract CLI - Antibody Descriptor Calculator")
    # logger.info("=" * 60)
    
    # Determine AbXtract path
    abxtract_path = args.abxtract_path
    if abxtract_path is None:
        # Try to auto-detect
        script_dir = Path(__file__).parent.absolute()
        potential_paths = [
            script_dir,
            script_dir.parent,
            Path.home() / "github" / "AbXtract",
            Path("/home/HX46_FR5/github/AbXtract"),
        ]
        for path in potential_paths:
            if (path / "AbXtract" / "__init__.py").exists():
                abxtract_path = str(path)
                break
        if abxtract_path is None:
            logger.error("Could not auto-detect AbXtract path. Please specify with --abxtract-path")
            sys.exit(1)
    
    # logger.info(f"Using AbXtract from: {abxtract_path}")
    os.environ['ABXTRACT_PATH'] = abxtract_path
    sys.path.insert(0, abxtract_path)
    
    # Load input data
    try:
        input_df = pd.read_csv(input_path)
        # logger.info(f"Loaded input file: {input_path} ({len(input_df)} rows)")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['ID', 'sequence_VH', 'pdb_path']
    alt_cols = {
        'ID': ['id', 'Identifier'],
        'sequence_VH': ['VH', 'heavy_sequence', 'Heavy'],
        'pdb_path': ['PDB', 'structure_path', 'pdb']
    }
    
    # Map alternative column names
    for col, alternatives in alt_cols.items():
        if col not in input_df.columns:
            for alt in alternatives:
                if alt in input_df.columns:
                    input_df[col] = input_df[alt]
                    break
    
    # Check columns
    missing_cols = [col for col in required_cols if col not in input_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {list(input_df.columns)}")
        sys.exit(1)
    
    # Determine number of parallel jobs
    n_cpus = multiprocessing.cpu_count()
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    else:
        n_jobs = max(1, n_cpus - 2)
    
    # Calculate jobs per batch
    n_antibodies = len(input_df)
    jobs_per_core = n_antibodies / n_jobs if n_jobs > 0 else n_antibodies
    
    # logger.info(f"CPU Configuration: {n_cpus} available, using {n_jobs} jobs")
    # logger.info(f"Distribution: ~{jobs_per_core:.1f} antibodies per core")
    
    # Get mode configuration
    mode_config = MODE_CONFIGS.get(args.mode, MODE_CONFIGS['r'])
    # logger.info(f"Using computation mode: {args.mode}")
    
    # CDR definition defaults to numbering scheme if not specified
    cdr_definition = args.cdr_definition if args.cdr_definition else args.numbering_scheme
    
    # Build configuration dictionary
    config_dict = {
        'numbering_scheme': args.numbering_scheme,
        'cdr_definition': cdr_definition,
        'pH': args.pH,
        'temperature': args.temperature,
        'hydrophobicity_scale': args.hydrophobicity_scale,
        'n_jobs': 1,  # Each worker processes one antibody
        'verbose': False,  # Suppress per-antibody verbosity in parallel
        'temp_dir': str(output_dir.absolute() / 'temp'),  # Use absolute path
        **mode_config
    }
    
    # Log configuration
    # logger.info(f"Configuration:")
    # logger.info(f"  - Numbering scheme: {args.numbering_scheme}")
    # logger.info(f"  - CDR definition: {cdr_definition}")
    # logger.info(f"  - pH: {args.pH}")
    # logger.info(f"  - Temperature: {args.temperature}°C")
    # logger.info(f"  - Hydrophobicity scale: {args.hydrophobicity_scale}")
    # logger.info(f"  - Mode calculations: {mode_config}")
    
    # Create run parameters for logging
    run_params = RunParameters(
        input_file=str(input_path),
        output_dir=str(output_dir),
        mode=args.mode,
        numbering_scheme=args.numbering_scheme,
        cdr_definition=cdr_definition,
        pH=args.pH,
        temperature=args.temperature,
        hydrophobicity_scale=args.hydrophobicity_scale,
        n_jobs=n_jobs,
        n_cpus_available=n_cpus,
        n_antibodies=n_antibodies,
        timestamp=datetime.now().isoformat(),
        **mode_config
    )
    
    # Process antibodies
    # logger.info("-" * 60)
    # logger.info("Starting antibody processing...")
    
    # Get base directory for resolving relative PDB paths
    # Priority: --base-dir argument > directory containing input CSV > cwd
    if args.base_dir:
        base_dir = str(Path(args.base_dir).absolute())
    else:
        base_dir = str(Path(input_path).parent.absolute())
        # If input is in current directory, use cwd
        if base_dir == '.' or base_dir == '':
            base_dir = os.getcwd()
    
    # logger.info(f"Base directory for PDB path resolution: {base_dir}")
    
    # Change to base directory for external tools that use relative paths
    original_cwd = os.getcwd()
    os.chdir(base_dir)
    # logger.info(f"Changed working directory to: {base_dir}")
    
    try:
        results, failed_ids = process_antibody_batch(
            input_df,
            config_dict,
            abxtract_path,
            n_jobs,
            logger,
            base_dir=base_dir
        )
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    # Aggregate results
    # logger.info("-" * 60)
    # logger.info("Aggregating results...")
    df_mod, df_fin_heavy, df_fin_light, df_test = aggregate_results(results, logger)
    
    # Save results
    # logger.info("Saving output files...")
    output_files = save_results(
        df_mod, df_fin_heavy, df_fin_light, df_test,
        output_dir, logger
    )
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Save run log
    save_run_log(run_params, output_files, failed_ids, total_time, output_dir, logger)
    
    # Final summary
    # logger.info("=" * 60)
    # logger.info("PROCESSING COMPLETE")
    # logger.info("=" * 60)
    # logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    # logger.info(f"Successful: {n_antibodies - len(failed_ids)}/{n_antibodies}")
    if failed_ids:
        logger.warning(f"Failed IDs: {failed_ids}")
    # logger.info(f"Output directory: {output_dir}")
    # logger.info("Output files:")
    # for name, path in output_files.items():
    #     logger.info(f"  - {name}: {path}")
    
    # Exit with appropriate code
    if len(failed_ids) == n_antibodies:
        sys.exit(1)  # All failed
    elif failed_ids:
        sys.exit(0)  # Partial success
    else:
        sys.exit(0)  # Full success


if __name__ == "__main__":
    main()
