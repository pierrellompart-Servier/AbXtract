"""
Command-line interface for antibody descriptor calculations.

This module provides CLI access to all antibody descriptor functionality,
supporting both single and batch processing of sequences and structures.
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import logging
from typing import Optional, List, Dict, Any
import warnings
from datetime import datetime
import csv

from .core import AntibodyDescriptorCalculator, Config, load_config
from .__version__ import __version__

# Setup logging
def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure logging based on verbosity settings."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger(__name__)


# Main CLI group
@click.group()
@click.version_option(version=__version__, prog_name='antibody-descriptors')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def cli(ctx, verbose, quiet):
    """
    Antibody Descriptor Calculator - Comprehensive antibody analysis toolkit.
    
    Calculate sequence and structure-based descriptors for antibodies and nanobodies,
    including developability metrics, physicochemical properties, and structural features.
    
    Examples:
    
        # Calculate all descriptors for a sequence
        antibody-descriptors calculate -h QVQLVQSGAEVKKPGA... -o results.csv
        
        # Analyze a PDB structure
        antibody-descriptors calculate -p antibody.pdb --structure-only
        
        # Batch process multiple sequences
        antibody-descriptors batch sequences.fasta -o batch_results.csv
        
        # List available descriptors
        antibody-descriptors list
    """
    setup_logging(verbose, quiet)
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


# Main calculation command
@cli.command()
@click.option('--heavy', '-h', help='Heavy chain sequence or FASTA file')
@click.option('--light', '-l', help='Light chain sequence or FASTA file')
@click.option('--pdb', '-p', type=click.Path(exists=True), help='PDB structure file')
@click.option('--output', '-o', default='descriptors.csv', help='Output file path')
@click.option('--format', '-f', 
              type=click.Choice(['csv', 'json', 'excel', 'parquet', 'pickle']),
              default='csv', help='Output format')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration JSON file')
@click.option('--sequence-only', is_flag=True, 
              help='Calculate only sequence descriptors')
@click.option('--structure-only', is_flag=True, 
              help='Calculate only structure descriptors')
@click.option('--sample-id', default='Sample', help='Sample identifier')
@click.option('--numbering-scheme', 
              type=click.Choice(['imgt', 'kabat', 'chothia', 'martin']),
              help='Antibody numbering scheme')
@click.option('--ph', type=float, help='pH for calculations (default: 7.0)')
@click.option('--no-validate', is_flag=True, 
              help='Skip sequence/structure validation')
@click.pass_context
def calculate(ctx, heavy, light, pdb, output, format, config, sequence_only,
              structure_only, sample_id, numbering_scheme, ph, no_validate):
    """
    Calculate antibody descriptors for sequences and/or structures.
    
    Supports direct sequence input, FASTA files, and PDB structures.
    Can calculate sequence-only, structure-only, or combined descriptors.
    """
    try:
        # Load configuration
        config_dict = {}
        if config:
            with open(config, 'r') as f:
                config_dict = json.load(f)
        
        # Override specific config parameters if provided
        if numbering_scheme:
            config_dict['numbering_scheme'] = numbering_scheme
        if ph is not None:
            config_dict['pH'] = ph
        if ctx.obj['verbose']:
            config_dict['verbose'] = True
        
        # Initialize calculator
        calc = AntibodyDescriptorCalculator(config=config_dict)
        
        # Parse sequences
        heavy_seq = _parse_sequence_input(heavy) if heavy else None
        light_seq = _parse_sequence_input(light) if light else None
        
        # Validate inputs
        if not (heavy_seq or light_seq or pdb):
            raise click.ClickException(
                "At least one input required (heavy, light, or pdb)"
            )
        
        if sequence_only and not (heavy_seq or light_seq):
            raise click.ClickException(
                "Sequence input required for sequence-only mode"
            )
        
        if structure_only and not pdb:
            raise click.ClickException(
                "PDB input required for structure-only mode"
            )
        
        # Calculate descriptors
        click.echo(f"Calculating descriptors for {sample_id}...")
        
        if structure_only:
            results = calc.calculate_structure_descriptors(
                pdb, 
                structure_id=sample_id,
                preprocess=not no_validate
            )
        elif sequence_only:
            results = calc.calculate_sequence_descriptors(
                heavy_seq, light_seq,
                sequence_id=sample_id
            )
        else:
            results = calc.calculate_all(
                heavy_seq, light_seq, pdb,
                sample_id=sample_id,
                validate_consistency=not no_validate
            )
        
        # Save results
        _save_results(results, output, format)
        
        # Print summary
        if not ctx.obj['quiet']:
            summary = calc.get_summary()
            click.echo(f"\nSummary:")
            click.echo(f"  Descriptors calculated: {summary['n_descriptors_calculated']}")
            click.echo(f"  Analyses performed: {', '.join(summary['analyses_performed'])}")
            if 'total_liabilities' in summary:
                click.echo(f"  Total liabilities: {summary['total_liabilities']}")
            if 'developability_index' in summary:
                click.echo(f"  Developability index: {summary['developability_index']:.3f}")
            
            click.secho(f"\n✓ Results saved to {output}", fg='green')
    
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        if ctx.obj['verbose']:
            raise
        else:
            raise click.ClickException(str(e))


# Batch processing command
@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='batch_results.csv', 
              help='Output file path')
@click.option('--format', '-f',
              type=click.Choice(['csv', 'excel', 'json']),
              default='csv', help='Output format')
@click.option('--input-format', 
              type=click.Choice(['fasta', 'csv', 'json', 'excel']),
              help='Input file format (auto-detected if not specified)')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration JSON file')
@click.option('--parallel', '-j', type=int, default=1,
              help='Number of parallel jobs')
@click.option('--heavy-column', default='Heavy_Sequence',
              help='Column name for heavy sequences (CSV/Excel input)')
@click.option('--light-column', default='Light_Sequence',
              help='Column name for light sequences (CSV/Excel input)')
@click.option('--pdb-column', default='PDB_File',
              help='Column name for PDB files (CSV/Excel input)')
@click.option('--id-column', default='ID',
              help='Column name for sample IDs (CSV/Excel input)')
@click.option('--progress/--no-progress', default=True,
              help='Show progress bar')
@click.pass_context
def batch(ctx, input_file, output, format, input_format, config, parallel,
          heavy_column, light_column, pdb_column, id_column, progress):
    """
    Batch process multiple antibody sequences or structures.
    
    Supports FASTA files, CSV/Excel files with sequences, or directories
    containing multiple PDB files.
    
    Examples:
    
        # Process FASTA file with multiple sequences
        antibody-descriptors batch sequences.fasta -o results.csv
        
        # Process CSV file with heavy and light chains
        antibody-descriptors batch antibodies.csv -o results.xlsx
        
        # Process with parallel jobs
        antibody-descriptors batch sequences.fasta -j 4 -o results.csv
    """
    try:
        # Load configuration
        config_dict = {'n_jobs': parallel}
        if config:
            with open(config, 'r') as f:
                config_dict.update(json.load(f))
        
        # Auto-detect input format if not specified
        if not input_format:
            input_format = _detect_file_format(input_file)
        
        # Load input data
        samples = _load_batch_input(
            input_file, input_format,
            heavy_column, light_column, pdb_column, id_column
        )
        
        if not samples:
            raise click.ClickException("No valid samples found in input file")
        
        click.echo(f"Processing {len(samples)} samples...")
        
        # Initialize calculator
        calc = AntibodyDescriptorCalculator(config=config_dict)
        
        # Process samples
        results_list = []
        
        if progress:
            import tqdm
            iterator = tqdm.tqdm(samples, desc="Processing")
        else:
            iterator = samples
        
        for sample in iterator:
            try:
                sample_results = calc.calculate_all(
                    heavy_sequence=sample.get('heavy'),
                    light_sequence=sample.get('light'),
                    pdb_file=sample.get('pdb'),
                    sample_id=sample['id']
                )
                results_list.append(sample_results)
            except Exception as e:
                logger.error(f"Failed to process {sample['id']}: {e}")
                if ctx.obj['verbose']:
                    raise
                # Add failed sample with error message
                error_result = pd.DataFrame([{
                    'SampleID': sample['id'],
                    'Error': str(e)
                }])
                results_list.append(error_result)
        
        # Combine results
        if results_list:
            combined_results = pd.concat(results_list, ignore_index=True)
            
            # Save results
            _save_results(combined_results, output, format)
            
            # Print summary
            if not ctx.obj['quiet']:
                successful = combined_results['Error'].isna().sum() if 'Error' in combined_results.columns else len(combined_results)
                failed = len(combined_results) - successful
                
                click.echo(f"\nBatch processing complete:")
                click.echo(f"  Successful: {successful}")
                if failed > 0:
                    click.echo(f"  Failed: {failed}")
                click.secho(f"\n✓ Results saved to {output}", fg='green')
        else:
            raise click.ClickException("No results generated")
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        if ctx.obj['verbose']:
            raise
        else:
            raise click.ClickException(str(e))


# List descriptors command
@cli.command(name='list')
@click.option('--category', '-c',
              type=click.Choice(['all', 'sequence', 'structure', 'liability',
                               'bashour', 'peptide', 'sasa', 'charge',
                               'dssp', 'propka', 'arpeggio']),
              default='all', help='Descriptor category to list')
@click.option('--output', '-o', help='Save list to file')
@click.option('--format', '-f',
              type=click.Choice(['text', 'json', 'csv']),
              default='text', help='Output format')
def list_descriptors(category, output, format):
    """
    List available descriptors with descriptions.
    
    Shows all available descriptors organized by category with
    brief descriptions of what each descriptor measures.
    """
    descriptors = _get_descriptor_definitions()
    
    if category != 'all':
        descriptors = {k: v for k, v in descriptors.items() 
                      if k.lower() == category or category in k.lower()}
    
    if format == 'text':
        output_text = _format_descriptors_text(descriptors)
        if output:
            with open(output, 'w') as f:
                f.write(output_text)
        else:
            click.echo(output_text)
    
    elif format == 'json':
        output_json = json.dumps(descriptors, indent=2)
        if output:
            with open(output, 'w') as f:
                f.write(output_json)
        else:
            click.echo(output_json)
    
    elif format == 'csv':
        rows = []
        for category, items in descriptors.items():
            for name, info in items.items():
                rows.append({
                    'Category': category,
                    'Descriptor': name,
                    'Description': info['description'],
                    'Type': info.get('type', 'float')
                })
        df = pd.DataFrame(rows)
        if output:
            df.to_csv(output, index=False)
        else:
            click.echo(df.to_string())


# Number command for standalone numbering
@cli.command()
@click.argument('sequence')
@click.option('--chain', '-c',
              type=click.Choice(['H', 'L']),
              default='H', help='Chain type')
@click.option('--scheme', '-s',
              type=click.Choice(['imgt', 'kabat', 'chothia', 'martin']),
              default='imgt', help='Numbering scheme')
@click.option('--output', '-o', help='Output file')
@click.option('--format', '-f',
              type=click.Choice(['text', 'csv', 'json']),
              default='text', help='Output format')
def number(sequence, chain, scheme, output, format):
    """
    Number antibody sequence according to standard schemes.
    
    Provides IMGT, Kabat, Chothia, or Martin numbering for
    antibody sequences with CDR annotations.
    
    Example:
        antibody-descriptors number QVQLVQSGAEVKKPGA... -s imgt
    """
    from .sequence import AntibodyNumbering
    
    numbering = AntibodyNumbering(scheme=scheme)
    
    if chain == 'H':
        results = numbering.number_single_sequence(sequence, 'H')
    else:
        results = numbering.number_single_sequence(sequence, 'L')
    
    # Format output
    if format == 'text':
        output_lines = []
        for (pos, ins), aa in results:
            output_lines.append(f"{pos:4d}{ins} {aa}")
        output_text = '\n'.join(output_lines)
        
        if output:
            with open(output, 'w') as f:
                f.write(output_text)
        else:
            click.echo(output_text)
    
    elif format == 'csv':
        rows = [{'Position': pos, 'Insertion': ins, 'Residue': aa}
                for (pos, ins), aa in results]
        df = pd.DataFrame(rows)
        if output:
            df.to_csv(output, index=False)
        else:
            click.echo(df.to_string())
    
    elif format == 'json':
        json_data = [{'position': [pos, ins], 'residue': aa}
                    for (pos, ins), aa in results]
        if output:
            with open(output, 'w') as f:
                json.dump(json_data, f, indent=2)
        else:
            click.echo(json.dumps(json_data, indent=2))


# Validate command
@cli.command()
@click.option('--heavy', '-h', help='Heavy chain sequence or FASTA file')
@click.option('--light', '-l', help='Light chain sequence or FASTA file')
@click.option('--pdb', '-p', type=click.Path(exists=True), 
              help='PDB structure file')
@click.option('--check-numbering', is_flag=True,
              help='Validate antibody numbering')
@click.option('--check-liabilities', is_flag=True,
              help='Check for sequence liabilities')
@click.option('--check-structure', is_flag=True,
              help='Validate structure quality')
@click.pass_context
def validate(ctx, heavy, light, pdb, check_numbering, check_liabilities,
             check_structure):
    """
    Validate antibody sequences and structures.
    
    Performs various validation checks including sequence format,
    numbering validity, liability detection, and structure quality.
    """
    from .utils import validate_sequence, validate_pdb
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Validate sequences
    if heavy or light:
        click.echo("Validating sequences...")
        
        if heavy:
            heavy_seq = _parse_sequence_input(heavy)
            try:
                validate_sequence(heavy_seq, chain_type='H')
                click.secho("  ✓ Heavy chain valid", fg='green')
            except Exception as e:
                click.secho(f"  ✗ Heavy chain invalid: {e}", fg='red')
                validation_results['errors'].append(f"Heavy: {e}")
                validation_results['valid'] = False
        
        if light:
            light_seq = _parse_sequence_input(light)
            try:
                validate_sequence(light_seq, chain_type='L')
                click.secho("  ✓ Light chain valid", fg='green')
            except Exception as e:
                click.secho(f"  ✗ Light chain invalid: {e}", fg='red')
                validation_results['errors'].append(f"Light: {e}")
                validation_results['valid'] = False
        
        # Check numbering if requested
        if check_numbering and validation_results['valid']:
            click.echo("\nChecking antibody numbering...")
            from .sequence import AntibodyNumbering
            numbering = AntibodyNumbering()
            
            try:
                results = numbering.number_sequences(heavy_seq, light_seq)
                click.secho("  ✓ Numbering successful", fg='green')
            except Exception as e:
                click.secho(f"  ⚠ Numbering warning: {e}", fg='yellow')
                validation_results['warnings'].append(f"Numbering: {e}")
        
        # Check liabilities if requested
        if check_liabilities and validation_results['valid']:
            click.echo("\nChecking sequence liabilities...")
            from .sequence import SequenceLiabilityAnalyzer
            analyzer = SequenceLiabilityAnalyzer()
            
            liabilities = analyzer.analyze(heavy_seq, light_seq)
            if liabilities['total'] > 0:
                click.secho(f"  ⚠ {liabilities['total']} liabilities found",
                          fg='yellow')
                for liability_type, count in liabilities.items():
                    if liability_type != 'total' and count > 0:
                        click.echo(f"    - {liability_type}: {count}")
                validation_results['warnings'].append(
                    f"Liabilities: {liabilities['total']} found"
                )
            else:
                click.secho("  ✓ No liabilities detected", fg='green')
    
    # Validate structure
    if pdb:
        click.echo("\nValidating PDB structure...")
        
        try:
            validate_pdb(Path(pdb))
            click.secho("  ✓ PDB structure valid", fg='green')
            
            if check_structure:
                # Additional structure checks
                from .structure import check_structure_quality
                quality = check_structure_quality(Path(pdb))
                
                if quality['missing_atoms'] > 0:
                    click.secho(
                        f"  ⚠ {quality['missing_atoms']} missing atoms",
                        fg='yellow'
                    )
                    validation_results['warnings'].append(
                        f"Missing atoms: {quality['missing_atoms']}"
                    )
                
                if quality['clashes'] > 0:
                    click.secho(
                        f"  ⚠ {quality['clashes']} atomic clashes",
                        fg='yellow'
                    )
                    validation_results['warnings'].append(
                        f"Clashes: {quality['clashes']}"
                    )
        
        except Exception as e:
            click.secho(f"  ✗ PDB invalid: {e}", fg='red')
            validation_results['errors'].append(f"PDB: {e}")
            validation_results['valid'] = False
    
    # Print summary
    click.echo("\n" + "="*50)
    if validation_results['valid']:
        if validation_results['warnings']:
            click.secho("VALIDATION PASSED WITH WARNINGS", fg='yellow', bold=True)
            click.echo(f"Warnings: {len(validation_results['warnings'])}")
        else:
            click.secho("VALIDATION PASSED", fg='green', bold=True)
    else:
        click.secho("VALIDATION FAILED", fg='red', bold=True)
        click.echo(f"Errors: {len(validation_results['errors'])}")
        if validation_results['warnings']:
            click.echo(f"Warnings: {len(validation_results['warnings'])}")
    
    return validation_results


# Config management commands
@cli.group()
def config():
    """Manage configuration files."""
    pass


@config.command()
@click.option('--output', '-o', default='config.json',
              help='Output file path')
def create(output):
    """Create a default configuration file."""
    from .core import Config
    
    config = Config()
    config.save(output)
    click.secho(f"✓ Default configuration saved to {output}", fg='green')
    
    # Show key settings
    click.echo("\nKey settings:")
    click.echo(f"  Numbering scheme: {config.numbering_scheme}")
    click.echo(f"  pH: {config.pH}")
    click.echo(f"  Temperature: {config.temperature}°C")
    click.echo(f"  Output format: {config.output_format}")


@config.command()
@click.argument('config_file', type=click.Path(exists=True))
def show(config_file):
    """Display configuration file contents."""
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    click.echo(json.dumps(config_dict, indent=2))


@config.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('key')
@click.argument('value')
def set(config_file, key, value):
    """Update a configuration parameter."""
    from .core import Config
    
    config = Config.load(config_file)
    
    # Try to parse value as appropriate type
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass  # Keep as string
    
    config.update(**{key: value})
    config.save(config_file)
    
    click.secho(f"✓ Updated {key} = {value}", fg='green')


# Helper functions
def _parse_sequence_input(input_str: str) -> str:
    """Parse sequence from direct input or file."""
    input_path = Path(input_str)
    
    if input_path.exists():
        # Try to read as FASTA
        from Bio import SeqIO
        try:
            record = next(SeqIO.parse(input_path, 'fasta'))
            return str(record.seq).upper()
        except:
            # Try reading as plain text
            with open(input_path, 'r') as f:
                seq = f.read().strip()
                # Remove any non-sequence characters
                seq = ''.join(c for c in seq if c.isalpha())
                return seq.upper()
    else:
        # Direct sequence input
        seq = ''.join(c for c in input_str if c.isalpha())
        return seq.upper()


def _detect_file_format(filepath: str) -> str:
    """Auto-detect file format from extension."""
    suffix = Path(filepath).suffix.lower()
    
    format_map = {
        '.fasta': 'fasta',
        '.fa': 'fasta',
        '.csv': 'csv',
        '.tsv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.json': 'json'
    }
    
    return format_map.get(suffix, 'fasta')


def _load_batch_input(filepath: str, format: str, heavy_col: str,
                     light_col: str, pdb_col: str, id_col: str) -> List[Dict]:
    """Load batch input from various file formats."""
    samples = []
    
    if format == 'fasta':
        from Bio import SeqIO
        for i, record in enumerate(SeqIO.parse(filepath, 'fasta')):
            # Try to determine if heavy or light from ID
            seq = str(record.seq).upper()
            sample = {'id': record.id}
            
            if 'heavy' in record.id.lower() or '_h' in record.id.lower():
                sample['heavy'] = seq
            elif 'light' in record.id.lower() or '_l' in record.id.lower():
                sample['light'] = seq
            else:
                # Default to heavy
                sample['heavy'] = seq
            
            samples.append(sample)
    
    elif format == 'csv':
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            sample = {}
            
            if id_col in df.columns:
                sample['id'] = row[id_col]
            else:
                sample['id'] = f"Sample_{_}"
            
            if heavy_col in df.columns and pd.notna(row[heavy_col]):
                sample['heavy'] = str(row[heavy_col]).upper()
            
            if light_col in df.columns and pd.notna(row[light_col]):
                sample['light'] = str(row[light_col]).upper()
            
            if pdb_col in df.columns and pd.notna(row[pdb_col]):
                sample['pdb'] = row[pdb_col]
            
            samples.append(sample)
    
    elif format == 'excel':
        df = pd.read_excel(filepath)
        for _, row in df.iterrows():
            sample = {}
            
            if id_col in df.columns:
                sample['id'] = row[id_col]
            else:
                sample['id'] = f"Sample_{_}"
            
            if heavy_col in df.columns and pd.notna(row[heavy_col]):
                sample['heavy'] = str(row[heavy_col]).upper()
            
            if light_col in df.columns and pd.notna(row[light_col]):
                sample['light'] = str(row[light_col]).upper()
            
            if pdb_col in df.columns and pd.notna(row[pdb_col]):
                sample['pdb'] = row[pdb_col]
            
            samples.append(sample)
    
    elif format == 'json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                sample = {
                    'id': item.get('id', f"Sample_{len(samples)}"),
                    'heavy': item.get('heavy', '').upper() if 'heavy' in item else None,
                    'light': item.get('light', '').upper() if 'light' in item else None,
                    'pdb': item.get('pdb')
                }
                samples.append(sample)
    
    return samples


def _save_results(df: pd.DataFrame, output: str, format: str):
    """Save results in specified format."""
    if format == 'csv':
        df.to_csv(output, index=False)
    elif format == 'excel':
        df.to_excel(output, index=False, engine='openpyxl')
    elif format == 'json':
        df.to_json(output, orient='records', indent=2)
    elif format == 'parquet':
        df.to_parquet(output, index=False)
    elif format == 'pickle':
        df.to_pickle(output)


def _get_descriptor_definitions() -> Dict[str, Dict]:
    """Get definitions of all available descriptors."""
    return {
        "Sequence Liabilities": {
            "N_Glycosylation": {
                "description": "N-linked glycosylation sites (NXS/T, X≠P)",
                "type": "int"
            },
            "Deamidation": {
                "description": "Asparagine deamidation sites (NG, NS, NT)",
                "type": "int"
            },
            "Oxidation": {
                "description": "Oxidation-prone residues (Met, Trp)",
                "type": "int"
            },
            "Unpaired_Cysteine": {
                "description": "Free cysteines not in disulfide bonds",
                "type": "int"
            },
            "Isomerization": {
                "description": "Aspartate isomerization sites (DG, DS, DT, DD, DH)",
                "type": "int"
            }
        },
        "Bashour Descriptors": {
            "Molecular_Weight": {
                "description": "Molecular weight in Daltons",
                "type": "float"
            },
            "pI": {
                "description": "Isoelectric point",
                "type": "float"
            },
            "Instability_Index": {
                "description": "Protein instability index",
                "type": "float"
            },
            "Aliphatic_Index": {
                "description": "Relative volume occupied by aliphatic side chains",
                "type": "float"
            },
            "GRAVY": {
                "description": "Grand average of hydropathy",
                "type": "float"
            }
        },
        "Structure Descriptors": {
            "SASA": {
                "description": "Total solvent accessible surface area (Ų)",
                "type": "float"
            },
            "SAP_Score": {
                "description": "Spatial aggregation propensity score",
                "type": "float"
            },
            "Charge_Heterogeneity": {
                "description": "Spatial distribution of charged residues",
                "type": "float"
            },
            "DI_Score": {
                "description": "Developability index (SAP - β×charge²)",
                "type": "float"
            }
        },
        "DSSP Features": {
            "Helix_Content": {
                "description": "Percentage of residues in alpha helices",
                "type": "float"
            },
            "Sheet_Content": {
                "description": "Percentage of residues in beta sheets",
                "type": "float"
            },
            "Turn_Content": {
                "description": "Percentage of residues in turns",
                "type": "float"
            },
            "Mean_RSA": {
                "description": "Mean relative solvent accessibility",
                "type": "float"
            }
        },
        "PROPKA Results": {
            "Folded_pI": {
                "description": "Isoelectric point of folded protein",
                "type": "float"
            },
            "Folding_Energy": {
                "description": "Free energy of folding at pH 7 (kcal/mol)",
                "type": "float"
            },
            "Net_Charge_pH7": {
                "description": "Net charge at pH 7",
                "type": "float"
            }
        },
        "Arpeggio Interactions": {
            "Hydrogen_Bonds": {
                "description": "Number of intramolecular hydrogen bonds",
                "type": "int"
            },
            "Salt_Bridges": {
                "description": "Number of salt bridge interactions",
                "type": "int"
            },
            "Hydrophobic_Contacts": {
                "description": "Number of hydrophobic interactions",
                "type": "int"
            }
        }
    }


def _format_descriptors_text(descriptors: Dict) -> str:
    """Format descriptor definitions as readable text."""
    lines = ["Available Antibody Descriptors", "=" * 40, ""]
    
    for category, items in descriptors.items():
        lines.append(f"\n{category}")
        lines.append("-" * len(category))
        
        for name, info in items.items():
            lines.append(f"  • {name}")
            lines.append(f"    {info['description']}")
            lines.append(f"    Type: {info.get('type', 'float')}")
        
        lines.append("")
    
    return "\n".join(lines)


# Entry point
if __name__ == '__main__':
    cli()
    
    
"""
# Basic calculation
antibody-descriptors calculate -h QVQLV... -l DIQMT... -o results.csv

# Structure analysis
antibody-descriptors calculate -p antibody.pdb --structure-only

# Batch processing
antibody-descriptors batch sequences.fasta -o batch_results.csv -j 4

# List descriptors
antibody-descriptors list --category sequence

# Validate inputs
antibody-descriptors validate -h heavy.fasta -p structure.pdb --check-all

# Create config
antibody-descriptors config create -o my_config.json
"""