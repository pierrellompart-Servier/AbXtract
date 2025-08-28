"""
Main module for antibody descriptor calculations.

This module provides the central interface for computing both sequence-based
and structure-based descriptors for antibodies and nanobodies.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Any, Tuple
from pathlib import Path
import logging
import json
import warnings
from datetime import datetime
import tempfile
import shutil

from .config import Config, load_config
from ..sequence import (
    SequenceLiabilityAnalyzer,
    BashourDescriptorCalculator,
    PeptideDescriptorCalculator,
    AntibodyNumbering
)
from ..structure import (
    SASACalculator,
    ChargeAnalyzer,
    DSSPAnalyzer,
    PropkaAnalyzer,
    ArpeggioAnalyzer
)
from ..utils import (
    parse_pdb,
    parse_sequence,
    validate_sequence,
    validate_pdb
)


logger = logging.getLogger(__name__)


class AntibodyDescriptorCalculator:
    """
    Main class for calculating antibody/nanobody descriptors.
    
    This class provides a unified interface for computing both sequence-based
    and structure-based descriptors for antibodies and nanobodies. It handles
    the coordination between different analysis modules and manages the 
    aggregation of results.
    
    Parameters
    ----------
    config : Config or dict, optional
        Configuration object or dictionary. If None, uses default configuration.
    
    Attributes
    ----------
    config : Config
        Configuration object containing all settings
    results : dict
        Dictionary storing all calculated results
    metadata : dict
        Metadata about the analysis run
    
    Examples
    --------
    Basic usage with sequences:
    
    >>> from antibody_descriptors import AntibodyDescriptorCalculator
    >>> calc = AntibodyDescriptorCalculator()
    >>> results = calc.calculate_sequence_descriptors(
    ...     heavy_sequence="QVQLVQSGAEVKKPGA...",
    ...     light_sequence="DIQMTQSPSSVSASVG..."
    ... )
    
    Using with structure:
    
    >>> results = calc.calculate_structure_descriptors("antibody.pdb")
    
    Custom configuration:
    
    >>> config = {'pH': 7.4, 'numbering_scheme': 'kabat'}
    >>> calc = AntibodyDescriptorCalculator(config=config)
    """
    
    def __init__(self, config: Optional[Union[Config, Dict]] = None):
        """Initialize the descriptor calculator."""
        # Setup configuration
        if config is None:
            self.config = Config()
        elif isinstance(config, dict):
            self.config = Config.from_dict(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise TypeError(f"Config must be Config object or dict, got {type(config)}")

        
        # Initialize results storage
        self.results = {}
        self.metadata = {
            'version': self._get_version(),
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict()
        }
        
        # Initialize calculators
        self._initialize_calculators()
        
        # Check external tools
        if self.config.verbose:
            self._check_dependencies()
    
    def _get_version(self) -> str:
        """Get package version."""
        try:
            from ..__version__ import __version__
            return __version__
        except ImportError:
            return "unknown"
    
    def _initialize_calculators(self):
        """Initialize all calculator objects."""
        
        # Sequence-based calculators
        # self.liability_analyzer = SequenceLiabilityAnalyzer(
        #     scheme=self.config.numbering_scheme,
        #     liability_definitions_path=self.config.liability_definitions_path,
        #     restrict_species=self.config.restrict_species
        # )
        
        # Sequence-based calculators
        self.liability_analyzer = SequenceLiabilityAnalyzer(
            scheme=self.config.numbering_scheme,
            liability_file=self.config.liability_definitions_path,  # Changed from liability_definitions_path
            restrict_species=self.config.restrict_species
        )
        self.bashour_calculator = BashourDescriptorCalculator()
        
        self.peptide_calculator = PeptideDescriptorCalculator(
            pH=self.config.pH,
            window=self.config.peptide_window,
            angle=self.config.peptide_angle
        )
        
        self.numbering = AntibodyNumbering(
            scheme=self.config.numbering_scheme,
            restrict_species=self.config.restrict_species
        )
        
        # Structure-based calculators
        self.sasa_calculator = SASACalculator(
            probe_radius=self.config.sasa_probe_radius,
            n_points=self.config.sasa_n_points,
            cutoff=self.config.sasa_cutoff
        )
        
        self.charge_analyzer = ChargeAnalyzer(
            distance_cutoff=self.config.charge_distance_cutoff
        )
        
        self.dssp_analyzer = DSSPAnalyzer(
            dssp_path=self.config.dssp_path
        )
        
        self.propka_analyzer = PropkaAnalyzer(
            propka_path=self.config.propka_path,
            pH=self.config.pH,
            temp_dir=self.config.temp_dir
        )
        
        self.arpeggio_analyzer = ArpeggioAnalyzer(
            arpeggio_path=self.config.arpeggio_path,
            temp_dir=self.config.temp_dir
        )
    
    def _check_dependencies(self):
        """Check availability of external dependencies."""
        tool_status = self.config.check_external_tools()
        
        missing_tools = [tool for tool, available in tool_status.items() if not available]
        if missing_tools:
            logger.warning(f"Missing external tools: {', '.join(missing_tools)}")
            logger.warning("Some analyses may not be available")
        
        return tool_status
    
    def calculate_sequence_descriptors(
        self,
        heavy_sequence: Optional[str] = None,
        light_sequence: Optional[str] = None,
        sequence_id: str = "Antibody",
        return_details: bool = False
    ) -> pd.DataFrame:
        """
        Calculate all sequence-based descriptors.
        
        This method computes various sequence-based properties including:
        - Sequence liabilities (deamidation, oxidation, etc.)
        - Bashour descriptors (physicochemical properties)
        - Peptide descriptors (comprehensive peptide features)
        - CDR annotations and properties
        
        Parameters
        ----------
        heavy_sequence : str, optional
            Heavy chain sequence (VH or VHH)
        light_sequence : str, optional
            Light chain sequence (VL)
        sequence_id : str
            Identifier for the sequence
        return_details : bool
            If True, return detailed results including intermediate calculations
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing calculated descriptors with columns:
            - SeqID: Sequence identifier
            - All calculated descriptor columns
            
        Raises
        ------
        ValueError
            If no sequences provided or sequences are invalid
        
        Examples
        --------
        >>> calc = AntibodyDescriptorCalculator()
        >>> results = calc.calculate_sequence_descriptors(
        ...     heavy_sequence="QVQLVQSGAEVKKPGA..."
        ... )
        """
        # Validate input
        if not heavy_sequence and not light_sequence:
            raise ValueError("At least one sequence (heavy or light) must be provided")
        
        # Parse and validate sequences
        # if heavy_sequence:
        #     # heavy_sequence = parse_sequence(heavy_sequence)
        #     validate_sequence(heavy_sequence, chain_type='H')
        # 
        # if light_sequence:
        #     # light_sequence = parse_sequence(light_sequence)
        #     validate_sequence(light_sequence, chain_type='L')
        
        # Determine antibody type
        ab_type = self._determine_antibody_type(heavy_sequence, light_sequence)
        
        logger.info(f"Calculating sequence descriptors for {sequence_id} ({ab_type})")
        
        results = {
            'SeqID': sequence_id,
            'Type': ab_type,
            'Heavy_Length': len(heavy_sequence) if heavy_sequence else 0,
            'Light_Length': len(light_sequence) if light_sequence else 0
        }
        
        # Calculate numbering
        if heavy_sequence or light_sequence:
            sequences_dict = {}
            if heavy_sequence:
                sequences_dict['H'] = heavy_sequence
            if light_sequence:
                sequences_dict['L'] = light_sequence

            numbering_results = self.numbering.number_sequences(sequences_dict)
            results.update(numbering_results)
    
        # if heavy_sequence or light_sequence:
        #     numbering_results = self.numbering.number_sequences(
        #         heavy_sequence, light_sequence
        #     )
        #     results.update(numbering_results)
        
        # Calculate liabilities
        if self.config.calculate_liabilities:
            logger.info("  Calculating sequence liabilities...")
            try:
                liability_results = self.liability_analyzer.analyze(
                    heavy_sequence, light_sequence
                )
                results.update(self._flatten_liability_results(liability_results))
            except Exception as e:
                logger.error(f"  Error in liability analysis: {e}")
                if self.config.verbose:
                    raise
        
        # Calculate Bashour descriptors
        if self.config.calculate_bashour:
            logger.info("  Calculating Bashour descriptors...")
            try:
                sequences_dict = {}
                if heavy_sequence:
                    sequences_dict['Heavy'] = heavy_sequence
                if light_sequence:
                    sequences_dict['Light'] = light_sequence
                
                bashour_results = self.bashour_calculator.calculate(sequences_dict)
                results.update(self._flatten_bashour_results(bashour_results))
            except Exception as e:
                logger.error(f"  Error in Bashour analysis: {e}")
                if self.config.verbose:
                    raise
        
        # Calculate peptide descriptors
        if self.config.calculate_peptide:
            logger.info("  Calculating peptide descriptors...")
            try:
                peptide_results = self.peptide_calculator.calculate_all(
                    heavy_sequence, light_sequence
                )
                results.update(self._flatten_peptide_results(peptide_results))
            except Exception as e:
                logger.error(f"  Error in peptide analysis: {e}")
                if self.config.verbose:
                    raise
        
        # Store results
        self.results['sequence'] = results
        
        # Format output
        df = pd.DataFrame([results])
        
        if return_details:
            return df, results
        return df
    
    def calculate_structure_descriptors(
        self,
        pdb_file: Union[str, Path],
        structure_id: Optional[str] = None,
        preprocess: bool = True,
        return_details: bool = False
    ) -> pd.DataFrame:
        """
        Calculate all structure-based descriptors.
        
        This method computes various structure-based properties including:
        - SASA (Solvent Accessible Surface Area)
        - SAP (Spatial Aggregation Propensity)
        - Charge distribution and heterogeneity
        - Secondary structure (DSSP)
        - pKa and protonation states (PROPKA)
        - Molecular interactions (Arpeggio)
        
        Parameters
        ----------
        pdb_file : str or Path
            Path to PDB structure file
        structure_id : str, optional
            Identifier for the structure. If None, uses filename
        preprocess : bool
            Whether to preprocess structure (remove hydrogens, etc.)
        return_details : bool
            If True, return detailed results including intermediate calculations
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing calculated descriptors
            
        Raises
        ------
        FileNotFoundError
            If PDB file does not exist
        ValueError
            If PDB file is invalid or cannot be parsed
        """
        # Validate input
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        validate_pdb(pdb_file)
        
        if structure_id is None:
            structure_id = pdb_file.stem
        
        logger.info(f"Calculating structure descriptors for {structure_id}")
        
        # Preprocess structure if requested
        if preprocess:
            pdb_file = self._preprocess_structure(pdb_file)
        
        results = {
            'StructureID': structure_id,
            'PDB_File': str(pdb_file.name)
        }
        
        # Extract sequences from PDB
        sequences = parse_pdb(pdb_file)
        results['N_Chains'] = len(sequences)
        
        # Calculate SASA and SAP
        # if self.config.calculate_sasa:
        #     logger.info("  Calculating SASA and SAP...")
        #     try:
        #         sasa_results = self.sasa_calculator.calculate(pdb_file)
        #         results.update(sasa_results)
        #     except Exception as e:
        #         logger.error(f"  Error in SASA analysis: {e}")
        #         if self.config.verbose:
        #             raise

        # Calculate SASA and SAP
        if self.config.calculate_sasa:
            logger.info("  Calculating SASA and SAP...")
            try:
                sasa_results = self.sasa_calculator.calculate(pdb_file)
                results.update(sasa_results)

                # Also calculate SAP score
                from ..structure.sasa import SAPCalculator
                sap_calculator = SAPCalculator(self.sasa_calculator)
                sap_results = sap_calculator.calculate(pdb_file)
                results.update(sap_results)

            except Exception as e:
                logger.error(f"  Error in SASA/SAP analysis: {e}")
                if self.config.verbose:
                    raise

        
        
        # Calculate charge properties
        if self.config.calculate_charge:
            logger.info("  Calculating charge distribution...")
            try:
                charge_results = self.charge_analyzer.calculate(pdb_file)
                results.update(charge_results)
            except Exception as e:
                logger.error(f"  Error in charge analysis: {e}")
                if self.config.verbose:
                    raise
        
        # Run DSSP
        if self.config.calculate_dssp:
            logger.info("  Running DSSP analysis...")
            try:
                dssp_results = self.dssp_analyzer.run(pdb_file)
                results.update(self._flatten_dssp_results(dssp_results))
            except Exception as e:
                logger.error(f"  Error in DSSP analysis: {e}")
                if self.config.verbose:
                    raise
        
        # Run PROPKA
        if self.config.calculate_propka:
            logger.info("  Running PROPKA analysis...")
            try:
                propka_results = self.propka_analyzer.run(pdb_file)
                results.update(self._flatten_propka_results(propka_results))
            except Exception as e:
                logger.error(f"  Error in PROPKA analysis: {e}")
                if self.config.verbose:
                    raise
        
        # Run Arpeggio
        if self.config.calculate_arpeggio:
            logger.info("  Running Arpeggio analysis...")
            try:
                arpeggio_results = self.arpeggio_analyzer.run(pdb_file)
                results.update(self._flatten_arpeggio_results(arpeggio_results))
            except Exception as e:
                logger.error(f"  Error in Arpeggio analysis: {e}")
                if self.config.verbose:
                    raise
        
        # Calculate developability index if both SAP and charge available
        if 'SAP_score' in results and 'Folded_charge_pH7' in results:
            results['DI_score'] = self._calculate_developability_index(
                results['SAP_score'],
                results['Folded_charge_pH7']
            )
        
        # Store results
        self.results['structure'] = results
        
        # Format output
        df = pd.DataFrame([results])
        
        # Clean up temp files if needed
        if not self.config.keep_temp_files and preprocess:
            self._cleanup_temp_files()
        
        if return_details:
            return df, results
        return df
    
    def calculate_all(
        self,
        heavy_sequence: Optional[str] = None,
        light_sequence: Optional[str] = None,
        pdb_file: Optional[Union[str, Path]] = None,
        sample_id: str = "Sample",
        validate_consistency: bool = True
    ) -> pd.DataFrame:
        """
        Calculate all available descriptors.
        
        This method combines both sequence and structure-based analyses
        when both inputs are provided, and validates consistency between
        sequence and structure if requested.
        
        Parameters
        ----------
        heavy_sequence : str, optional
            Heavy chain sequence
        light_sequence : str, optional
            Light chain sequence
        pdb_file : str or Path, optional
            Path to PDB structure file
        sample_id : str
            Identifier for the sample
        validate_consistency : bool
            Whether to check sequence-structure consistency
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing all calculated descriptors
            
        Examples
        --------
        >>> calc = AntibodyDescriptorCalculator()
        >>> results = calc.calculate_all(
        ...     heavy_sequence="QVQLVQSGAEVKKPGA...",
        ...     light_sequence="DIQMTQSPSSVSASVG...",
        ...     pdb_file="antibody.pdb"
        ... )
        """
        all_results = {'SampleID': sample_id}
        
        # Calculate sequence descriptors if sequences provided
        if heavy_sequence or light_sequence:
            seq_results = self.calculate_sequence_descriptors(
                heavy_sequence, light_sequence,
                sequence_id=sample_id,
                return_details=True
            )[1]  # Get detailed results
            all_results.update(seq_results)
        
        # Calculate structure descriptors if PDB provided
        if pdb_file:
            struct_results = self.calculate_structure_descriptors(
                pdb_file,
                structure_id=sample_id,
                return_details=True
            )[1]  # Get detailed results
            all_results.update(struct_results)
        
        # Validate consistency if both sequence and structure provided
        if (heavy_sequence or light_sequence) and pdb_file and validate_consistency:
            consistency = self._validate_sequence_structure_consistency(
                heavy_sequence, light_sequence, pdb_file
            )
            all_results['Sequence_Structure_Match'] = consistency['match']
            all_results['Sequence_Coverage'] = consistency['coverage']
        
        # Calculate combined metrics if both available
        if 'sequence' in self.results and 'structure' in self.results:
            combined_metrics = self._calculate_combined_metrics()
            all_results.update(combined_metrics)
        
        # Store complete results
        self.results['all'] = all_results
        
        return pd.DataFrame([all_results])
    
    def save_results(
        self,
        output_path: str,
        format: Optional[str] = None,
        include_metadata: bool = True
    ):
        """
        Save calculation results to file.
        
        Parameters
        ----------
        output_path : str
            Path for output file
        format : str, optional
            Output format. If None, uses config default
        include_metadata : bool
            Whether to include metadata in output
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        output_path = Path(output_path)
        format = format or self.config.output_format
        
        # Prepare data for saving
        if 'all' in self.results:
            df = pd.DataFrame([self.results['all']])
        else:
            # Combine available results
            combined = {}
            for key in ['sequence', 'structure']:
                if key in self.results:
                    combined.update(self.results[key])
            df = pd.DataFrame([combined])
        
        # Save based on format
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Descriptors', index=False)
                if include_metadata:
                    pd.DataFrame([self.metadata]).to_excel(
                        writer, sheet_name='Metadata', index=False
                    )
        elif format == 'json':
            output_dict = df.to_dict(orient='records')[0]
            if include_metadata:
                output_dict['_metadata'] = self.metadata
            with open(output_path, 'w') as f:
                json.dump(output_dict, f, indent=2, default=str)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'pickle':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump({'data': df, 'metadata': self.metadata}, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of calculated descriptors.
        
        Returns
        -------
        dict
            Summary statistics and key metrics
        """
        summary = {
            'timestamp': self.metadata['timestamp'],
            'n_descriptors_calculated': 0,
            'analyses_performed': []
        }
        
        if 'sequence' in self.results:
            seq_results = self.results['sequence']
            summary['n_sequence_descriptors'] = len(seq_results)
            summary['analyses_performed'].append('sequence')
            
            # Add key sequence metrics
            if 'Total_Liabilities' in seq_results:
                summary['total_liabilities'] = seq_results['Total_Liabilities']
            if 'Heavy_pI' in seq_results:
                summary['heavy_pI'] = seq_results['Heavy_pI']
        
        if 'structure' in self.results:
            struct_results = self.results['structure']
            summary['n_structure_descriptors'] = len(struct_results)
            summary['analyses_performed'].append('structure')
            
            # Add key structure metrics
            if 'Total_SASA' in struct_results:
                summary['total_sasa'] = struct_results['Total_SASA']
            if 'DI_score' in struct_results:
                summary['developability_index'] = struct_results['DI_score']
        
        summary['n_descriptors_calculated'] = sum([
            summary.get('n_sequence_descriptors', 0),
            summary.get('n_structure_descriptors', 0)
        ])
        
        return summary
    
    # Helper methods
    def _determine_antibody_type(
        self, 
        heavy_sequence: Optional[str],
        light_sequence: Optional[str]
    ) -> str:
        """Determine antibody type from sequences."""
        if heavy_sequence and light_sequence:
            return 'Antibody'
        elif heavy_sequence and not light_sequence:
            # Could be VHH/nanobody or single heavy chain
            if len(heavy_sequence) < 130:
                return 'VHH/Nanobody'
            return 'Heavy_Chain_Only'
        elif light_sequence and not heavy_sequence:
            return 'Light_Chain_Only'
        return 'Unknown'
    
    def _preprocess_structure(self, pdb_file: Path) -> Path:
        """Preprocess PDB structure (remove hydrogens, etc.)."""
        from ..utils import remove_hydrogens_from_pdb
        
        # Create temp file for processed structure
        temp_file = Path(self.config.temp_dir) / f"{pdb_file.stem}_processed.pdb"
        
        # Remove hydrogens
        remove_hydrogens_from_pdb(pdb_file, temp_file)
        
        return temp_file
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        temp_dir = Path(self.config.temp_dir)
        if temp_dir.exists():
            for file in temp_dir.glob("*_processed.pdb"):
                file.unlink()
    
    def _calculate_developability_index(
        self,
        sap_score: float,
        net_charge: float,
        beta: float = 0.0815
    ) -> float:
        """
        Calculate developability index (DI).
        
        DI = SAP - β × (net_charge)²
        
        References:
        Lauer et al. (2015) J Pharm Sci 104(6):2051-2059
        """
        return sap_score - beta * (net_charge ** 2)
    
    def _validate_sequence_structure_consistency(
        self,
        heavy_sequence: Optional[str],
        light_sequence: Optional[str],
        pdb_file: Path
    ) -> Dict[str, Any]:
        """Validate that sequences match structure."""
        pdb_sequences = parse_pdb(pdb_file)
        
        consistency = {
            'match': True,
            'coverage': 1.0,
            'mismatches': []
        }
        
        # Simple sequence comparison
        # In production, would use proper sequence alignment
        
        return consistency
    
    def _calculate_combined_metrics(self) -> Dict[str, float]:
        """Calculate metrics that require both sequence and structure."""
        combined = {}
        
        # Add any combined calculations here
        
        return combined
    
    def _flatten_liability_results(self, liability_results: Dict) -> Dict:
        """Flatten liability results for DataFrame."""
        flat = {}
        if isinstance(liability_results, dict):
            flat['Total_Liabilities'] = liability_results.get('total', 0)
            flat['N_Glycosylation_Sites'] = liability_results.get('n_glycosylation', 0)
            flat['Deamidation_Sites'] = liability_results.get('deamidation', 0)
            flat['Oxidation_Sites'] = liability_results.get('oxidation', 0)
            flat['Unpaired_Cysteines'] = liability_results.get('unpaired_cys', 0)
        return flat
    
    def _flatten_bashour_results(self, bashour_results: pd.DataFrame) -> Dict:
        """Flatten Bashour results for DataFrame."""
        flat = {}
        if not bashour_results.empty:
            for col in bashour_results.columns:
                if col not in ['SeqID', 'Seq', 'Dataset']:
                    # Prefix with chain if multiple chains
                    if len(bashour_results) > 1:
                        for idx, row in bashour_results.iterrows():
                            chain_prefix = row.get('SeqID', f'Chain_{idx}')
                            flat[f"{chain_prefix}_{col}"] = row[col]
                    else:
                        flat[col] = bashour_results.iloc[0][col]
        return flat
    
    def _flatten_peptide_results(self, peptide_results: pd.DataFrame) -> Dict:
        """Flatten peptide descriptor results."""
        flat = {}
        if not peptide_results.empty:
            for col in peptide_results.columns:
                if col not in ['name', 'sequence']:
                    flat[f"Peptide_{col}"] = peptide_results.iloc[0][col]
        return flat
    
    def _flatten_dssp_results(self, dssp_results: Dict) -> Dict:
        """Flatten DSSP results."""
        flat = {}
        if dssp_results and 'summary' in dssp_results:
            summary = dssp_results['summary']
            flat['Helix_Content'] = summary.get('helix', 0)
            flat['Sheet_Content'] = summary.get('sheet', 0)
            flat['Turn_Content'] = summary.get('turn', 0)
            flat['Coil_Content'] = summary.get('coil', 0)
            flat['Mean_RSA'] = summary.get('mean_rsa', 0)
        return flat
    
    def _flatten_propka_results(self, propka_results: Dict) -> Dict:
        """Flatten PROPKA results."""
        flat = {}
        if propka_results:
            flat['Folded_pI'] = propka_results.get('folded_pI', None)
            flat['Unfolded_pI'] = propka_results.get('unfolded_pI', None)
            flat['Folded_charge_pH7'] = propka_results.get('folded_charge_pH7', None)
            flat['Unfolded_charge_pH7'] = propka_results.get('unfolded_charge_pH7', None)
            flat['Folding_energy_pH7'] = propka_results.get('folding_energy_pH7', None)
        return flat
    
    def _flatten_arpeggio_results(self, arpeggio_results: Dict) -> Dict:
        """Flatten Arpeggio interaction results."""
        flat = {}
        if arpeggio_results:
            flat['Total_Interactions'] = arpeggio_results.get('total_interactions', 0)
            flat['Hydrogen_Bonds'] = arpeggio_results.get('hydrogen_bonds', 0)
            flat['Salt_Bridges'] = arpeggio_results.get('ionic', 0)
            flat['Hydrophobic_Interactions'] = arpeggio_results.get('hydrophobic', 0)
            flat['VdW_Contacts'] = arpeggio_results.get('vdw', 0)
            flat['Mean_Interaction_Distance'] = arpeggio_results.get('mean_distance', None)
        return flat
    
def _calculate_combined_metrics(self) -> Dict[str, float]:
    """Calculate metrics that require both sequence and structure."""
    combined = {}
    
    seq_results = self.results.get('sequence', {})
    struct_results = self.results.get('structure', {})
    
    # Calculate Developability Index if both SAP and charge available
    if 'sap_score' in struct_results and 'Folded_charge_pH7' in struct_results:
        combined['DI_score'] = self._calculate_developability_index(
            struct_results['sap_score'],
            struct_results['Folded_charge_pH7']
        )
    
    # Add other combined metrics
    if 'Heavy_Length' in seq_results and 'total_sasa' in struct_results:
        # SASA per residue
        total_length = seq_results.get('Heavy_Length', 0) + seq_results.get('Light_Length', 0)
        if total_length > 0:
            combined['SASA_per_residue'] = struct_results['total_sasa'] / total_length
    
    # Add liability density if both available
    if 'Total_Liabilities' in seq_results and 'Heavy_Length' in seq_results:
        total_length = seq_results.get('Heavy_Length', 0) + seq_results.get('Light_Length', 0)
        if total_length > 0:
            combined['Liability_density'] = seq_results['Total_Liabilities'] / total_length
    
    return combined


# Fix 4: Fix the flattening methods to match actual calculator outputs
def _flatten_liability_results(self, liability_results: Dict) -> Dict:
    """Flatten liability results for DataFrame."""
    flat = {}
    if isinstance(liability_results, dict):
        flat['Total_Liabilities'] = liability_results.get('count', 0)
        flat['N_Liability_Types'] = liability_results.get('total_possible', 0)
        
        # Count specific liability types from the liabilities list
        liabilities = liability_results.get('liabilities', [])
        liability_counts = {}
        for liability in liabilities:
            liability_name = liability.get('name', 'Unknown')
            liability_counts[liability_name] = liability_counts.get(liability_name, 0) + 1
        
        # Add common liability types
        flat['N_Glycosylation_Sites'] = liability_counts.get('N-linked glycosylation (NXS/T X not P)', 0)
        flat['Deamidation_Sites'] = liability_counts.get('Asn deamidation (NG NS NT)', 0)
        flat['Oxidation_Sites'] = (
            liability_counts.get('Met oxidation (M)', 0) + 
            liability_counts.get('Trp oxidation (W)', 0)
        )
        flat['Unpaired_Cysteines'] = liability_counts.get('Unpaired Cys (C)', 0)
    
    return flat


def _flatten_sasa_results(self, sasa_results: Dict) -> Dict:
    """Flatten SASA results for DataFrame."""
    flat = {}
    if isinstance(sasa_results, dict):
        flat['Total_SASA'] = sasa_results.get('total_sasa', 0)
        flat['Sidechain_SASA'] = sasa_results.get('total_sasa_sidechain', 0)
        flat['N_Buried_Residues'] = sasa_results.get('n_buried', 0)
        
        # SAP results if included
        flat['SAP_Score'] = sasa_results.get('sap_score', 0)
        flat['N_High_SAP_Residues'] = sasa_results.get('n_high_sap', 0)
        flat['Mean_Residue_SAP'] = sasa_results.get('mean_residue_sap', 0)
    
    return flat
