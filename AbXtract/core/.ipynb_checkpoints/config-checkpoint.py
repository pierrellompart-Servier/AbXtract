"""
Configuration management for antibody descriptor calculations.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration class for antibody descriptor calculations.
    
    This class holds all configuration parameters needed for various
    descriptor calculations and external tool paths.
    
    Attributes
    ----------
    numbering_scheme : str
        Antibody numbering scheme ('imgt', 'kabat', 'chothia', 'martin')
    pH : float
        pH value for charge and pKa calculations
    temperature : float
        Temperature in Celsius for calculations
    restrict_species : bool
        Whether to restrict to human/mouse in numbering
    dssp_path : str
        Path to DSSP executable
    propka_path : str
        Path to PROPKA executable
    arpeggio_path : str
        Path to Arpeggio executable
    reduce_path : str
        Path to Reduce executable for adding hydrogens
    
    Sequence-specific parameters
    -----------------------------
    liability_definitions_path : str
        Path to liability definitions CSV file
    peptide_window : int
        Window size for peptide profile calculations
    peptide_angle : int
        Angle for hydrophobic moment calculation
    
    Structure-specific parameters
    ------------------------------
    sasa_probe_radius : float
        Probe radius for SASA calculations (Angstroms)
    sasa_n_points : int
        Number of points for SASA calculations
    sasa_cutoff : float
        Cutoff for exposed/buried classification
    charge_distance_cutoff : float
        Distance cutoff for charge interactions (Angstroms)
    ss_bond_distance : float
        Maximum distance for disulfide bonds (Angstroms)
    
    Output parameters
    -----------------
    output_format : str
        Default output format ('csv', 'json', 'excel')
    verbose : bool
        Whether to print detailed progress messages
    n_jobs : int
        Number of parallel jobs for computations
    """
    
    # General parameters
    numbering_scheme: str = 'imgt'
    pH: float = 7.0
    temperature: float = 25.0
    restrict_species: bool = True
    
    # External tool paths
    dssp_path: str = 'dssp'
    propka_path: str = 'propka3'
    arpeggio_path: str = 'pdbe-arpeggio'
    reduce_path: str = 'reduce'
    muscle_path: str = 'muscle'
    
    # Sequence analysis parameters
    liability_definitions_path: Optional[str] = None
    peptide_window: int = 5
    peptide_angle: int = 100
    hydrophobicity_scale: str = 'Eisenberg'
    
    # Structure analysis parameters
    sasa_probe_radius: float = 1.4
    sasa_n_points: int = 100
    sasa_cutoff: float = 0.075
    charge_distance_cutoff: float = 4.0
    ss_bond_distance: float = 4.0
    
    # CDR definition
    cdr_definition: str = 'imgt'
    
    # Output parameters
    output_format: str = 'csv'
    verbose: bool = True
    n_jobs: int = 1
    temp_dir: str = './temp'
    keep_temp_files: bool = False
    
    # Analysis selection flags
    calculate_liabilities: bool = True
    calculate_bashour: bool = True
    calculate_peptide: bool = True
    calculate_protpy: bool = False
    calculate_sasa: bool = True
    calculate_charge: bool = True
    calculate_dssp: bool = True
    calculate_propka: bool = True
    calculate_arpeggio: bool = True
    calculate_cdr_properties: bool = True
    
    # Advanced parameters
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_paths()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        valid_schemes = ['imgt', 'kabat', 'chothia', 'martin']
        if self.numbering_scheme not in valid_schemes:
            raise ValueError(f"Invalid numbering scheme: {self.numbering_scheme}. "
                           f"Must be one of {valid_schemes}")
        
        valid_cdr_defs = ['imgt', 'kabat', 'chothia', 'north', 'contact']
        if self.cdr_definition not in valid_cdr_defs:
            raise ValueError(f"Invalid CDR definition: {self.cdr_definition}. "
                           f"Must be one of {valid_cdr_defs}")
        
        if not 0 <= self.pH <= 14:
            raise ValueError(f"pH must be between 0 and 14, got {self.pH}")
        
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")
        
        if self.sasa_probe_radius <= 0:
            raise ValueError(f"SASA probe radius must be positive, got {self.sasa_probe_radius}")
        
        valid_formats = ['csv', 'json', 'excel', 'parquet', 'pickle']
        if self.output_format not in valid_formats:
            raise ValueError(f"Invalid output format: {self.output_format}. "
                           f"Must be one of {valid_formats}")
    
    def _setup_paths(self):
        """Setup and validate paths."""
        # Create temp directory if needed
        if self.temp_dir:
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Set default liability definitions path if not provided
        if self.liability_definitions_path is None:
            package_dir = Path(__file__).parent.parent
            self.liability_definitions_path = str(
                package_dir / 'data' / 'liabilities.csv'
            )
    
    def check_external_tools(self) -> Dict[str, bool]:
        """
        Check availability of external tools.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping tool names to availability status
        """
        import shutil
        
        tools = {
            'dssp': self.dssp_path,
            'propka': self.propka_path,
            'arpeggio': self.arpeggio_path,
            'reduce': self.reduce_path,
            'muscle': self.muscle_path
        }
        
        availability = {}
        for tool_name, tool_path in tools.items():
            available = shutil.which(tool_path) is not None
            availability[tool_name] = available
            if not available and self.verbose:
                logger.warning(f"{tool_name} not found at {tool_path}")
        
        return availability
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save configuration file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create Config instance from dictionary.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        Config
            Configuration instance
        """
        return cls(**config_dict)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to configuration file
            
        Returns
        -------
        Config
            Configuration instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Parameters
        ----------
        **kwargs
            Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        self._validate_config()
    
    def get_analysis_modules(self) -> Dict[str, bool]:
        """
        Get dictionary of which analysis modules are enabled.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping analysis names to enabled status
        """
        return {
            'liabilities': self.calculate_liabilities,
            'bashour': self.calculate_bashour,
            'peptide': self.calculate_peptide,
            'protpy': self.calculate_protpy,
            'sasa': self.calculate_sasa,
            'charge': self.calculate_charge,
            'dssp': self.calculate_dssp,
            'propka': self.calculate_propka,
            'arpeggio': self.calculate_arpeggio,
            'cdr_properties': self.calculate_cdr_properties
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(scheme={self.numbering_scheme}, pH={self.pH}, n_jobs={self.n_jobs})"


def load_config(filepath: Optional[str] = None) -> Config:
    """
    Load configuration from file or environment.
    
    This function attempts to load configuration from:
    1. Specified filepath
    2. Environment variable ANTIBODY_DESCRIPTORS_CONFIG
    3. Default configuration
    
    Parameters
    ----------
    filepath : str, optional
        Path to configuration file
        
    Returns
    -------
    Config
        Configuration instance
    """
    if filepath and Path(filepath).exists():
        return Config.load(filepath)
    
    env_config = os.environ.get('ANTIBODY_DESCRIPTORS_CONFIG')
    if env_config and Path(env_config).exists():
        logger.info(f"Loading config from environment: {env_config}")
        return Config.load(env_config)
    
    logger.info("Using default configuration")
    return Config()


def create_default_config(output_path: str = 'config.json'):
    """
    Create a default configuration file.
    
    Parameters
    ----------
    output_path : str
        Path to save default configuration
    """
    config = Config()
    config.save(output_path)
    print(f"Default configuration saved to {output_path}")
    return config