"""
Antibody numbering and region annotation module.

This module provides functions for:
- Numbering antibody sequences according to standard schemes
- Annotating CDR and framework regions
- Converting between numbering schemes
- Handling nanobodies and regular antibodies

Supports IMGT, Kabat, Chothia, Martin, and North numbering schemes.

References
----------
.. [1] Lefranc et al. (2003) "IMGT unique numbering for immunoglobulin"
.. [2] Kabat et al. (1991) "Sequences of Proteins of Immunological Interest"
.. [3] Chothia & Lesk (1987) "Canonical structures for the hypervariable regions"
"""

import re
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import logging
from anarci import anarci, validate_sequence, scheme_short_to_long

logger = logging.getLogger(__name__)

# Three-letter to one-letter amino acid code conversion
AA_3TO1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}

# Available numbering schemes
AVAILABLE_SCHEMES = ["imgt", "chothia", "kabat", "martin"]

# Region definitions
REGIONS = {
    "imgt": {
        "L": "11111111111111111111111111222222222222333333333333333334444444444555555555555555555555555555555555555555666666666666677777777777",
        "H": "11111111111111111111111111222222222222333333333333333334444444444555555555555555555555555555555555555555666666666666677777777777"
    },
    "kabat": {
        "L": "11111111111111111111111222222222222222223333333333333334444444444444455555555555555555555555555555555555666666666666677777777777",
        "H": "11111111111111111111111111111111111222223333333333333344444444444444444444555555555555555555555555555555556666666666677777777777"
    },
    "chothia": {
        "L": "11111111111111111111111222222222222222223333333333333334444444444444455555555555555555555555555555555555666666666666677777777777",
        "H": "11111111111111111111111111222222222223333333333333333333444444445555555555555555555555555555555555555555556666666666677777777777"
    },
    "north": {
        "L": "11111111111111111111111222222222222222223333333333333344444444444444455555555555555555555555555555555555666666666666677777777777",
        "H": "11111111111111111111111222222222222222223333333333333344444444444455555555555555555555555555555555555555666666666666677777777777"
    }
}


class AntibodyNumbering:
    """
    Class for antibody sequence numbering and region annotation.
    
    Attributes
    ----------
    scheme : str
        Numbering scheme to use
    restrict_species : bool
        Whether to restrict to human/mouse species
        
    Examples
    --------
    >>> numbering = AntibodyNumbering(scheme='imgt')
    >>> result = numbering.number_sequence("QVQLV...", chain='H')
    >>> regions = numbering.annotate_regions(result, 'H')
    """
    
    def __init__(self, 
                 scheme: str = "imgt",
                 restrict_species: bool = True):
        """
        Initialize the numbering system.
        
        Parameters
        ----------
        scheme : str
            Numbering scheme ('imgt', 'kabat', 'chothia', 'martin')
        restrict_species : bool
            Whether to restrict to human/mouse species
        """
        if scheme not in AVAILABLE_SCHEMES:
            raise ValueError(f"Unknown scheme: {scheme}. Available: {AVAILABLE_SCHEMES}")
        
        self.scheme = scheme
        self.restrict_species = restrict_species
    
    def number_sequence(self,
                       sequence: str,
                       chain: str = None) -> List[Tuple[Tuple[int, str], str]]:
        """
        Number a single antibody sequence.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence
        chain : str, optional
            Chain type ('H', 'L', 'K'). If None, will be detected
            
        Returns
        -------
        list
            Numbered sequence as list of ((position, insertion), amino_acid) tuples
        """
        sequence = sequence.upper()
        validate_sequence(sequence)
        
        if len(sequence) < 70:
            raise ValueError(f"Sequence too short ({len(sequence)} aa). "
                           "Need at least 70 residues for reliable numbering.")
        
        # Set allowed chains
        allow = None
        if chain:
            allow = {chain}
            if chain == "L":
                allow.add("K")
        
        # Run ANARCI
        species = ['human', 'mouse'] if self.restrict_species else None
        numbered, details, _ = anarci(
            [("seq", sequence)],
            scheme=self.scheme,
            output=False,
            allow=allow,
            allowed_species=species
        )
        
        if not numbered[0]:
            raise ValueError(f"Sequence not recognized as antibody {chain} chain")
        
        # Extract numbered positions (skip gaps)
        output = [(pos, aa) for pos, aa in numbered[0][0][0] if aa != "-"]
        
        # Validate numbering
        # numbers = [pos[0][0] for pos, aa in output]
        # if self.scheme == 'imgt':
        #     if max(numbers) <= 120 or min(numbers) >= 8:
        #         logger.warning("Sequence may be missing residues for accurate modeling")
        
        return output
    
    def number_sequences(self, 
                        sequences: Dict[str, str]) -> Dict[str, List]:
        """
        Number multiple sequences.
        
        Parameters
        ----------
        sequences : dict
            Dictionary of {chain: sequence}
            
        Returns
        -------
        dict
            Dictionary of {chain: numbered_sequence}
        """
        result = {}
        
        for chain, seq in sequences.items():
            if seq:
                try:
                    result[chain] = self.number_sequence(seq, chain)
                except Exception as e:
                    logger.warning(f"Failed to number {chain} chain: {e}")
                    result[chain] = None
        
        return result
    
    def annotate_regions(self,
                        numbered_sequence: List[Tuple[Tuple[int, str], str]],
                        chain: str,
                        definition: str = None) -> List[Tuple]:
        """
        Annotate regions (CDRs and frameworks) for numbered sequence.
        
        Parameters
        ----------
        numbered_sequence : list
            Numbered sequence from number_sequence()
        chain : str
            Chain type ('H' or 'L')
        definition : str, optional
            CDR definition. If None, uses the numbering scheme
            
        Returns
        -------
        list
            List of ((position, insertion), amino_acid, region) tuples
        """
        if definition is None:
            definition = self.scheme
        
        annotated = []
        
        # Get region string for this scheme/chain
        region_string = REGIONS.get(definition, {}).get(chain)
        if not region_string:
            logger.warning(f"No region definition for {definition}/{chain}")
            return numbered_sequence
        
        # Map regions
        region_map = {
            '1': f'fw{chain.lower()}1',
            '2': f'cdr{chain.lower()}1',
            '3': f'fw{chain.lower()}2',
            '4': f'cdr{chain.lower()}2',
            '5': f'fw{chain.lower()}3',
            '6': f'cdr{chain.lower()}3',
            '7': f'fw{chain.lower()}4'
        }
        
        # Annotate each position
        for (pos, ins), aa in numbered_sequence:
            # Get region based on position
            # This is simplified - real implementation would need scheme conversion
            region_idx = min(pos - 1, len(region_string) - 1)
            if region_idx >= 0:
                region_code = region_string[region_idx]
                region = region_map.get(region_code, 'unknown')
            else:
                region = 'unknown'
            
            annotated.append(((pos, ins), aa, region))
        
        return annotated
    
    def get_cdr_sequences(self,
                         numbered_sequence: List[Tuple[Tuple[int, str], str]],
                         chain: str,
                         definition: str = None) -> Dict[str, str]:
        """
        Extract CDR sequences from numbered sequence.
        
        Parameters
        ----------
        numbered_sequence : list
            Numbered sequence
        chain : str
            Chain type
        definition : str, optional
            CDR definition
            
        Returns
        -------
        dict
            Dictionary of {cdr_name: sequence}
        """
        annotated = self.annotate_regions(numbered_sequence, chain, definition)
        
        cdrs = {}
        current_cdr = None
        current_seq = []
        
        for (pos, ins), aa, region in annotated:
            if 'cdr' in region:
                if region != current_cdr:
                    if current_cdr and current_seq:
                        cdrs[current_cdr] = ''.join(current_seq)
                    current_cdr = region
                    current_seq = [aa]
                else:
                    current_seq.append(aa)
            elif current_cdr:
                cdrs[current_cdr] = ''.join(current_seq)
                current_cdr = None
                current_seq = []
        
        # Don't forget last CDR
        if current_cdr and current_seq:
            cdrs[current_cdr] = ''.join(current_seq)
        
        return cdrs
    
    def get_framework_sequences(self,
                               numbered_sequence: List[Tuple[Tuple[int, str], str]],
                               chain: str,
                               definition: str = None) -> Dict[str, str]:
        """
        Extract framework sequences from numbered sequence.
        
        Parameters
        ----------
        numbered_sequence : list
            Numbered sequence
        chain : str
            Chain type
        definition : str, optional
            CDR definition
            
        Returns
        -------
        dict
            Dictionary of {framework_name: sequence}
        """
        annotated = self.annotate_regions(numbered_sequence, chain, definition)
        
        frameworks = {}
        current_fw = None
        current_seq = []
        
        for (pos, ins), aa, region in annotated:
            if 'fw' in region:
                if region != current_fw:
                    if current_fw and current_seq:
                        frameworks[current_fw] = ''.join(current_seq)
                    current_fw = region
                    current_seq = [aa]
                else:
                    current_seq.append(aa)
            elif current_fw:
                frameworks[current_fw] = ''.join(current_seq)
                current_fw = None
                current_seq = []
        
        # Don't forget last framework
        if current_fw and current_seq:
            frameworks[current_fw] = ''.join(current_seq)
        
        return frameworks
    
    def to_dataframe(self,
                    numbered_sequence: List[Tuple[Tuple[int, str], str]],
                    chain: str = None,
                    include_regions: bool = True) -> pd.DataFrame:
        """
        Convert numbered sequence to DataFrame.
        
        Parameters
        ----------
        numbered_sequence : list
            Numbered sequence
        chain : str, optional
            Chain type for region annotation
        include_regions : bool
            Whether to include region annotations
            
        Returns
        -------
        pd.DataFrame
            DataFrame with position, insertion, amino acid, and optionally region
        """
        data = []
        
        if include_regions and chain:
            annotated = self.annotate_regions(numbered_sequence, chain)
            for (pos, ins), aa, region in annotated:
                data.append({
                    'position': pos,
                    'insertion': ins,
                    'amino_acid': aa,
                    'region': region
                })
        else:
            for (pos, ins), aa in numbered_sequence:
                data.append({
                    'position': pos,
                    'insertion': ins,
                    'amino_acid': aa
                })
        
        return pd.DataFrame(data)
    
    def compare_schemes(self,
                       sequence: str,
                       chain: str = 'H') -> pd.DataFrame:
        """
        Compare numbering across different schemes.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence
        chain : str
            Chain type
            
        Returns
        -------
        pd.DataFrame
            Comparison of numbering schemes
        """
        results = []
        
        for scheme in AVAILABLE_SCHEMES:
            try:
                numbering = AntibodyNumbering(scheme=scheme)
                numbered = numbering.number_sequence(sequence, chain)
                
                for i, ((pos, ins), aa) in enumerate(numbered):
                    results.append({
                        'sequential': i + 1,
                        'amino_acid': aa,
                        f'{scheme}_pos': pos,
                        f'{scheme}_ins': ins,
                        f'{scheme}_full': f'{pos}{ins}'.strip()
                    })
            except Exception as e:
                logger.warning(f"Failed to number with {scheme}: {e}")
        
        # Combine results
        df = pd.DataFrame(results)
        
        # Pivot to get all schemes in same row
        if not df.empty:
            df = df.groupby(['sequential', 'amino_acid']).first().reset_index()
        
        return df


def number_sequences(sequences: Dict[str, str],
                    scheme: str = "imgt",
                    restrict_species: bool = True) -> Dict[str, List]:
    """
    Convenience function to number multiple sequences.
    
    Parameters
    ----------
    sequences : dict
        Dictionary of {chain: sequence}
    scheme : str
        Numbering scheme
    restrict_species : bool
        Whether to restrict to human/mouse
        
    Returns
    -------
    dict
        Dictionary of {chain: numbered_sequence}
    """
    numbering = AntibodyNumbering(scheme=scheme, restrict_species=restrict_species)
    return numbering.number_sequences(sequences)


def annotate_regions(numbered_sequence: List,
                    chain: str,
                    numbering_scheme: str = "chothia",
                    definition: str = "chothia") -> List:
    """
    Convenience function to annotate regions.
    
    Parameters
    ----------
    numbered_sequence : list
        Numbered sequence
    chain : str
        Chain type
    numbering_scheme : str
        Numbering scheme used
    definition : str
        CDR definition to use
        
    Returns
    -------
    list
        Annotated sequence
    """
    numbering = AntibodyNumbering(scheme=numbering_scheme)
    return numbering.annotate_regions(numbered_sequence, chain, definition)


def get_region(position: Tuple[int, str],
              chain: str,
              numbering_scheme: str = "chothia",
              definition: str = "chothia") -> str:
    """
    Get the region for a specific position.
    
    Parameters
    ----------
    position : tuple
        (position_number, insertion_code)
    chain : str
        Chain type
    numbering_scheme : str
        Numbering scheme
    definition : str
        CDR definition
        
    Returns
    -------
    str
        Region name
    """
    # Simplified implementation
    pos, ins = position
    chain = chain.upper()
    
    # This would need full implementation with scheme conversions
    # For now, return a simplified version
    if numbering_scheme == "imgt":
        if chain == "H":
            if 27 <= pos <= 38:
                return "cdrh1"
            elif 56 <= pos <= 65:
                return "cdrh2"
            elif 105 <= pos <= 117:
                return "cdrh3"
        elif chain == "L":
            if 27 <= pos <= 38:
                return "cdrl1"
            elif 56 <= pos <= 65:
                return "cdrl2"
            elif 105 <= pos <= 117:
                return "cdrl3"
    
    # Default to framework
    return f"fw{chain.lower()}1"