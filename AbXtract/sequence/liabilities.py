"""
Sequence liability detection module for antibody developability assessment.

This module identifies sequence motifs associated with developability issues
in antibodies, including:
- Unpaired cysteines
- N-linked glycosylation sites  
- Deamidation motifs
- Oxidation-prone residues
- Aggregation-prone regions
- Polyreactivity motifs

References
----------
.. [1] Raybould et al. (2019) "Five computational developability 
       guidelines for therapeutic antibody profiling"
.. [2] Jain et al. (2017) "Biophysical properties of the clinical-stage 
       antibody landscape"
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import logging

from anarci import anarci
from .numbering import annotate_regions

logger = logging.getLogger(__name__)

# Default liability definitions
DEFAULT_LIABILITIES = """# Define liabilities, name, region(s), regex motif
Unpaired Cys (C),fv,C
N-linked glycosylation (NXS/T X not P),fv,N[^P][ST]
Met oxidation (M),cdrs;verniers,M
Trp oxidation (W),cdrs;verniers,W
Asn deamidation (NG NS NT),cdrs;verniers,N[GST]
Asp isomerisation (DG DS DT DD DH),cdrs;verniers,D[GSTDH]
Lysine Glycation (KE KD EK ED),cdrs;verniers,KE|KD|EK|ED
N-terminal glutamate (VH and VL) (E),nterminus,E
Integrin binding (RGD RYD LDV),fv,RGD|RYD|LDV
CD11c/CD18 binding (GPR),fv,GPR
Fragmentation (DP),cdrs;verniers,DP
Polyreactivity (RR VG VV VVV WW WWW WXW),fv,RR|VG|VV|VVV|WW|WWW|WXW"""

# Position macros for special regions
POSITION_MACROS = {
    "verniers": {
        "H": [(2," "), (28," "), (29," "), (54," "), (55," "), 
              (78," "), (88," "), (105," "), (106," "), (118," ")],
        "L": [(4," "), (27," "), (28," "), (29," "), (30," "), 
              (31," "), (32," "), (33," "), (34," "), (35," "), 
              (36," "), (41," "), (42," "), (52," "), (53," "), 
              (55," "), (84," "), (94," "), (118," ")]
    },
    "nterminus": {
        "H": [(1," ")],
        "L": [(1," ")]
    }
}


class Accept:
    """
    A class to select which positions should be compared for liability detection.
    """
    
    _defined_regions = [
        "fwh1", "fwh2", "fwh3", "fwh4",
        "fwl1", "fwl2", "fwl3", "fwl4",
        "cdrh1", "cdrh2", "cdrh3",
        "cdrl1", "cdrl2", "cdrl3"
    ]
    
    _macro_regions = {
        "hframework": set(["fwh1", "fwh2", "fwh3", "fwh4"]),
        "hcdrs": set(["cdrh1", "cdrh2", "cdrh3"]),
        "lframework": set(["fwl1", "fwl2", "fwl3", "fwl4"]),
        "lcdrs": set(["cdrl1", "cdrl2", "cdrl3"]),
    }
    
    _macro_regions.update({
        "framework": _macro_regions["hframework"] | _macro_regions["lframework"],
        "cdrs": _macro_regions["hcdrs"] | _macro_regions["lcdrs"],
        "vh": _macro_regions["hcdrs"] | _macro_regions["hframework"],
        "vl": _macro_regions["lcdrs"] | _macro_regions["lframework"],
    })
    
    _macro_regions.update({"fv": _macro_regions["vh"] | _macro_regions["vl"]})
    
    def __init__(self, numbering_scheme="chothia", definition="north", NOT=False):
        self.NOT = NOT
        self.set_regions()
        self.positions = {"H": set(), "L": set()}
        self.numbering_scheme = numbering_scheme
        self.definition = definition
        self.exclude = {"H": set(), "L": set()}
    
    def set_regions(self, regions=[]):
        """Set the regions to be used."""
        if self.NOT:
            self.regions = self._macro_regions["fv"]
        else:
            self.regions = set()
        self.add_regions(regions)
    
    def add_regions(self, regions):
        """Add regions to the selection."""
        for region in regions:
            region = region.lower()
            if region in self._defined_regions:
                if self.NOT:
                    self.regions = self.regions - set([region])
                else:
                    self.regions.add(region)
            elif region in self._macro_regions:
                if self.NOT:
                    self.regions = self.regions - self._macro_regions[region]
                else:
                    self.regions = self.regions | self._macro_regions[region]
    
    def add_positions(self, positions, chain):
        """Add specific positions to check."""
        for position in positions:
            index, insertion = position
            self.positions[chain].add((index, insertion))
    
    def exclude_positions(self, positions, chain):
        """Exclude specific positions from checking."""
        for position in positions:
            index, insertion = position
            self.exclude[chain].add((index, insertion))
    
    def accept(self, position, chain):
        """Check if position should be accepted."""
        from .numbering import get_region
        
        if position in self.exclude[chain]:
            return False
        
        if (get_region(position, chain, self.numbering_scheme, self.definition) 
                in self.regions or position in self.positions[chain]):
            return True
        return False


class SequenceLiabilityAnalyzer:
    """
    Analyzer for detecting sequence liabilities in antibodies.
    
    Sequence liabilities are motifs or patterns that can cause:
    - Manufacturing issues (aggregation, fragmentation)
    - Stability problems (oxidation, deamidation)  
    - Immunogenicity concerns
    - Reduced expression yields
    
    Attributes
    ----------
    scheme : str
        Numbering scheme to use ('imgt', 'kabat', 'chothia', 'martin')
    restrict_species : bool
        Whether to restrict to human/mouse species for numbering
    liabilities : list
        List of liability definitions
        
    Examples
    --------
    >>> analyzer = SequenceLiabilityAnalyzer()
    >>> liabilities = analyzer.analyze("QVQLV...", "DIQMT...")
    >>> analyzer.save_results("liabilities.csv")
    """
    
    def __init__(self, 
                 scheme: str = "imgt",
                 restrict_species: bool = True,
                 liability_file: Optional[Union[str, Path]] = None):
        """
        Initialize the liability analyzer.
        
        Parameters
        ----------
        scheme : str
            Numbering scheme ('imgt', 'kabat', 'chothia', 'martin')
        restrict_species : bool
            Whether to restrict to human/mouse species
        liability_file : str or Path, optional
            Path to custom liability definitions CSV
        """
        self.scheme = scheme
        self.restrict_species = restrict_species
        self.liabilities = []
        self.results = None
        
        self._load_liability_definitions(liability_file)
    
    def _load_liability_definitions(self, liability_file: Optional[Union[str, Path]] = None):
        """Load liability definitions from file or use defaults."""
        if liability_file and Path(liability_file).exists():
            liability_path = Path(liability_file)
        else:
            # Create temporary default file
            liability_path = Path("liabilities_temp.csv")
            with open(liability_path, 'w') as f:
                f.write(DEFAULT_LIABILITIES)
        
        self.liabilities = []
        with open(liability_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.strip().split(",")
                    if len(parts) == 3:
                        name, regions, regex = parts
                        self.liabilities.append([
                            name,
                            regions.split(";"),
                            re.compile(regex)
                        ])
        
        # Clean up temp file if created
        if not liability_file:
            liability_path.unlink()
        
        logger.info(f"Loaded {len(self.liabilities)} liability definitions")
    
    def analyze(self,
                heavy_sequence: Optional[str] = None,
                light_sequence: Optional[str] = None) -> Dict:
        """
        Analyze sequences for liabilities.
        
        Parameters
        ----------
        heavy_sequence : str, optional
            Heavy chain sequence
        light_sequence : str, optional
            Light chain sequence
            
        Returns
        -------
        dict
            Dictionary containing found liabilities and statistics
        """
        if not heavy_sequence and not light_sequence:
            raise ValueError("At least one sequence must be provided")
        
        # Number sequences
        numbered = self._number_sequences(heavy_sequence, light_sequence)
        if not numbered:
            return {"liabilities": [], "count": 0}
        
        # Find liabilities
        found_liabilities = self._find_liabilities(numbered)
        
        # Format results
        self.results = {
            "liabilities": found_liabilities,
            "count": len(found_liabilities),
            "total_possible": len(self.liabilities),
            "heavy_sequence": heavy_sequence,
            "light_sequence": light_sequence,
            "scheme": self.scheme
        }
        
        return self.results
    
    def _number_sequences(self, 
                         heavy_sequence: Optional[str],
                         light_sequence: Optional[str]) -> Dict:
        """Number sequences using ANARCI."""
        sequences = []
        if heavy_sequence:
            sequences.append(("heavy", heavy_sequence))
        if light_sequence:
            sequences.append(("light", light_sequence))
        
        # Run ANARCI numbering
        if self.restrict_species:
            numbered, alignment_details, _ = anarci(
                sequences, 
                scheme=self.scheme,
                allowed_species=['human', 'mouse']
            )
        else:
            numbered, alignment_details, _ = anarci(
                sequences,
                scheme=self.scheme
            )
        
        # Process results
        result = {"H": None, "L": None}
        
        for i, (seq_numbered, details) in enumerate(zip(numbered, alignment_details)):
            if seq_numbered and len(seq_numbered) == 1:
                chain_type = details[0]["chain_type"]
                if chain_type == "H":
                    result["H"] = [(n, a) for n, a in seq_numbered[0][0] if a != "-"]
                elif chain_type in ["K", "L"]:
                    result["L"] = [(n, a) for n, a in seq_numbered[0][0] if a != "-"]
        
        return result
    
    def _find_liabilities(self, numbering: Dict) -> List[Dict]:
        """Find liabilities in numbered sequences."""
        found_liabilities = []
        termHglut = False
        
        # Convert numbering to sequences and position maps
        sequences = {"H": "", "L": ""}
        index_to_position = {"H": {}, "L": {}}
        
        for chain in ["H", "L"]:
            if numbering[chain]:
                i = 0
                for position, aa in numbering[chain]:
                    if aa != "-":
                        sequences[chain] += aa
                        index_to_position[chain][i] = position
                        i += 1
        
        # Check each liability
        for li, (liability_name, regions, motif) in enumerate(self.liabilities):
            for chain in ["H", "L"]:
                if not numbering[chain]:
                    continue
                
                acceptor = self._get_acceptor(regions, chain)
                
                # Find matches
                for match in motif.finditer(sequences[chain]):
                    start, end = match.start(), match.end() - 1
                    pos_start = index_to_position[chain][start]
                    pos_end = index_to_position[chain][end]
                    
                    # Skip known cysteines
                    if (liability_name == "Unpaired Cys (C)" and 
                        pos_start in [(23, " "), (104, " ")]):
                        continue
                    
                    if acceptor.accept(pos_start, chain):
                        liability_dict = {
                            "name": liability_name,
                            "chain": chain,
                            "start_index": start,
                            "end_index": end,
                            "start_position": list(pos_start),
                            "end_position": list(pos_end),
                            "sequence": sequences[chain][start:end+1],
                            "region": regions
                        }
                        
                        # Handle N-terminal glutamate special case
                        if liability_name == "N-terminal glutamate (VH and VL) (E)":
                            if chain == "L" and termHglut:
                                found_liabilities.append(liability_dict)
                            else:
                                termHglut = liability_dict
                        else:
                            found_liabilities.append(liability_dict)
        
        return found_liabilities
    
    def _get_acceptor(self, regions: List[str], chain: str) -> Accept:
        """Get acceptor object for region checking."""
        accept = Accept(numbering_scheme=self.scheme, definition="north")
        
        for region in regions:
            if region in POSITION_MACROS:
                accept.add_positions(POSITION_MACROS[region][chain], chain)
            else:
                accept.add_regions([region])
        
        return accept
    
    def save_results(self, output_file: Union[str, Path], format: str = "csv"):
        """
        Save liability results to file.
        
        Parameters
        ----------
        output_file : str or Path
            Output file path
        format : str
            Output format ('csv', 'json', 'excel')
        """
        if not self.results:
            raise ValueError("No results to save. Run analyze() first.")
        
        output_file = Path(output_file)
        
        if format == "csv":
            df = self.to_dataframe()
            df.to_csv(output_file, index=False)
        elif format == "json":
            import json
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        elif format == "excel":
            df = self.to_dataframe()
            df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {output_file}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.results or not self.results["liabilities"]:
            return pd.DataFrame()
        
        rows = []
        for liability in self.results["liabilities"]:
            rows.append({
                "Chain": liability["chain"],
                "Liability": liability["name"],
                "Position_Start": f"{liability['start_position'][0]}{liability['start_position'][1]}",
                "Position_End": f"{liability['end_position'][0]}{liability['end_position'][1]}",
                "Sequence": liability["sequence"],
                "Region": ";".join(liability["region"])
            })
        
        return pd.DataFrame(rows)


def get_liabilities(heavy_sequence: Optional[str] = None,
                   light_sequence: Optional[str] = None,
                   scheme: str = "imgt",
                   save: bool = False,
                   outfile: str = "sequence_liabilities.csv",
                   restrict_species: bool = True) -> Tuple[List, int]:
    """
    Convenience function for liability detection.
    
    Parameters
    ----------
    heavy_sequence : str, optional
        Heavy chain sequence
    light_sequence : str, optional
        Light chain sequence
    scheme : str
        Numbering scheme
    save : bool
        Whether to save results
    outfile : str
        Output file path
    restrict_species : bool
        Whether to restrict to human/mouse
        
    Returns
    -------
    tuple
        (list of found liabilities, total number of possible liabilities)
    """
    analyzer = SequenceLiabilityAnalyzer(scheme=scheme, restrict_species=restrict_species)
    results = analyzer.analyze(heavy_sequence, light_sequence)
    
    if save:
        analyzer.save_results(outfile, format="csv")
    
    return results["liabilities"], results["total_possible"]