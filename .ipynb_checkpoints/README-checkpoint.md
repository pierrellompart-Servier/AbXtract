# Antibody Descriptors

A comprehensive toolkit for calculating sequence and structure-based descriptors for antibodies and nanobodies.

## Features

### Sequence-based Descriptors
- **Liability Detection**: Identify problematic sequence motifs
- **Bashour Descriptors**: Physicochemical properties (MW, pI, hydrophobicity)
- **Peptide Descriptors**: 102 comprehensive peptide features

### Structure-based Descriptors  
- **SASA/SAP**: Solvent accessibility and aggregation propensity
- **Charge Analysis**: Charge distribution and heterogeneity
- **DSSP**: Secondary structure assignment
- **PROPKA**: pKa predictions and pH-dependent properties
- **Arpeggio**: Protein interaction analysis

## Installation

### Using conda (recommended)
```bash
conda env create -f environment.yml
conda activate antibody-descriptors
pip install -e .
```

Using pip
bashpip install antibody-descriptors
External Dependencies
Some analyses require external tools:

DSSP: Install via conda or from https://github.com/cmbi/dssp
PROPKA: Install via pip (pip install propka)
Arpeggio: Install from https://github.com/PDBeurope/arpeggio

Usage
As a Python Module
pythonfrom antibody_descriptors import AntibodyDescriptorCalculator

# Initialize calculator
calc = AntibodyDescriptorCalculator()

# Calculate sequence descriptors
seq_results = calc.calculate_sequence_descriptors(
    heavy_sequence="QVQLVQSGAEVKKPGA...",
    light_sequence="DIQMTQSPSSVSASVG..."
)

# Calculate structure descriptors
struct_results = calc.calculate_structure_descriptors("antibody.pdb")

# Calculate all descriptors
all_results = calc.calculate_all(
    heavy_sequence="QVQLVQSGAEVKKPGA...",
    light_sequence="DIQMTQSPSSVSASVG...",
    pdb_file="antibody.pdb"
)

# Save results
all_results.to_csv("descriptors.csv")
Command Line Interface
bash# Calculate all descriptors
antibody-descriptors calculate -h HEAVY_SEQ -l LIGHT_SEQ -p antibody.pdb -o results.csv

# Calculate only sequence descriptors
antibody-descriptors calculate -h heavy.fasta -l light.fasta --sequence-only -o seq_descriptors.csv

# Calculate only structure descriptors
antibody-descriptors calculate -p antibody.pdb --structure-only -o struct_descriptors.csv

# List available descriptors
antibody-descriptors list-descriptors
Configuration
Create a config.json file to customize settings:
json{
    "numbering_scheme": "imgt",
    "pH": 7.0,
    "dssp_path": "/usr/local/bin/dssp",
    "propka_path": "propka3",
    "restrict_species": true
}
Then use it:
bashantibody-descriptors calculate -h HEAVY -l LIGHT --config config.json -o results.csv
