# Antibody Descriptors

[![PyPI version](https://badge.fury.io/py/antibody-descriptors.svg)](https://badge.fury.io/py/antibody-descriptors)
[![Python Version](https://img.shields.io/pypi/pyversions/antibody-descriptors)](https://pypi.org/project/antibody-descriptors/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/antibody-descriptors/badge/?version=latest)](https://antibody-descriptors.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/yourusername/antibody-descriptors/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/antibody-descriptors/actions/workflows/tests.yml)

A comprehensive toolkit for calculating sequence and structure-based descriptors for antibodies and nanobodies, including developability metrics, physicochemical properties, and structural features.

## 🚀 Features

### Sequence-based Descriptors
- **Liability Detection**: Identify problematic sequence motifs (glycosylation, deamidation, oxidation, etc.)
- **Bashour Descriptors**: Comprehensive physicochemical properties (MW, pI, hydrophobicity, instability index)
- **Peptide Descriptors**: 100+ peptide features including Atchley factors, BLOSUM indices, Z-scales
- **CDR Analysis**: Automatic CDR identification and property calculation

### Structure-based Descriptors  
- **SASA/SAP**: Solvent accessibility and spatial aggregation propensity
- **Charge Analysis**: Charge distribution, heterogeneity, and dipole moment
- **DSSP**: Secondary structure content and accessibility
- **PROPKA**: pKa predictions, folding energy, pH-dependent properties
- **Arpeggio**: Detailed interaction analysis (H-bonds, salt bridges, hydrophobic)
- **Developability Index**: Combined metric for therapeutic potential

## 📦 Installation

### Using pip (recommended)
```bash
pip install antibody-descriptors
Using conda
bashconda env create -f environment.yml
conda activate antibody-descriptors
pip install -e .
Development installation
bashgit clone https://github.com/yourusername/antibody-descriptors.git
cd antibody-descriptors
pip install -e .[dev,docs,viz]
External Dependencies
Some analyses require external tools:

DSSP: For secondary structure assignment
bashconda install -c conda-forge dssp

PROPKA: For pKa calculations (included via pip)
Arpeggio: For interaction analysis
bashpip install biopython numpy
git clone https://github.com/PDBeurope/arpeggio.git


🎯 Quick Start
As a Python Module
pythonfrom antibody_descriptors import AntibodyDescriptorCalculator

# Initialize calculator
calc = AntibodyDescriptorCalculator()

# Calculate sequence descriptors
seq_results = calc.calculate_sequence_descriptors(
    heavy_sequence="QVQLVQSGAEVKKPGASVKVSCKASGGTFGSSAIHWVRQ...",
    light_sequence="DIQMTQSPSSVSASVGDRVTITCRASHSIGTYLAWYQQK..."
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

# Batch process multiple sequences
antibody-descriptors batch sequences.fasta -o batch_results.csv -j 4

# List available descriptors
antibody-descriptors list

# Validate inputs
antibody-descriptors validate -h heavy.fasta -p structure.pdb --check-all