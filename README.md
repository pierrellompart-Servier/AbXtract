# AbXtract - Comprehensive Antibody Descriptor Analysis Toolkit

[![PyPI version](https://badge.fury.io/py/AbXtract.svg)](https://badge.fury.io/py/AbXtract)
[![Python Version](https://img.shields.io/pypi/pyversions/AbXtract)](https://pypi.org/project/AbXtract/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AbXtract** is a comprehensive Python toolkit for extracting and analyzing antibody descriptors from sequences and structures. It provides a unified interface for calculating thousands of physicochemical, structural, and sequence-based features for antibody characterization and machine learning applications.

## 🎯 Key Features

- **📊 Comprehensive Descriptor Calculation**: Extract 30,000+ descriptors from antibody sequences and structures
- **🔬 Multi-level Analysis**: 
  - Sequence-based features (Bashour, ProtPy, Peptide descriptors)
  - Structure-based features (SASA, DSSP, Arpeggio interactions)
  - Physicochemical properties (charge, pKa via PROPKA, hydrophobicity)
  - Liability detection (PTMs, aggregation hotspots, immunogenicity)
- **⚡ High Performance**: Parallel processing with optimized algorithms
- **🧬 Antibody-Specific**: 
  - CDR identification and analysis
  - Multiple numbering schemes (IMGT, Kabat, Chothia, Martin, AHo)
  - VH/VL interface analysis
- **📈 Advanced Visualization**: Built-in plotting for property profiles
- **🔧 Flexible Integration**: Easy integration with ML pipelines

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Modules](#-core-modules)
- [Usage Examples](#-usage-examples)
- [Descriptor Categories](#-descriptor-categories)
- [Environment Management](#-environment-management)
- [External Tools](#-external-tools)


## 🚀 Installation

### Quick Install via Conda Environment (Recommended)

```bash
# Download the environment file
wget https://github.com/pierrellompart-Servier/AbXtract/raw/main/environment.yml

# Create and activate environment
conda env create -f environment.yml
conda activate abxtract

# Verify installation
python -c "import AbXtract; print(f'AbXtract v{AbXtract.__version__} ready!')"
```

### Export Your Working Environment

If you have a working AbXtract setup, you can save it for others:

```bash
# Quick export
conda activate abxtract
conda env export > abxtract_environment.yml

# Cross-platform export (recommended)
conda env export --no-builds > abxtract_environment_crossplatform.yml

# Or use the provided script
bash export_env_bash.sh abxtract
```

### Alternative Installation Methods

#### Method 1: Development Installation
```bash
# Clone repository
git clone https://github.com/pierrellompart-Servier/AbXtract.git
cd AbXtract

# Install in development mode
pip install -e .
```

## ⚡ Quick Start

### Basic Usage

```python
from AbXtract import AntibodyDescriptorCalculator

# Initialize calculator
calc = AntibodyDescriptorCalculator()

# Calculate descriptors from sequences
sequence_results, liabilities = calc.calculate_sequence_descriptors(
    heavy_sequence="QVQLVQSGAEVKKPGASVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGG",
    light_sequence="DIQMTQSPSSLSASVGDRVTITCRASHSISWLAWYQQKPGKAPKLLIY",
    sequence_id="TestAb"
)

print(f"Calculated {len(sequence_results.columns)} sequence descriptors")
print(f"Detected liabilities: {liabilities['liabilities'].iloc[0]}")
```

### Structure-Based Analysis

```python
# Calculate structure descriptors
structure_results_seq, structure_results_comp, df_residues, df_AA, df_Ab = calc.calculate_structure_descriptors(
    heavy_sequence=heavy_sequence,
    light_sequence=light_sequence,
    pdb_file="antibody.pdb",
    structure_id="TestAb_Structure"
)

print(f"Residue-level descriptors: {df_residues.shape}")
print(f"Amino acid descriptors: {df_AA.shape}")
print(f"Antibody-level descriptors: {df_Ab.shape}")
```

## 🔬 Core Modules

### Sequence Analysis (`AbXtract.sequence`)

- **`BashourDescriptorCalculator`**: Bashour et al. antibody-specific descriptors
- **`PeptideDescriptorCalculator`**: Comprehensive peptide properties
- **`SequenceLiabilityAnalyzer`**: PTM and liability detection
- **`AntibodyNumbering`**: Multiple numbering schemes and CDR identification
- **`protpy_descriptors`**: ProtPy-based sequence features

### Structure Analysis (`AbXtract.structure`)

- **`SASACalculator`**: Solvent accessible surface area analysis
- **`DSSPAnalyzer`**: Secondary structure assignment
- **`ChargeAnalyzer`**: Charge distribution and patches
- **`PropkaAnalyzer`**: pKa predictions and pH-dependent properties
- **`ArpeggioAnalyzer`**: Molecular interactions (H-bonds, salt bridges, π-stacking)
- **`properdesc`**: Proper descriptor calculation

### Utilities (`AbXtract.utils`)

- **`analysis_descriptors`**: Advanced analysis and visualization functions
- **`validators`**: Sequence and structure validation
- **`converters`**: Format conversion utilities
- **`pdb_utils`**: PDB file manipulation

## 📊 Descriptor Categories

### 1. Sequence-Based Descriptors
- Amino acid composition and properties
- Hydrophobicity profiles (multiple scales)
- Charge distribution patterns
- Sequence complexity metrics
- CDR-specific features
- Peptide descriptors for all k-mers

### 2. Structure-Based Descriptors
- Per-residue SASA and burial
- Secondary structure elements
- Interaction networks
- Spatial aggregation propensity (SAP)
- Charge patches
- pH-dependent properties

### 3. Physicochemical Properties
- Molecular weight and pI
- Instability and aliphatic indices
- GRAVY scores
- Extinction coefficients
- Dipole moments
- Electrostatic properties

### 4. Liability Features
- Post-translational modifications
  - N-glycosylation sites
  - Deamidation hotspots
  - Oxidation sites (Met, Trp)
  - Isomerization sites
- Aggregation prone regions
- Immunogenicity motifs
- Polyreactivity signatures

### 5. pH-Dependent Properties
- Charge profiles
- Folded/unfolded states
- Free energy changes
- pKa shifts
- Titration curves

## 📈 Visualization

AbXtract includes comprehensive visualization capabilities:

```python
from AbXtract.utils import analysis_descriptors

# Create complete antibody dataframe with all features
df_heavy_final, df_light_final, df_final = desc_Ab(
    HEAVY_SEQUENCE, 
    LIGHT_SEQUENCE, 
    PDB_FILE
)

# Plot protein properties
fig = analysis_descriptors.plot_protein_properties(df_heavy_final, chain_type='heavy')
plt.show()

# Plot pH profiles
patterns = ["Light_Charges_pH_", "Heavy_Charge_pH_", "Free_Energy_kcal_mol_"]
col_ph = [col for col in df_final.columns if any(p in col for p in patterns)]
object_df = analysis_descriptors.reshape_dataframe_by_object(df_final[col_ph])[0]
fig = analysis_descriptors.plot_ph_profiles(object_df, object_id=0)
plt.show()

# Plot PROPKA-specific properties
fig_propka = analysis_descriptors.plot_propka_properties(df_heavy_final, chain_type='heavy')
plt.show()
```

## 🔧 External Tools Integration

AbXtract integrates with several computational biology tools:

### Required Tools
```python
from AbXtract import Config

# Check tool availability
config = Config()
tool_status = config.check_external_tools()

for tool, available in tool_status.items():
    status = "✅" if available else "❌"
    print(f"{tool}: {status}")
```

### Tool Installation
```bash
# Core tools via conda
conda install -c conda-forge dssp freesasa
conda install -c bioconda hmmer=3.3.2 muscle

# PROPKA for pKa calculations
pip install propka

# ANARCI for antibody numbering
pip install anarci
python -c "import anarci; anarci.setup()"

# Arpeggio for interactions (included in repo)
cd arpeggio && python setup.py install
```

## 📖 Usage Examples

### Example 1: Complete Antibody Analysis Pipeline

```python
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, '/path/to/AbXtract')

from AbXtract import AntibodyDescriptorCalculator, Config
from AbXtract.sequence import (
    SequenceLiabilityAnalyzer,
    BashourDescriptorCalculator,
    PeptideDescriptorCalculator,
    AntibodyNumbering
)
from AbXtract.utils import analysis_descriptors

# Initialize components
config = Config()
numbering = AntibodyNumbering(scheme='imgt')
peptide_calc = PeptideDescriptorCalculator()
calc = AntibodyDescriptorCalculator(config=config)

# Define sequences
HEAVY_SEQUENCE = "QVQLVQSGAEVKKPGASVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGG..."
LIGHT_SEQUENCE = "DIQMTQSPSSLSASVGDRVTITCRASHSISWLAWYQQKPGKAPKLLIY..."
PDB_FILE = Path("data/test/test.pdb")

# Run complete analysis
df_heavy_final, df_light_final, df_final = desc_Ab(
    HEAVY_SEQUENCE, 
    LIGHT_SEQUENCE, 
    PDB_FILE
)

# Export results
df_final.to_csv("antibody_descriptors.csv", index=False)
print(f"Total descriptors calculated: {df_final.shape[1]}")
```

### Example 2: Batch Processing

```python
# Process multiple antibodies
antibodies = [
    {"id": "Ab1", "heavy": "QVQLV...", "light": "DIQMT...", "pdb": "ab1.pdb"},
    {"id": "Ab2", "heavy": "EVQLV...", "light": "EIVLT...", "pdb": "ab2.pdb"},
]

all_results = []
for ab in antibodies:
    df_h, df_l, df_final = desc_Ab(ab["heavy"], ab["light"], ab["pdb"])
    df_final["antibody_id"] = ab["id"]
    all_results.append(df_final)

# Combine results
combined_df = pd.concat(all_results, axis=0)
combined_df.to_csv("batch_analysis.csv")
```

### Example 3: Custom Configuration

```python
# Custom configuration for specific analyses
custom_config = Config.from_dict({
    'pH': 7.4,
    'numbering_scheme': 'kabat',
    'verbose': True,
    'calculate_dssp': True,
    'calculate_propka': True,
    'calculate_arpeggio': True,
    'n_jobs': 4  # Parallel processing
})

calc = AntibodyDescriptorCalculator(config=custom_config)
```


#### `Config`
```python
config = Config(
    pH=7.4,
    numbering_scheme='imgt',  # imgt, kabat, chothia, martin, aho
    calculate_liabilities=True,
    calculate_bashour=True,
    calculate_peptide=True,
    calculate_protpy=True,
    calculate_dssp=True,
    calculate_propka=True,
    calculate_arpeggio=True
)
```


## 👩‍💻 Development

### Repository Structure
```
AbXtract/
├── AbXtract/
│   ├── __init__.py
│   ├── core/           # Main calculation logic
│   ├── sequence/       # Sequence analysis modules
│   ├── structure/      # Structure analysis modules
│   ├── utils/          # Utility functions
│   └── data/           # Reference data and test files
├── arpeggio/           # Arpeggio integration
├── examples/           # Usage examples and notebooks
├── tests/              # Unit tests
├── docs/               # Documentation
└── environment.yml     # Conda environment file
```

