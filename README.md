# AbXtract - Comprehensive Antibody Descriptor Analysis Toolkit

[![PyPI version](https://badge.fury.io/py/AbXtract.svg)](https://badge.fury.io/py/AbXtract)
[![Python Version](https://img.shields.io/pypi/pyversions/AbXtract)](https://pypi.org/project/AbXtract/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/abxtract/badge/?version=latest)](https://abxtract.readthedocs.io)

**AbXtract** is a comprehensive Python toolkit for extracting and analyzing antibody descriptors from sequences and structures. It provides a unified interface for calculating hundreds of physicochemical, structural, and sequence-based features for antibody characterization and machine learning applications.

## 🎯 Key Features

- **📊 Comprehensive Descriptor Calculation**: Extract 500+ descriptors from antibody sequences and structures
- **🔬 Multi-level Analysis**: Sequence, structure, and interaction-based features
- **⚡ High Performance**: Parallel processing and optimized algorithms
- **🧬 Antibody-Specific**: Specialized features for VH/VL domains and CDR regions
- **🔧 Flexible Integration**: Easy integration with ML pipelines
- **📈 Visualization Tools**: Built-in plotting and analysis functions
- **🐳 Docker Support**: Containerized deployment for reproducibility

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Usage Examples](#-usage-examples)
- [API Reference](#-api-reference)
- [Environment Management](#-environment-management)
- [External Tools](#-external-tools)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)

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

### Alternative Installation Methods

#### Method 1: PyPI Installation
```bash
# Basic installation
pip install AbXtract

# With all optional dependencies
pip install AbXtract[all]
```

#### Method 2: Development Installation
```bash
# Clone repository
git clone https://github.com/pierrellompart-Servier/AbXtract.git
cd AbXtract

# Install in development mode
pip install -e .[dev,docs,viz]
```

#### Method 3: Docker Installation
```bash
# Pull and run Docker image
docker pull abxtract/abxtract:latest
docker run -it abxtract/abxtract:latest
```

## ⚡ Quick Start

### Basic Usage

```python
from AbXtract import AntibodyDescriptorCalculator

# Initialize calculator
calc = AntibodyDescriptorCalculator()

# Calculate descriptors from sequences
results = calc.calculate_sequence_descriptors(
    heavy_sequence="QVQLVQSGAEVKKPGASVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGG",
    light_sequence="DIQMTQSPSSLSASVGDRVTITCRASHSISWLAWYQQKPGKAPKLLIY"
)

print(f"Calculated {len(results.columns)} descriptors")
print(results.head())
```

### Structure-Based Analysis

```python
# Calculate descriptors from PDB structure
results = calc.calculate_structure_descriptors(
    pdb_file="antibody.pdb",
    heavy_chain="H",
    light_chain="L"
)

# Access specific descriptor categories
sequence_features = results.filter(regex='sequence_')
structural_features = results.filter(regex='structure_')
cdr_features = results.filter(regex='cdr_')
```

### Command-Line Interface

```bash
# Basic sequence analysis
abxtract analyze --heavy QVQLVQSG... --light DIQMTQSP... -o results.csv

# Structure analysis
abxtract analyze --pdb antibody.pdb --heavy-chain H --light-chain L

# Batch processing
abxtract batch --input sequences.fasta --output descriptors/ --parallel 4
```

## 🔬 Features

### Descriptor Categories

#### 1. **Sequence-Based Descriptors**
- Amino acid composition and properties
- Hydrophobicity profiles
- Charge distribution
- Sequence motifs and patterns
- CDR length and composition

#### 2. **Structure-Based Descriptors**
- Secondary structure elements
- Solvent accessibility (SASA)
- Radius of gyration
- B-factors and flexibility
- Structural compactness

#### 3. **Physicochemical Properties**
- Molecular weight and pI
- Instability index
- Aliphatic index
- GRAVY score
- Extinction coefficients

#### 4. **Interaction Features**
- Hydrogen bonds
- Salt bridges
- Disulfide bonds
- Aromatic interactions
- VH-VL interface properties

#### 5. **CDR-Specific Features**
- CDR canonical classes
- Loop geometry
- Residue preferences
- Structural variability
- Paratope predictions

### Supported Numbering Schemes
- **IMGT**: Standard IMGT numbering
- **Kabat**: Classical Kabat scheme
- **Chothia**: Structural numbering
- **Martin**: Enhanced Chothia
- **AHo**: Aho numbering scheme

## 📖 Usage Examples

### Example 1: Batch Processing Multiple Sequences

```python
import pandas as pd
from AbXtract import AntibodyDescriptorCalculator

# Load sequences
sequences = pd.read_csv("antibody_sequences.csv")

# Initialize calculator with custom config
calc = AntibodyDescriptorCalculator(
    config={
        'n_jobs': 4,  # Parallel processing
        'calculate_structure': False,  # Skip structure features
        'numbering_scheme': 'imgt'  # Use IMGT numbering
    }
)

# Process all sequences
all_descriptors = []
for idx, row in sequences.iterrows():
    descriptors = calc.calculate_sequence_descriptors(
        heavy_sequence=row['VH'],
        light_sequence=row['VL'],
        sequence_id=row['ID']
    )
    all_descriptors.append(descriptors)

# Combine results
results_df = pd.concat(all_descriptors)
results_df.to_csv("antibody_descriptors.csv", index=False)
```

### Example 2: Structure Analysis with Visualization

```python
from AbXtract import AntibodyDescriptorCalculator, Visualizer

# Calculate structural descriptors
calc = AntibodyDescriptorCalculator()
results = calc.calculate_structure_descriptors(
    pdb_file="1igm.pdb",
    heavy_chain="H",
    light_chain="L",
    include_interactions=True
)

# Visualize results
viz = Visualizer()

# Plot hydrophobicity surface
viz.plot_hydrophobicity_surface(results)

# CDR interaction network
viz.plot_cdr_interactions(results)

# Save report
viz.generate_report(results, output="antibody_analysis_report.html")
```

### Example 3: Machine Learning Pipeline Integration

```python
from AbXtract import AntibodyDescriptorCalculator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv("antibody_affinity_data.csv")

# Calculate descriptors
calc = AntibodyDescriptorCalculator()
X = []
for idx, row in data.iterrows():
    descriptors = calc.calculate_sequence_descriptors(
        heavy_sequence=row['VH'],
        light_sequence=row['VL']
    )
    X.append(descriptors)

X = pd.DataFrame(X)
y = data['affinity_nM']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 important features:")
print(feature_importance.head(10))
```

## 🔧 Environment Management

### Saving Your Environment

If you've successfully set up AbXtract and want to share your environment:

```bash
# Method 1: Quick export
conda activate abxtract
conda env export > abxtract_environment.yml

# Method 2: Cross-platform export
conda env export --no-builds > abxtract_environment_crossplatform.yml

# Method 3: Use our export script
python export_env.py
```

### Installing from Environment File

```bash
# Create environment from file
conda env create -f abxtract_environment.yml

# Activate environment
conda activate abxtract

# Update existing environment
conda env update -f abxtract_environment.yml --prune
```

## 🛠️ External Tools

AbXtract integrates with several external tools for enhanced functionality:

### Required Tools
- **ANARCI**: Antibody numbering (installed automatically)
- **DSSP**: Secondary structure assignment
- **FreeSASA**: Solvent accessibility calculations
- **BioPython**: Sequence manipulation

### Optional Tools
- **PROPKA**: pKa predictions
- **Arpeggio**: Interaction analysis
- **Reduce**: Hydrogen addition
- **MUSCLE**: Sequence alignment
- **OpenBabel**: Molecular conversions

### Installing External Tools

```bash
# Via conda (recommended)
conda install -c conda-forge dssp freesasa propka

# ANARCI setup
pip install anarci
python -c "import anarci; anarci.setup()"

# Verify tools
python -c "from AbXtract import Config; Config().check_external_tools()"
```

## 🔬 API Reference

### Main Classes

#### `AntibodyDescriptorCalculator`
Main class for descriptor calculation.

```python
calc = AntibodyDescriptorCalculator(
    config=None,  # Configuration dict
    verbose=True,  # Print progress
    n_jobs=1,  # Parallel jobs
    numbering_scheme='imgt'  # Numbering scheme
)
```

#### `Config`
Configuration management for AbXtract.

```python
config = Config(
    calculate_sequence=True,
    calculate_structure=True,
    calculate_interactions=True,
    dssp_path='/path/to/dssp',
    freesasa_path='/path/to/freesasa'
)
```

#### `Visualizer`
Visualization tools for analysis results.

```python
viz = Visualizer(
    style='seaborn',
    figsize=(10, 8),
    dpi=100
)
```

### Key Methods

- `calculate_sequence_descriptors()`: Extract sequence-based features
- `calculate_structure_descriptors()`: Extract structure-based features
- `calculate_all_descriptors()`: Calculate all available descriptors
- `filter_descriptors()`: Select specific descriptor subsets
- `normalize_descriptors()`: Normalize features for ML

## 👩‍💻 Development

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

```bash
# Fork and clone repository
git clone https://github.com/yourusername/AbXtract.git
cd AbXtract

# Create development environment
conda env create -f environment_dev.yml
conda activate abxtract-dev

# Install in development mode
pip install -e .[dev,test,docs]

# Run tests
pytest tests/

# Check code style
flake8 AbXtract/
black --check AbXtract/

# Build documentation
cd docs && make html
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=AbXtract --cov-report=html

# Run specific test
pytest tests/test_descriptors.py::test_sequence_features
```

## 🔍 Troubleshooting

### Common Issues

#### Import Error
```bash
# Solution: Reinstall in correct environment
conda activate abxtract
pip uninstall AbXtract
pip install -e .
```

#### Missing External Tools
```python
# Check and install missing tools
from AbXtract import Config
config = Config()
missing = config.check_external_tools()
print(f"Missing tools: {[k for k,v in missing.items() if not v]}")
```

#### Memory Issues
```python
# Use batch processing for large datasets
calc = AntibodyDescriptorCalculator(config={'batch_size': 100})
```

### Getting Help

- 📖 [Documentation](https://abxtract.readthedocs.io)
- 💬 [GitHub Issues](https://github.com/pierrellompart-Servier/AbXtract/issues)
- 🎯 [Discussions](https://github.com/pierrellompart-Servier/AbXtract/discussions)
- 📧 Email: support@abxtract.org

## 📝 Citation

If you use AbXtract in your research, please cite:

```bibtex
@software{abxtract2024,
  author = {Llompart, Pierre and Contributors},
  title = {AbXtract: Comprehensive Antibody Descriptor Analysis Toolkit},
  url = {https://github.com/pierrellompart-Servier/AbXtract},
  version = {0.1.0},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Servier Research Institute for supporting this project
- The BioPython team for sequence handling tools
- ANARCI developers for antibody numbering
- All contributors and users of AbXtract

---

**Made with ❤️ by the AbXtract Team**

*For more information, visit our [documentation](https://abxtract.readthedocs.io) or [contact us](mailto:support@abxtract.org).*