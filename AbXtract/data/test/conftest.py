"""
Test setup and configuration files for AbXtract.
"""

# tests/conftest.py - pytest configuration
CONFTEST_PY = '''
"""
Pytest configuration for AbXtract tests.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import os

# Test data constants
TEST_HEAVY_SEQUENCE = (
    "QVQLVQSGAEVKKPGASVKVSCKASGGTFSSYAISWVRQAPGQGLEWMG"
    "GIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCAR"
    "SHYGLDYWGQGTLVTVSS"
)

TEST_LIGHT_SEQUENCE = (
    "DIQMTQSPSSLSASVGDRVTITCRASHSISSYLAWYQQKPGKAPKLLIY"
    "AASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTF"
    "GGGTKVEIK"
)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_sequences():
    """Provide test antibody sequences."""
    return {
        'heavy': TEST_HEAVY_SEQUENCE,
        'light': TEST_LIGHT_SEQUENCE
    }

@pytest.fixture
def safe_config():
    """Provide a safe configuration for testing."""
    from AbXtract import Config
    return Config.from_dict({
        'calculate_dssp': False,
        'calculate_propka': False,
        'calculate_arpeggio': False,
        'verbose': False,
        'temp_dir': './test_temp'
    })

@pytest.fixture
def sample_calculator(safe_config):
    """Provide configured calculator instance."""
    from AbXtract import AntibodyDescriptorCalculator
    return AntibodyDescriptorCalculator(config=safe_config)

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_tools: mark test as requiring external tools"
    )
'''

# pytest.ini - pytest configuration file
PYTEST_INI = '''
[tool:pytest]
minversion = 6.0
addopts = 
    -ra 
    -q 
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=AbXtract
    --cov-report=term-missing
    --cov-report=html:htmlcov
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    requires_tools: marks tests requiring external tools
    integration: marks tests as integration tests
filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning
'''

# data/test/README.md - Instructions for test data
TEST_DATA_README = '''
# AbXtract Test Data

This directory contains test data for AbXtract functionality testing.

## Files

### Required Files (auto-generated)
- `test_antibody.fasta` - Test antibody sequences (Heavy and Light chains)

### Optional Files (user-provided)
- `test.pdb` - Test PDB structure file for structure-based analysis

## Test PDB File

To test structure-based functionality, place a PDB file named `test.pdb` in this directory.

### Requirements for test.pdb:
- Must be a valid PDB format file
- Should contain antibody or antibody fragment structure
- Recommended: Fab or scFv structure with both heavy and light chains
- Size: Preferably < 10MB for faster testing

### Suggested PDB structures to download:
You can download test structures from the PDB database (www.rcsb.org):
- **1HZH** - Anti-lysozyme Fab
- **1IGT** - Anti-tissue factor Fab  
- **1A0O** - Anti-progesterone Fab
- **4KMT** - Therapeutic antibody Fab

### Download example:
```bash
# Download a test structure
wget https://files.rcsb.org/download/1HZH.pdb -O test.pdb
```

## Generated Test Files

When you run the test scripts, additional files will be created:
- Test results (CSV, Excel, JSON formats)
- Analysis reports
- Visualization plots

## Directory Structure
```
data/test/
├── README.md                 # This file
├── test_antibody.fasta      # Generated test sequences
├── test.pdb                 # User-provided structure (optional)
└── results/                 # Generated results (created by tests)
    ├── *.csv               # Result tables
    ├── *.xlsx              # Excel workbooks  
    ├── *.json              # JSON exports
    ├── *.png               # Plots
    └── *.txt               # Analysis reports
```

## Usage

1. **Sequence-only testing**: No additional setup required
2. **Structure-based testing**: Add `test.pdb` file to this directory
3. **Run tests**: Use the provided test scripts or notebooks

## Test Data Properties

The provided test sequences represent realistic therapeutic antibody sequences:
- **Heavy Chain**: 117 amino acids (VH domain + partial constant region)
- **Light Chain**: 107 amino acids (VL domain + partial constant region)
- **Source**: Based on therapeutic antibody frameworks
- **Species**: Human
- **Isotype**: IgG1/Kappa (typical therapeutic format)

These sequences are designed to:
- Test all descriptor calculations
- Represent realistic antibody properties
- Avoid known liability motifs (clean test case)
- Work with standard numbering schemes (IMGT, Kabat, etc.)
'''

# setup_test_environment.py - Script to set up test environment
SETUP_SCRIPT = '''#!/usr/bin/env python3
"""
Setup script for AbXtract test environment.

This script creates the necessary directory structure and files
for testing AbXtract functionality.
"""

import os
import sys
from pathlib import Path
import shutil

def create_directory_structure():
    """Create test directory structure."""
    directories = [
        "data/test",
        "data/test/results", 
        "tests",
        "examples",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_test_files():
    """Create test configuration files."""
    
    # Create conftest.py
    conftest_path = Path("tests/conftest.py")
    with open(conftest_path, 'w') as f:
        f.write(CONFTEST_PY)
    print(f"✓ Created: {conftest_path}")
    
    # Create pytest.ini
    pytest_ini_path = Path("pytest.ini")
    with open(pytest_ini_path, 'w') as f:
        f.write(PYTEST_INI)
    print(f"✓ Created: {pytest_ini_path}")
    
    # Create test data README
    readme_path = Path("data/test/README.md")
    with open(readme_path, 'w') as f:
        f.write(TEST_DATA_README)
    print(f"✓ Created: {readme_path}")

def create_test_sequences():
    """Create test FASTA file."""
    test_sequences = {
        "Test_Heavy_Chain": (
            "QVQLVQSGAEVKKPGASVKVSCKASGGTFSSYAISWVRQAPGQGLEWMG"
            "GIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCAR"
            "SHYGLDYWGQGTLVTVSS"
        ),
        "Test_Light_Chain": (
            "DIQMTQSPSSLSASVGDRVTITCRASHSISSYLAWYQQKPGKAPKLLIY"
            "AASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTF"
            "GGGTKVEIK"
        )
    }
    
    fasta_path = Path("data/test/test_antibody.fasta")
    with open(fasta_path, 'w') as f:
        for seq_id, seq in test_sequences.items():
            f.write(f">{seq_id}\\n")
            # Write sequence with line breaks every 60 characters
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\\n")
    
    print(f"✓ Created: {fasta_path}")

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'pandas', 'numpy', 'biopython', 'pytest'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✓ All required packages are available")
        return True

def main():
    """Main setup function."""
    print("🚀 Setting up AbXtract test environment...")
    print("=" * 50)
    
    # Create directories
    create_directory_structure()
    print()
    
    # Create test files
    create_test_files()
    print()
    
    # Create test sequences
    create_test_sequences()
    print()
    
    # Check dependencies
    deps_ok = check_dependencies()
    print()
    
    print("=" * 50)
    if deps_ok:
        print("✅ Test environment setup complete!")
        print()
        print("Next steps:")
        print("1. Optionally add test.pdb file to data/test/ directory")
        print("2. Run tests with: python test_abxtract_simple.py")
        print("3. Or run unit tests with: pytest tests/ -v")
        print("4. Or open the Jupyter notebook for interactive testing")
        return 0
    else:
        print("❌ Setup incomplete - please install missing dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''

print("Test setup files created:")
print("1. conftest.py - pytest configuration")
print("2. pytest.ini - pytest settings") 
print("3. Test data README")
print("4. setup_test_environment.py - Environment setup script")