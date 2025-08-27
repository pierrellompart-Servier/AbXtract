"""
Setup script for antibody-descriptors package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read version
version_file = this_directory / "antibody_descriptors" / "__version__.py"
version_dict = {}
with open(version_file) as f:
    exec(f.read(), version_dict)
version = version_dict["__version__"]

setup(
    name="antibody-descriptors",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive toolkit for antibody and nanobody descriptor calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/antibody-descriptors",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/antibody-descriptors/issues",
        "Documentation": "https://antibody-descriptors.readthedocs.io",
        "Source Code": "https://github.com/yourusername/antibody-descriptors",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "biopython>=1.79",
        "click>=8.0.0",
        "peptides>=0.3.0",
        "prody>=2.0.0",
        "freesasa>=2.1.0",
        "anarci>=1.3",
        "gemmi>=0.5.0",
        "protpy>=1.0.0",
        "tqdm>=4.62.0",
        "openpyxl>=3.0.0",  # For Excel support
        "pyarrow>=6.0.0",   # For Parquet support
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=2.5.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
            "ipykernel>=6.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.17.0",
            "sphinxcontrib-napoleon>=0.7",
            "myst-parser>=0.17.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.6.0",
            "py3Dmol>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "antibody-descriptors=antibody_descriptors.cli:cli",
            "ab-desc=antibody_descriptors.cli:cli",  # Short alias
        ],
    },
    include_package_data=True,
    package_data={
        "antibody_descriptors": [
            "data/*.csv",
            "data/tripeptides/*.pdb",
            "data/config/*.json",
        ],
    },
    zip_safe=False,
)