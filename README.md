# AbXtract - Antibody Descriptor Calculator

A comprehensive Python package for calculating structural and sequence-based descriptors for antibodies, including physicochemical properties, liability identification, and detailed structural analysis.

-> Trouble with DSSP library that can't process some pdb (I think due to missing hydrogrens)

## 🚀 Quick Installation

```bash
# Create conda environment
conda env create -f abxtract.yml -n abxtract
conda activate abxtract
pip install abnumber propka
conda install -c salilab dssp -y # i had difficulties making dssp work
pip install peptides protpy prody tqdm numba matplotlib
pip install scikit-learn
pip install seaborn 
conda install bioconda::anarci
pip install freesasa
conda install conda-forge::openmm
conda install conda-forge::pdbfixer
pip install pdbe-arpeggio
conda install bioconda::muscle
conda install bioconda::reduce
pip install ipykernel
python -m ipykernel install --user --name=abxtract --display-name "abxtract"
```

```bash
# Docker build
docker build -t abxtract:latest .
mkdir -p data/test data/output
```

```bash
# Docker clean-up
# Stop and remove any running containers using the image
docker stop $(docker ps -q --filter ancestor=abxtract:latest) 2>/dev/null || true
docker rm $(docker ps -aq --filter ancestor=abxtract:latest) 2>/dev/null || true

# Remove the image
docker rmi abxtract:latest

# Optional: Remove all dangling/unused images to free space
docker image prune -f

# Optional: Full cleanup (removes all unused images, containers, volumes)
docker system prune -a
```

```bash
# Docker run
docker run --rm \
    -v $(pwd):/workspace:ro \
    -v $(pwd)/data/output:/data/output:rw \
    abxtract:latest \
    python /app/run_abxtract.py \
    -i /workspace/data/test/input.csv \
    -o /data/output/ \
    --base-dir /workspace \
    -m r

# Or use docker-compose
docker-compose run --rm abxtract \
    python /app/run_abxtract.py \
    -i /workspace/data/test/input.csv \
    -o /data/output/ \
    --base-dir /workspace

```

## 💡 Quick Start

```bash
conda activate abxtract

# Basic run (mode r by default)
python run_abxtract.py -i ./data/test/input.csv -o results/

# Wide/deep mode with custom pH
python run_abxtract.py -i ./data/test/input.csv -o output/ -m wd -p 6.5

# Custom numbering scheme
python run_abxtract.py -i ./data/test/input.csv -o output/ --numbering-scheme kabat

# Force 8 parallel jobs
python run_abxtract.py -i ./data/test/input.csv -o output/ --n-jobs 8
```

### All CLI Options
```
-i, --input           Input CSV file (required)
-o, --output          Output directory (required)
-m, --mode            b/r/mr/mw/wd (default: r)
--numbering-scheme    imgt/kabat/chothia (default: imgt)
--cdr-definition      imgt/kabat/chothia/north/contact
-p, --pH              pH value (default: 7.4)
-t, --temperature     Temperature °C (default: 25.0)
-hs, --hydrophobicity-scale  Eisenberg/KyteDoolittle/etc.
--n-jobs              Override parallel jobs count
--abxtract-path       Path to AbXtract package
```

## 📊 Descriptor Types

### 1. Sequence Descriptors
Analyze sequence composition and identify potential liabilities.

```python
sequence_results, liabilities = calc.calculate_sequence_descriptors(
    heavy_sequence=HEAVY_SEQ,
    light_sequence=LIGHT_SEQ,
    sequence_id="Ab1"
)
```

**Outputs:**
- Amino acid composition
- PTM sites (N-glycosylation, oxidation, deamidation)
- Liability motifs (unpaired Cys, polyreactivity, integrin binding)
- Charge distribution
- Isoelectric point (pI)

### 2. Peptide Descriptors
Calculate physicochemical properties for full sequences and CDRs.

```python
from AbXtract.sequence import PeptideDescriptorCalculator

peptide_calc = PeptideDescriptorCalculator()
peptide_results = peptide_calc.calculate_all(
    heavy_sequence=HEAVY_SEQ,
    light_sequence=LIGHT_SEQ
)
```

**Outputs:**
- Molecular weight (MW)
- Hydrophobicity (HW, Eisenberg, Rose, Janin, Engelman scales)
- etc...

### 3. Structure Descriptors
Analyze 3D structure from PDB files.

```python
structure_seq, structure_comp, df_res, df_AA, df_Ab = calc.calculate_structure_descriptors(
    heavy_sequence=HEAVY_SEQ,
    light_sequence=LIGHT_SEQ,
    pdb_file="antibody.pdb",
    structure_id="Ab1"
)
```

**Outputs:**
- **SASA/RASA**: Solvent accessible surface area (absolute & relative)
- **SAP**: Spatial aggregation propensity scores
- **Buried residues**: Core vs surface classification
- **Disulfide bonds**: Cysteine pairing identification
- **Secondary structure**: α-helix, β-sheet content (via DSSP)
- **pKa values**: Ionization states (via PROPKA)
- **Interactions**: H-bonds, salt bridges (via Arpeggio)

### 4. Numbering & CDR Extraction
Apply IMGT/Kabat/Chothia numbering schemes and extract CDR sequences.

```python
from AbXtract.sequence import AntibodyNumbering

numbering = AntibodyNumbering(scheme='imgt')

# Number sequences
heavy_numbered = numbering.number_sequence(HEAVY_SEQ, 'H')
light_numbered = numbering.number_sequence(LIGHT_SEQ, 'L')

# Extract CDRs
annotated_H, cdrs_H = numbering.get_cdr_sequences(heavy_numbered, 'H')
annotated_L, cdrs_L = numbering.get_cdr_sequences(light_numbered, 'L')

print(cdrs_H)
# {'CDR1-IMGT': 'GGTFGRYG', 'CDR2-IMGT': 'ISPSGGTT', 'CDR3-IMGT': 'AREKDGYPGKGFDI'}
```

## 📋 Output Data Structures

### Residue-Level DataFrame (`df_AA`)
Per-residue annotations for both heavy and light chains.

| Column | Description |
|--------|-------------|
| `position_seq` | Sequential position (1-based) |
| `position_num` | IMGT/Kabat numbered position |
| `amino_acid` | Single letter amino acid code |
| `region` | FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4 |
| `chain` | H (Heavy) or L (Light) |
| `SASA` | Solvent accessible surface area (Ų) |
| `RASA` | Relative ASA (%) |
| `sap` | Spatial aggregation propensity score |
| `high_sap` | Boolean for high SAP regions |
| `buried` | Boolean for buried residues |
| `disulfide_bond` | Boolean for Cys in disulfide |
| `pka` | Predicted pKa value |
| `pka_shift` | pKa shift from model value |
| `hydrophobicity_*` | Multiple hydrophobicity scales |
| `charge` | Charge at specified pH |
| Liability columns | PTM sites, liability motifs |

### Antibody-Level DataFrame (`df_Ab`)
Single-row summary with comprehensive antibody properties.

**Categories:**
- **Structure metrics**: Total SASA, buried area, disulfide count, high SAP residues
- **Sequence metrics**: CDR sequences (H1-3, L1-3), chain lengths, MW
- **Peptide properties**: Hydrophobicity indices, pI, instability, aromaticity (per chain)
- **Liability flags**: PTM sites, unpaired Cys, polyreactivity motifs

## ⚙️ Configuration

```python
from AbXtract import Config

# Default configuration
config = Config()

# Custom configuration
config = Config.from_dict({
    'pH': 7.4,
    'numbering_scheme': 'imgt',  # Options: 'imgt', 'kabat', 'chothia'
    'verbose': True,
    'calculate_dssp': True,
    'calculate_propka': True,
    'calculate_arpeggio': False
})

# Initialize with custom config
calc = AntibodyDescriptorCalculator(config=config)

# Check tool availability
tool_status = config.check_external_tools()
print(tool_status)
# {'dssp': True, 'propka': True, 'arpeggio': False}
```

## 🔍 Complete Analysis Workflow

```python
import sys
sys.path.insert(0, '/path/to/AbXtract')

from AbXtract import AntibodyDescriptorCalculator, Config
from AbXtract.sequence import AntibodyNumbering, PeptideDescriptorCalculator

# Initialize components
config = Config()
calc = AntibodyDescriptorCalculator(config=config)
numbering = AntibodyNumbering(scheme='imgt')
peptide_calc = PeptideDescriptorCalculator()

# Define sequences
HEAVY_SEQ = "QVQLVQSGAEVKKPGASVKVSCKASGGTFGRYGIHWVRQAPGKGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAREKDGYPGKGFDIWGQGTMVTVSS"
LIGHT_SEQ = "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQANSFPLTFGGGTKVEIK"

# 1. Complete analysis
df_AA, df_Ab = calc.calculate_antibody_features(
    pdb_file="antibody.pdb",
    heavy_sequence=HEAVY_SEQ,
    light_sequence=LIGHT_SEQ,
    isotype='igg1',
    lc_type='kappa',
    pH=7.4
)

# 2. Individual descriptor calculations
sequence_results, liabilities = calc.calculate_sequence_descriptors(
    heavy_sequence=HEAVY_SEQ,
    light_sequence=LIGHT_SEQ,
    sequence_id="Ab1"
)

peptide_results = peptide_calc.calculate_all(
    heavy_sequence=HEAVY_SEQ,
    light_sequence=LIGHT_SEQ
)

structure_seq, structure_comp, df_residues, _, _ = calc.calculate_structure_descriptors(
    heavy_sequence=HEAVY_SEQ,
    light_sequence=LIGHT_SEQ,
    pdb_file="antibody.pdb",
    structure_id="Ab1"
)

# 3. CDR extraction and numbering
heavy_numbered = numbering.number_sequence(HEAVY_SEQ, 'H')
light_numbered = numbering.number_sequence(LIGHT_SEQ, 'L')

annotated_H, cdrs_H = numbering.get_cdr_sequences(heavy_numbered, 'H')
annotated_L, cdrs_L = numbering.get_cdr_sequences(light_numbered, 'L')

# 4. Get hydrophobicity profiles
heavy_profiles = numbering.get_peptide_profiles(HEAVY_SEQ)
light_profiles = numbering.get_peptide_profiles(LIGHT_SEQ)

# 5. Export results
df_AA.to_csv("residue_analysis.csv", index=False)
df_Ab.to_csv("antibody_summary.csv", index=False)

# Print CDR sequences
print("Heavy Chain CDRs:")
for cdr, seq in cdrs_H.items():
    print(f"  {cdr}: {seq}")

print("\nLight Chain CDRs:")
for cdr, seq in cdrs_L.items():
    print(f"  {cdr}: {seq}")
```

## 🧪 Liability Screening Example

```python
# Focus on liability analysis
sequence_results, liabilities = calc.calculate_sequence_descriptors(
    heavy_sequence=HEAVY_SEQ,
    light_sequence=LIGHT_SEQ,
    sequence_id="Screening"
)

# Extract liability list
liability_list = liabilities['liabilities'].iloc[0]

# Filter high-risk liabilities
high_risk_liabilities = [
    'Unpaired_Cys', 
    'N-linked_glycosylation', 
    'Met_oxidation',
    'Asn_deamidation',
    'Asp_isomerization'
]

high_risk = [l for l in liability_list if l['name'] in high_risk_liabilities]

print(f"Found {len(high_risk)} high-risk liabilities:")
for liability in high_risk:
    print(f"  {liability['name']} at {liability['chain']}:{liability['start_position']}-{liability['end_position']}")
    print(f"    Sequence: {liability['sequence']}")
```
