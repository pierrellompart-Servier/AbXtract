# AbXtract Docker

Containerized deployment of AbXtract - Antibody Descriptor Calculator.

## 📦 Quick Start

### 1. Build the Docker Image

```bash
# From the AbXtract root directory
docker build -t abxtract:latest -f Dockerfile .

# Or using docker-compose
docker-compose build
```

### 2. Prepare Your Data

Create the data directories and add your input files:

```bash
mkdir -p data/test data/output
```

**Input CSV Format:**
```csv
ID,sequence_VH,sequence_VL,pdb_path
Ab001,QVQLVQ...,DIQMTQ...,structures/Ab001.pdb
VHH001,QVQLVE...,,structures/VHH001.pdb
```

> **Note:** PDB paths should be relative to `/data/test/` inside the container.

### 3. Run AbXtract

```bash
# Using docker run
docker run --rm \
    -v $(pwd)/data/test:/data/test:ro \
    -v $(pwd)/data/output:/data/output:rw \
    abxtract:latest \
    python /app/run_abxtract.py \
    -i /data/test/input.csv \
    -o /data/output/ \
    --base-dir /data/test \
    -m r

# Using docker-compose
docker-compose run --rm abxtract \
    python /app/run_abxtract.py \
    -i /data/test/input.csv \
    -o /data/output/ \
    --base-dir /data/test \
    -m r
```

## 🚀 Usage Examples

### Basic Run (Default Mode)

```bash
docker run --rm \
    -v $(pwd)/data/test:/data/test:ro \
    -v $(pwd)/data/output:/data/output:rw \
    abxtract:latest \
    python /app/run_abxtract.py \
    -i /data/test/input.csv \
    -o /data/output/ \
    --base-dir /data/test
```

### Wide/Deep Mode (All Calculations)

```bash
docker run --rm \
    -v $(pwd)/data/test:/data/test:ro \
    -v $(pwd)/data/output:/data/output:rw \
    abxtract:latest \
    python /app/run_abxtract.py \
    -i /data/test/input.csv \
    -o /data/output/ \
    --base-dir /data/test \
    -m wd
```

### Custom Parameters

```bash
docker run --rm \
    -v $(pwd)/data/test:/data/test:ro \
    -v $(pwd)/data/output:/data/output:rw \
    abxtract:latest \
    python /app/run_abxtract.py \
    -i /data/test/input.csv \
    -o /data/output/ \
    --base-dir /data/test \
    -m r \
    --pH 6.5 \
    --numbering-scheme kabat \
    --n-jobs 4
```

### Interactive Shell

```bash
# Open a bash shell inside the container
docker run --rm -it \
    -v $(pwd)/data/test:/data/test:ro \
    -v $(pwd)/data/output:/data/output:rw \
    abxtract:latest \
    bash
```

## 📋 CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input CSV file path | Required |
| `--output` | `-o` | Output directory | Required |
| `--base-dir` | | Base directory for PDB paths | CSV directory |
| `--mode` | `-m` | Computation mode: b, r, mr, mw, wd | r |
| `--numbering-scheme` | `-ns` | imgt, kabat, chothia | imgt |
| `--cdr-definition` | `-cd` | CDR definition scheme | Same as numbering |
| `--pH` | `-p` | pH for calculations | 7.4 |
| `--temperature` | `-t` | Temperature (°C) | 25.0 |
| `--hydrophobicity-scale` | `-hs` | Hydrophobicity scale | Eisenberg |
| `--n-jobs` | | Number of parallel jobs | N_CPUs - 2 |

### Computation Modes

| Mode | Description | Speed |
|------|-------------|-------|
| `b` | Basic - Bashour, charge, CDR only | ⚡ Fastest |
| `r` | Regular - Most analyses (default) | ⚖️ Balanced |
| `mr` | Medium Regular | ⚖️ Balanced |
| `mw` | Medium Wide - Adds ProtPy | 🐢 Slower |
| `wd` | Wide/Deep - All calculations | 🐌 Slowest |

## 📁 Output Files

After running, you'll find these files in your output directory:

| File | Description |
|------|-------------|
| `descriptors_full.csv` | Standard full descriptors (df_mod) |
| `residues_heavy.csv` | Heavy chain residue-level descriptors |
| `residues_light.csv` | Light chain residue-level descriptors |
| `descriptors_raw.csv` | Raw combined descriptors with pH profiles |
| `run_log_*.json` | Run parameters and execution summary |
| `abxtract_run_*.log` | Detailed execution log |

## 🐳 Docker Compose

For easier management, use docker-compose:

```bash
# Build
docker-compose build

# Run analysis
docker-compose run --rm abxtract \
    python /app/run_abxtract.py \
    -i /data/test/input.csv \
    -o /data/output/ \
    --base-dir /data/test

# Start Jupyter notebook (for interactive analysis)
docker-compose --profile notebook up
# Access at http://localhost:8888
```

## 🔧 Advanced Usage

### Limiting Resources

```bash
docker run --rm \
    --cpus="4" \
    --memory="8g" \
    -v $(pwd)/data/test:/data/test:ro \
    -v $(pwd)/data/output:/data/output:rw \
    abxtract:latest \
    python /app/run_abxtract.py \
    -i /data/test/input.csv \
    -o /data/output/ \
    --base-dir /data/test \
    --n-jobs 4
```

### Running on HPC/Cluster

```bash
# Example SLURM submission
singularity exec \
    --bind $(pwd)/data:/data \
    docker://abxtract:latest \
    python /app/run_abxtract.py \
    -i /data/test/input.csv \
    -o /data/output/ \
    --base-dir /data/test
```

### Building for Different Platforms

```bash
# Build for AMD64 (most servers)
docker buildx build --platform linux/amd64 -t abxtract:latest .

# Build for ARM64 (Apple Silicon, AWS Graviton)
docker buildx build --platform linux/arm64 -t abxtract:arm64 .
```

## 🐛 Troubleshooting

### Permission Issues

```bash
# Fix output directory permissions
chmod -R 777 data/output/

# Or run container as current user
docker run --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd)/data/test:/data/test:ro \
    -v $(pwd)/data/output:/data/output:rw \
    abxtract:latest \
    python /app/run_abxtract.py ...
```

### Memory Issues

If you run out of memory with large datasets:
1. Reduce `--n-jobs` to limit parallelism
2. Use mode `b` for faster/lighter calculations
3. Increase Docker memory limit

### PDB Path Issues

Ensure your CSV has PDB paths relative to `--base-dir`:
```csv
ID,sequence_VH,sequence_VL,pdb_path
Ab001,QVQL...,DIQM...,structures/Ab001.pdb
```

Then set `--base-dir /data/input` if your PDBs are at `/data/input/structures/Ab001.pdb`.

## 📝 License

See main repository for license information.
