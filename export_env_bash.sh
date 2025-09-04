#!/bin/bash

# Quick script to export conda environment for sharing
# Usage: bash export_env.sh [environment_name]

ENV_NAME="${1:-abxtract}"
OUTPUT_DIR="${2:-.}"

echo "🔍 Exporting conda environment: $ENV_NAME"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "❌ Environment '$ENV_NAME' does not exist"
    echo ""
    echo "Available environments:"
    conda env list
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Export full environment (platform-specific)
echo "📦 Exporting full environment..."
conda env export -n "$ENV_NAME" > "$OUTPUT_DIR/${ENV_NAME}_environment_full.yml"
echo "✅ Saved to: ${ENV_NAME}_environment_full.yml"

# Export cross-platform environment (no build strings)
echo "🌍 Exporting cross-platform environment..."
conda env export -n "$ENV_NAME" --no-builds > "$OUTPUT_DIR/${ENV_NAME}_environment_crossplatform.yml"
echo "✅ Saved to: ${ENV_NAME}_environment_crossplatform.yml"

# Export minimal environment (from history)
echo "📄 Exporting minimal environment..."
conda env export -n "$ENV_NAME" --from-history > "$OUTPUT_DIR/${ENV_NAME}_environment_minimal.yml"
echo "✅ Saved to: ${ENV_NAME}_environment_minimal.yml"

# Create a cleaned version (recommended for sharing)
echo "🧹 Creating cleaned version for distribution..."
cp "$OUTPUT_DIR/${ENV_NAME}_environment_crossplatform.yml" "$OUTPUT_DIR/${ENV_NAME}_environment.yml"
echo "✅ Saved to: ${ENV_NAME}_environment.yml (RECOMMENDED)"

# Create quick installation script
echo "📝 Creating installation script..."
cat > "$OUTPUT_DIR/install_${ENV_NAME}.sh" << 'EOF'
#!/bin/bash
# Quick installation script for AbXtract environment

echo "🚀 Installing AbXtract environment..."

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Find environment file
if [ -f "abxtract_environment.yml" ]; then
    ENV_FILE="abxtract_environment.yml"
elif [ -f "environment.yml" ]; then
    ENV_FILE="environment.yml"
else
    echo "❌ No environment file found!"
    echo "   Please ensure abxtract_environment.yml is in the current directory"
    exit 1
fi

echo "📦 Using environment file: $ENV_FILE"

# Create environment
echo "⏳ Creating conda environment (this may take a few minutes)..."
conda env create -f "$ENV_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Environment created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "   conda activate abxtract"
    echo ""
    echo "To verify installation:"
    echo "   python -c 'import AbXtract; print(AbXtract.__version__)'"
else
    echo "❌ Failed to create environment"
    echo "Try updating conda: conda update -n base conda"
    exit 1
fi
EOF

chmod +x "$OUTPUT_DIR/install_${ENV_NAME}.sh"
echo "✅ Saved to: install_${ENV_NAME}.sh"

# Create README
echo "📖 Creating README..."
cat > "$OUTPUT_DIR/README_ENVIRONMENT.md" << EOF
# $ENV_NAME Environment Files

This directory contains exported conda environment files for the $ENV_NAME project.

## Quick Installation

\`\`\`bash
# Option 1: Using the installation script
bash install_${ENV_NAME}.sh

# Option 2: Direct conda command
conda env create -f ${ENV_NAME}_environment.yml
conda activate ${ENV_NAME}
\`\`\`

## Available Environment Files

- **${ENV_NAME}_environment.yml** - 🌟 RECOMMENDED: Cleaned, cross-platform compatible
- **${ENV_NAME}_environment_minimal.yml** - Minimal dependencies only
- **${ENV_NAME}_environment_crossplatform.yml** - No build strings
- **${ENV_NAME}_environment_full.yml** - Complete with all packages (platform-specific)

## Verify Installation

\`\`\`bash
conda activate ${ENV_NAME}
python -c "import AbXtract; print('AbXtract version:', AbXtract.__version__)"
\`\`\`

## Update Existing Environment

\`\`\`bash
conda activate ${ENV_NAME}
conda env update -f ${ENV_NAME}_environment.yml --prune
\`\`\`

## Export Your Own Environment

To create these files from your environment:
\`\`\`bash
bash export_env.sh ${ENV_NAME}
\`\`\`

Generated on: $(date)
EOF

echo "✅ Saved to: README_ENVIRONMENT.md"

echo ""
echo "════════════════════════════════════════════════"
echo "✨ Environment export complete!"
echo "════════════════════════════════════════════════"
echo ""
echo "📁 Files created in: $OUTPUT_DIR/"
echo "  ├── ${ENV_NAME}_environment.yml (⭐ RECOMMENDED)"
echo "  ├── ${ENV_NAME}_environment_minimal.yml"
echo "  ├── ${ENV_NAME}_environment_crossplatform.yml"
echo "  ├── ${ENV_NAME}_environment_full.yml"
echo "  ├── install_${ENV_NAME}.sh"
echo "  └── README_ENVIRONMENT.md"
echo ""
echo "📤 To share your environment:"
echo "  1. Share '${ENV_NAME}_environment.yml' (recommended)"
echo "  2. Include 'install_${ENV_NAME}.sh' for easy setup"
echo ""
echo "📥 Others can install with:"
echo "  conda env create -f ${ENV_NAME}_environment.yml"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "════════════════════════════════════════════════"