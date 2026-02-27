#!/bin/bash
# Setup Python environment using uv for DoubleML.jl validation tests

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "⚠️  Warning: uv is not installed!"
    echo ""
    echo "To install uv, run one of the following:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Or visit: https://github.com/astral-sh/uv"
    echo ""
    echo "After installing uv, run this script again."
    exit 1
fi

echo "Setting up Python environment with uv..."
echo "Directory: $SCRIPT_DIR"

# Initialize uv project (creates .venv, pyproject.toml)
if [ ! -f "pyproject.toml" ]; then
    echo "Initializing uv project..."
    uv init
fi

# Add required packages
echo "Adding Python dependencies..."
uv add numpy pandas doubleml xgboost scikit-learn

echo "✓ Python environment setup complete!"
echo ""
echo "You can now run the validation tests:"
echo "  cd $SCRIPT_DIR"
echo "  uv run python generate_data_python.py"
