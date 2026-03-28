#!/usr/bin/env bash

# JAX Playground Setup Script
# This script automates the setup process for the JAX Playground project

set -e  # Exit on error

echo "╔════════════════════════════════════════════════╗"
echo "║   JAX Playground - Environment Setup Script    ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ Python 3.10+ required, but found Python $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION found"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✓ Poetry $(poetry --version) found"

echo ""
echo "════════════════════════════════════════════════"
echo "Setup Options:"
echo "════════════════════════════════════════════════"
echo ""
echo "1) Poetry Setup (Recommended)"
echo "2) pip + venv Setup"
echo "3) conda Setup"
echo ""
read -p "Select option (1-3) [1]: " setup_option
setup_option=${setup_option:-1}

case $setup_option in
    1)
        echo ""
        echo "════════════════════════════════════════════════"
        echo "Installing with Poetry..."
        echo "════════════════════════════════════════════════"
        echo ""
        
        poetry install --with dev
        
        echo ""
        echo "✓ Poetry installation complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Activate environment: poetry shell"
        echo "  2. Verify installation: make verify"
        echo "  3. Start learning: make learn"
        ;;
        
    2)
        echo ""
        echo "════════════════════════════════════════════════"
        echo "Setting up with pip + venv..."
        echo "════════════════════════════════════════════════"
        echo ""
        
        # Create venv
        python3 -m venv venv
        
        # Activate venv
        source venv/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip setuptools wheel
        
        # Install dependencies
        pip install -r requirements-dev.txt
        
        echo ""
        echo "✓ venv setup complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Activate environment: source venv/bin/activate"
        echo "  2. Verify installation: python -c 'import jax; print(jax.devices())'"
        echo "  3. Start learning: jupyter-lab jax_learn_basics.ipynb"
        ;;
        
    3)
        echo ""
        echo "════════════════════════════════════════════════"
        echo "Setting up with conda..."
        echo "════════════════════════════════════════════════"
        echo ""
        
        if ! command -v conda &> /dev/null; then
            echo "❌ Conda not found. Please install Miniconda or Anaconda first"
            exit 1
        fi
        
        echo "Creating conda environment: jax-playground"
        conda create -n jax-playground python=3.10 -y
        
        echo ""
        echo "Activating environment..."
        eval "$(conda shell.bash hook)"
        conda activate jax-playground
        
        echo ""
        echo "Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements-dev.txt
        
        echo ""
        echo "✓ Conda setup complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Activate environment: conda activate jax-playground"
        echo "  2. Verify installation: python -c 'import jax; print(jax.devices())'"
        echo "  3. Start learning: jupyter-lab jax_learn_basics.ipynb"
        ;;
        
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════"
echo "Setup Verification"
echo "════════════════════════════════════════════════"
echo ""

# Use make verify if Poetry was chosen
if [ "$setup_option" -eq 1 ]; then
    echo "Running make verify..."
    make verify
else
    echo "Verifying installation..."
    python -c "import jax; print('✓ JAX:', jax.__version__)"
    python -c "import flax; print('✓ Flax:', flax.__version__)"
    python -c "import optax; print('✓ Optax:', optax.__version__)"
    echo ""
    python << 'EOF'
import jax
import jax.numpy as jnp
x = jnp.array([1.0, 2.0, 3.0])
f = lambda x: jnp.sum(x**2)
print('✓ JAX working:', jax.grad(f)(x))
EOF
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║              Setup Complete! ✓                 ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "For more information, see SETUP.md"
echo ""
