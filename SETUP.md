# JAX Playground - Environment Setup Guide

This guide helps you set up the development environment for JAX research and learning.

## Prerequisites

- Python 3.10 or 3.11 (recommended 3.10)
- Git
- macOS (as indicated by your system)

## Option 1: Poetry Setup (Recommended ⭐)

Poetry is a modern Python dependency manager that provides better reproducibility and dependency resolution.

### Step 1: Install Poetry

```bash
# On macOS using Homebrew (recommended)
brew install poetry

# Or using pip
curl -sSL https://install.python-poetry.org | python3 -

# Verify installation
poetry --version
```

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd /Users/sandeshkatakam/Documents/local_github_repos/jax-playground

# Install all dependencies (creates virtual environment automatically)
poetry install

# For development with extra tools (testing, linting, formatting)
poetry install --with dev

# Or just production dependencies
poetry install --without dev
```

### Step 3: Activate Virtual Environment

```bash
# Activate poetry virtual environment
poetry shell

# Or run commands without activating
poetry run python script.py
poetry run jupyter notebook
```

### Step 4: Verify Installation

```bash
poetry run python -c "import jax; print(f'JAX version: {jax.__version__}')"
poetry run python -c "import flax; print(f'Flax version: {flax.__version__}')"
poetry run python -c "import jax.numpy as jnp; print(jnp.array([1, 2, 3]))"
```

---

## Option 2: conda/mamba Setup

If you prefer conda for managing Python versions:

```bash
# Create conda environment
conda create -n jax-playground python=3.10

# Activate environment
conda activate jax-playground

# Install dependencies from requirements.txt
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## Option 3: pip with venv (Alternative)

Using Python's built-in venv:

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

---

## macOS-Specific JAX Setup

### CPU-Only (Recommended for Learning)

By default, JAX installs with CPU support:

```bash
poetry install
```

### Metal (macOS GPU) Support

To use Metal acceleration on Apple Silicon (M1/M2/M3):

```bash
# Update jaxlib for Metal backend
pip install --upgrade jax jaxlib -f https://storage.googleapis.com/jax-releases/mac/index.html
```

Or with Poetry, add to `pyproject.toml`:
```toml
[tool.poetry.dependencies]
jaxlib = {version = "^0.4.0", markers = "platform_machine == 'arm64'"}
```

**Note**: For learning purposes, CPU is sufficient and avoids GPU-specific issues.

---

## Environment Management with Poetry

### Common Poetry Commands

```bash
# Add a new dependency
poetry add numpy

# Add a development dependency
poetry add --group dev pytest

# Update all dependencies
poetry update

# Update specific package
poetry update jax

# Check installed packages
poetry show

# Check for outdated packages
poetry show --outdated

# Export requirements.txt format
poetry export -f requirements.txt --output requirements.txt
```

### Poetry Virtual Environment

```bash
# Show virtual environment info
poetry env info

# Show all environments
poetry env list

# Use specific Python version
poetry env use python3.10

# Remove environment
poetry env remove jax-playground-py3.10
```

---

## Verify Full Setup

Run this verification script:

```bash
poetry run python << 'EOF'
import jax
import flax
import optax
import distrax
import tensorflow_probability as tfp
import chex
import scipy
import numpy as np

print("✓ JAX:", jax.__version__)
print("✓ Flax:", flax.__version__)
print("✓ Optax:", optax.__version__)
print("✓ Distrax:", distrax.__version__)
print("✓ TensorFlow Probability:", tfp.__version__)
print("✓ Chex:", chex.__version__)
print("✓ SciPy:", scipy.__version__)
print("✓ NumPy:", np.__version__)

# Test JAX functionality
print("\n✓ JAX Basic Test:")
x = jax.numpy.array([1, 2, 3])
print(f"  Array: {x}")
f = lambda x: jax.numpy.sum(x**2)
print(f"  Gradient of sum(x²): {jax.grad(f)(x)}")

print("\n✓ All dependencies installed successfully!")
EOF
```

---

## IDE Setup

### VS Code

1. Install Python extension
2. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose the Poetry environment
4. Optional: Install Jupyter, Pylance extensions

### PyCharm

1. Go to Settings → Project → Python Interpreter
2. Click ⚙️ → Add Interpreter → Add Local Interpreter
3. Select "Poetry Environment"
4. PyCharm will detect and configure automatically

---

## Library Overview

### Core JAX
- **jax**: Composable transformations (grad, jit, vmap)
- **jaxlib**: Low-level JAX operations
- **equinox**: Functional neural network patterns with JAX

### Optimization & Neural Networks
- **optax**: Gradient-based optimization algorithms
- **flax**: Flexible neural network library built on JAX

### Probabilistic Programming
- **distrax**: JAX distributions library
- **tensorflow-probability**: TensorFlow Probability (has JAX backend)

### Testing & Quality
- **chex**: Testing utilities for JAX
- **pytest**: Unit testing framework
- **hypothesis**: Property-based testing

### Utilities
- **scipy**: Scientific computing (linear algebra, optimization)
- **numpy**: Array operations (complementary to JAX)
- **matplotlib/seaborn**: Visualization
- **jupyter**: Interactive notebooks

---

## Development Workflow

### Running Notebooks

```bash
poetry run jupyter notebook
# or
poetry run jupyter-lab
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_samplers.py

# Run with coverage
poetry run pytest --cov

# Run specific test function
poetry run pytest tests/test_samplers.py::test_mcmc -v
```

### Code Formatting

```bash
# Format code
poetry run black jax_playground/

# Sort imports
poetry run isort jax_playground/

# Lint code
poetry run ruff check jax_playground/

# Type checking
poetry run mypy jax_playground/
```

### Pre-commit Hooks Setup

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

---

## Troubleshooting

### "jaxlib installation issue"
```bash
# Reinstall jaxlib
poetry remove jaxlib
poetry add jaxlib

# Or explicit version
poetry add jaxlib@^0.4.0
```

### "JAX CUDA/GPU not working"
- Verify you installed with correct backend for your system
- Use `jax.devices()` to check available devices
- For learning, CPU is sufficient

### "Memory issues with large arrays"
```python
import jax
# Disable x64 precision for memory
jax.config.update("jax_enable_x64", False)
```

### "Poetry lock issues"
```bash
# Update lock file
poetry lock --no-update

# Or rebuild
rm poetry.lock
poetry install
```

---

## Quick Start

After installation, run your first JAX program:

```bash
poetry run python << 'EOF'
import jax
import jax.numpy as jnp
from jax import grad

# Define a function
def f(x):
    return jnp.sum(x**2)

# Compute gradient
x = jnp.array([1.0, 2.0, 3.0])
print("x:", x)
print("∇f(x):", grad(f)(x))

# JIT compile
f_jit = jax.jit(f)
print("f(x) [compiled]:", f_jit(x))
EOF
```

---

## Next Steps

1. Explore the lesson notebooks: `jax_learn_basics.ipynb`
2. Practice with exercises: `jax_practice_notebook.ipynb`
3. Run the MCMC/sampling code in `jax-playground/mcmc/`
4. Read the JAX documentation: https://jax.readthedocs.io/

---

## Resources

- **JAX Documentation**: https://jax.readthedocs.io/
- **Poetry Documentation**: https://python-poetry.org/docs/
- **Flax Neural Networks**: https://github.com/google/flax
- **Optax Optimization**: https://github.com/deepmind/optax

---

## Support

If you encounter issues:

1. Check JAX version compatibility: `poetry show`
2. Update all packages: `poetry update`
3. Check Python version: `python --version` (should be 3.10+)
4. Review error messages - they're usually helpful!

Happy learning! 🚀
