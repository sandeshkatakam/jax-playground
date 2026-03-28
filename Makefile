.PHONY: help install install-dev setup test lint format check type clean notebook verify

help:
	@echo "JAX Playground - Project Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install with poetry (base dependencies)"
	@echo "  make install-dev      - Install with poetry (including dev dependencies)"
	@echo "  make setup            - Complete setup guide (same as: make help-setup)"
	@echo "  make verify           - Verify all dependencies are installed"
	@echo ""
	@echo "Development:"
	@echo "  make test             - Run pytest suite"
	@echo "  make lint             - Run linters (black, ruff, mypy)"
	@echo "  make format           - Format code with black and ruff"
	@echo "  make type             - Type check with mypy"
	@echo "  make check            - Run all checks (lint + type)"
	@echo ""
	@echo "Interactive:"
	@echo "  make notebook         - Launch Jupyter Lab"
	@echo "  make jupyter          - Launch Jupyter Notebook"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            - Remove cache and build artifacts"
	@echo "  make update           - Update all dependencies"

# Installation Targets
install:
	@echo "Installing JAX Playground with Poetry (base dependencies)..."
	poetry install --without dev

install-dev:
	@echo "Installing JAX Playground with Poetry (including dev tools)..."
	poetry install

setup:
	@echo "JAX Playground Setup Guide"
	@echo "============================"
	@echo ""
	@echo "Option 1: Poetry Setup (Recommended)"
	@echo "  1. poetry install"
	@echo "  2. poetry shell"
	@echo ""
	@echo "Option 2: pip + venv"
	@echo "  1. python3 -m venv venv"
	@echo "  2. source venv/bin/activate"
	@echo "  3. pip install -r requirements-dev.txt"
	@echo ""
	@echo "Option 3: conda"
	@echo "  1. conda create -n jax-playground python=3.10"
	@echo "  2. conda activate jax-playground"
	@echo "  3. pip install -r requirements-dev.txt"
	@echo ""
	@echo "For detailed instructions, see: SETUP.md"

verify:
	@echo "Verifying JAX Playground environment..."
	@poetry run python -c "import jax; print('✓ JAX:', jax.__version__)"
	@poetry run python -c "import flax; print('✓ Flax:', flax.__version__)"
	@poetry run python -c "import optax; print('✓ Optax:', optax.__version__)"
	@poetry run python -c "import distrax; print('✓ Distrax:', distrax.__version__)"
	@poetry run python -c "import scipy; print('✓ SciPy:', scipy.__version__)"
	@poetry run python -c "import numpy; print('✓ NumPy:', numpy.__version__)"
	@echo ""
	@echo "Testing JAX functionality..."
	@poetry run python << 'EOF'
import jax
import jax.numpy as jnp
x = jnp.array([1.0, 2.0, 3.0])
f = lambda x: jnp.sum(x**2)
print('✓ JAX gradient:', jax.grad(f)(x))
print('✓ All systems operational!')
EOF

# Testing & Quality
test:
	@echo "Running tests with pytest..."
	poetry run pytest -v

test-quick:
	poetry run pytest -q

test-cov:
	poetry run pytest --cov=jax_playground --cov-report=term-missing

lint:
	@echo "Running linters..."
	@echo "Checking with black..."
	poetry run black --check .
	@echo "Checking with ruff..."
	poetry run ruff check .
	@echo "Type checking with mypy..."
	poetry run mypy jax_playground

format:
	@echo "Formatting code..."
	poetry run black .
	poetry run isort .
	poetry run ruff check . --fix

type:
	@echo "Running type checker..."
	poetry run mypy jax_playground

check: lint type
	@echo "✓ All quality checks passed!"

# Interactive Development
notebook:
	poetry run jupyter-lab

jupyter:
	poetry run jupyter notebook

# Maintenance
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned!"

update:
	@echo "Updating dependencies..."
	poetry update

# Quick Commands for Learning
learn:
	@echo "Starting JAX learning environment..."
	poetry run jupyter-lab jax_learn_basics.ipynb

practice:
	@echo "Starting practice notebook..."
	poetry run jupyter-lab jax_practice_notebook.ipynb

exercise: practice