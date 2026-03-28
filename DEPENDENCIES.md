# JAX Playground - Dependencies Overview

This document describes all dependencies used in the JAX Playground project and why they're included.

## Core JAX Ecosystem

### jax & jaxlib
- **Purpose**: Core JAX library for composable function transformations and GPU/TPU support
- **Usage**: Automatic differentiation, JIT compilation, vectorization (vmap), random number generation
- **Key Functions**: `jax.grad`, `jax.jit`, `jax.vmap`, `jax.pmap`
- **Documentation**: https://jax.readthedocs.io/

### equinox
- **Purpose**: Functional neural network library with JAX philosophy
- **Usage**: Building models that play nicely with JAX transformations
- **Advantage over Flax**: Simpler, more Pythonic, better for research code
- **Documentation**: https://github.com/patrick-kidger/equinox

## Neural Networks & Optimization

### flax
- **Purpose**: Flexible neural network library built on JAX (by Google)
- **Usage**: Building complex deep learning models, variable management
- **Key Components**: `flax.linen` (neural network definitions), `flax.core` (runtime)
- **Use When**: Building production neural networks, need Flax ecosystem
- **Documentation**: https://flax.readthedocs.io/

### optax
- **Purpose**: Composable gradient-based optimization algorithms
- **Usage**: Training neural networks, general optimization
- **Key Algorithms**: SGD, Adam, RMSprop, learning rate scheduling
- **Advantage**: Works perfectly with JAX's functional paradigm
- **Documentation**: https://optax.readthedocs.io/

## Probabilistic Programming & Distributions

### distrax
- **Purpose**: JAX distributions library (from DeepMind)
- **Usage**: Sampling from distributions, computing log probabilities
- **Key Classes**: Normal, Uniform, Beta, Categorical, etc.
- **Advantage**: Native JAX support, composes with transformations
- **Documentation**: https://github.com/google-deepmind/distrax

### tensorflow-probability
- **Purpose**: Probabilistic programming library (TensorFlow project)
- **JAX Support**: Full JAX backend support via `tfp.substrates.jax`
- **Usage**: Advanced distributions, probabilistic models
- **Key Components**: `tfp.substrates.jax.distributions`, bijectors, variational inference
- **Advantage**: More distributions than Distrax, well-maintained
- **Documentation**: https://www.tensorflow.org/probability

### chex
- **Purpose**: Testing utilities and assertion library for JAX
- **Usage**: Assert JAX array properties (shape, dtype, device), snapshot testing
- **Key Features**: `chex.assert_trees_all_close`, `chex.assert_shape`
- **Important for**: Debugging transformations and numerical issues
- **Documentation**: https://github.com/google-deepmind/chex

## Linear Algebra & Scientific Computing

### scipy
- **Purpose**: Scientific computing library with JAX integration
- **Usage**: Linear algebra (beyond NumPy), optimization, statistics
- **Key Submodules**: 
  - `scipy.linalg`: Advanced linear algebra
  - `scipy.optimize`: Root finding, minimization
  - `scipy.special`: Special functions (logsumexp, gammainc)
  - `scipy.stats`: Statistical functions
- **JAX Compatibility**: `jax.scipy` namespace provides compatible versions
- **Documentation**: https://docs.scipy.org/

### numpy
- **Purpose**: Base array operations
- **Usage**: Array creation, manipulation, basic operations
- **Relationship to JAX**: JAX arrays are immutable versions of NumPy arrays
- **Note**: Use `jax.numpy` for JAX code, regular `numpy` for data prep
- **Documentation**: https://numpy.org/

## Visualization & Data Handling

### matplotlib
- **Purpose**: Plotting and visualization library
- **Usage**: Plotting loss curves, visualizing samples, debugging
- **Key Interface**: pyplot (similar to MATLAB)
- **Documentation**: https://matplotlib.org/

### seaborn
- **Purpose**: Statistical data visualization (built on matplotlib)
- **Usage**: Beautiful statistical plots, heatmaps
- **Common Use**: Visualization during research/exploration
- **Documentation**: https://seaborn.pydata.org/

## Interactive & Development

### jupyter & jupyter-lab
- **Purpose**: Interactive computing environment
- **jupyter-lab**: Modern web-based IDE with notebook support
- **Usage**: Exploratory research, teaching, prototyping
- **Advantage over Notebook**: Better UI, terminal, file editor integrated
- **Documentation**: https://jupyter.org/

### ipython
- **Purpose**: Enhanced Python interactive shell
- **Usage**: Better REPL experience, debugging
- **Integration**: Used by Jupyter under the hood
- **Features**: Magic commands (`%timeit`, `%run`), shell integration

### ipykernel
- **Purpose**: IPython kernel for Jupyter
- **Usage**: Enables Python execution in Jupyter notebooks

## Development & Quality Assurance

### pytest
- **Purpose**: Testing framework for Python
- **Usage**: Unit tests, integration tests
- **Features**: Fixtures, parametrization, plugins, coverage integration
- **Documentation**: https://pytest.org/

### hypothesis
- **Purpose**: Property-based testing library
- **Usage**: Generate random test inputs to find edge cases
- **Advantage**: Finds bugs in specifications, not just implementation
- **Documentation**: https://hypothesis.readthedocs.io/

### pytest-cov
- **Purpose**: Code coverage reporting for pytest
- **Usage**: Measure how much code is tested

### black
- **Purpose**: Code formatter (PEP 8 compliant)
- **Usage**: Automatic code formatting for consistency
- **Philosophy**: "Uncompromising" - no configuration options
- **Documentation**: https://black.readthedocs.io/

### ruff
- **Purpose**: Fast Python linter (replacement for flake8/pylint)
- **Usage**: Check for code style issues, potential bugs
- **Speed**: 10-100x faster than traditional linters
- **Documentation**: https://beta.ruff.rs/

### mypy
- **Purpose**: Static type checker for Python
- **Usage**: Find type-related bugs before runtime
- **JAX Support**: Improving, some JAX code requires type ignores
- **Documentation**: https://www.mypy-lang.org/

### isort
- **Purpose**: Import statement formatter
- **Usage**: Organize and sort imports consistently
- **Integration**: Works with black

### pre-commit
- **Purpose**: Git hook framework for running checks before commits
- **Usage**: Automated code quality checks on every commit
- **Setup**: `pre-commit install` in repo
- **Documentation**: https://pre-commit.com/

## Optional: Documentation

### sphinx
- **Purpose**: Documentation generation
- **Usage**: Build API documentation from source
- **Common in**: Academic and library projects

### sphinx-rtd-theme
- **Purpose**: Popular theme for Sphinx documentation
- **Usage**: Professional-looking documentation

## Optional: Debugging

### ipdb
- **Purpose**: IPython debugger (enhanced Python debugger)
- **Usage**: `import ipdb; ipdb.set_trace()` or `%debug` in IPython
- **Advantage**: Better UI than standard pdb

### tqdm
- **Purpose**: Progress bar library
- **Usage**: Show progress for long-running operations
- **Common**: Used in training loops, data processing

## Dependency Graph

```
jax-playground
├── Core Computation
│   ├── jax
│   ├── jaxlib
│   └── numpy
├── Neural Networks
│   ├── flax
│   ├── equinox
│   └── optax
├── Probabilistic
│   ├── distrax
│   ├── tensorflow-probability
│   ├── scipy
│   └── chex
├── Visualization
│   ├── matplotlib
│   └── seaborn
├── Development
│   ├── jupyter
│   ├── jupyter-lab
│   ├── ipython
│   └── ipykernel
└── Quality (dev-only)
    ├── pytest
    ├── pytest-cov
    ├── hypothesis
    ├── black
    ├── ruff
    ├── mypy
    ├── isort
    ├── pre-commit
    └── sphinx
```

## Minimum Installation

If you only need core JAX functionality:

```bash
pip install jax jaxlib numpy
```

## Recommended Installation

For learning and research:

```bash
pip install -r requirements.txt
```

## Full Installation (including development tools)

```bash
pip install -r requirements-dev.txt
# or
poetry install --with dev
```

## Library Selection Rationale

### Why Flax AND Equinox?
- **Flax**: Industry standard, many examples, production use
- **Equinox**: Simpler, more functional, better for research algorithms

### Why Multiple Distribution Libraries?
- **Distrax**: Lightweight, pure JAX
- **TensorFlow Probability**: More distributions, more mature, more examples

### Why scipy?
- Numerical computing beyond NumPy
- Linear algebra operations (SVD, QR decomposition)
- Special functions like logsumexp for numerical stability

## Common Installation Issues

### "jaxlib installation fails"
- Ensure you have Python 3.10+ 
- On M1/M2 Macs, may need `pip install jax jaxlib -f https://storage.googleapis.com/jax-releases/mac/index.html`

### "TensorFlow Probability doesn't work with JAX"
- Ensure you import from `tfp.substrates.jax` not main `tfp`
- Example: `import tensorflow_probability.substrates.jax as tfp`

### "Can't install scipy"
- May need to install pre-built wheel
- `pip install --only-binary :all: scipy`

## Updating Dependencies

With Poetry:
```bash
poetry update              # Update all
poetry update jax optax    # Update specific
```

With pip:
```bash
pip install --upgrade -r requirements.txt
```

## Version Pinning

Current versions (as of package creation):
- JAX: 0.4.0+
- Flax: 0.7.0+
- Optax: 0.1.7+
- SciPy: 1.11.0+
- NumPy: 1.24.0+
- Python: 3.10-3.11

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Repository](https://github.com/deepmind/optax)
- [PyTorch User?: JAX for PyTorch users](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [Thinking in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
