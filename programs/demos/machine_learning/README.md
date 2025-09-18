# Machine Learning Demonstrations

Neural network implementations that incorporate PMFlow (pushing-medium) dynamics into machine learning architectures.

## Directory Structure

### **BNN/** - Bayesian Neural Networks
Bayesian neural networks enhanced with PMFlow dynamics:

- **`pmflow_bnn_always_plastic.py`**: Always-plastic BNN with PMFlow flow blocks
- **`BNN-Minimal.py`**: Minimal BNN implementation for learning PMFlow concepts
- **`pmflow_bnn_prototype.py`**: Full prototype implementation with comprehensive PMFlow integration
- Supporting files: patches, LaTeX documentation

### **CNN/** - Convolutional Neural Networks
CNN implementations with PMFlow layers and comparative studies:

- **`pytorch_classifier_with_pushing-medium_flow_block.py`**: Main PMFlow-enhanced classifier
- **`mnist-flow.py`**: MNIST classification with PMFlow layers
- **`mnist-extended.py`**: Extended MNIST experiments
- **`mnist-benchmark2.py`**: Performance benchmarking
- **`multi-dataset-training-benchmark.py`**: Cross-dataset performance analysis
- **`nn-benchmark-oneswitch.py`**: Neural network switching benchmarks
- Result visualizations: decision boundaries, t-SNE plots, accuracy charts

### **nn_lib/** - PMFlow Neural Network Library
Complete PyTorch package for PMFlow-enhanced neural networks:

#### Core Library (`pmflow_bnn/`)
- **`pmflow.py`**: Core PMFlow layer implementations
- **`bnn.py`**: Bayesian neural network components
- **`factory.py`**: Model factory functions
- **`utils.py`**: Utility functions and helpers
- **`baselines.py`**: Baseline model implementations
- **`version.py`**: Version management

#### Package Infrastructure
- **`setup.py`**: Package installation script
- **`__init__.py`**: Package initialization
- **`tests/`**: Unit tests for library functionality
- **`README.md`**: Library-specific documentation

### **Training Scripts**

- **`train_lagrange_gradient_descent.py`**: ML training workflows for physics simulations combining gradient descent with Lagrange point dynamics

## Getting Started

### Installation
```bash
# Install the PMFlow neural network library
cd programs/demos/machine_learning/nn_lib/
pip install -e .

# Or from repository root
pip install .
```

### Quick Start

1. **BNN Beginners**: Start with `BNN/BNN-Minimal.py`
2. **CNN Users**: Try `CNN/mnist-flow.py`
3. **Advanced Research**: Use `CNN/pytorch_classifier_with_pushing-medium_flow_block.py`
4. **Library Development**: Explore `nn_lib/pmflow_bnn/`

### GPU Acceleration

For CUDA/GPU testing (especially useful on Jetson Nano):
```bash
# Test basic GPU functionality first
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run GPU-accelerated PMFlow training
python CNN/pytorch_classifier_with_pushing-medium_flow_block.py --gpu
```

## Key Innovations

### PMFlow Flow Blocks
Neural network layers that incorporate pushing-medium dynamics:
- Refractive index-based transformations
- Gravitational flow modeling in feature space
- Physics-informed activation functions

### Bayesian Integration
Uncertainty quantification enhanced by PMFlow:
- Probabilistic gravitational effects
- Uncertainty propagation through medium
- Physics-constrained priors

### Benchmarking Framework
Comprehensive comparison tools:
- PMFlow vs traditional architectures
- Multi-dataset evaluation
- Decision boundary analysis
- Feature space visualization (t-SNE)

## Research Applications

- **Physics-Informed ML**: Neural networks that respect gravitational physics
- **Uncertainty Quantification**: Bayesian methods with physical constraints
- **Feature Learning**: Discovery of gravitational structures in data
- **Hybrid Modeling**: Combining traditional ML with physics simulations

## Results & Visualizations

The CNN directory contains extensive result visualizations:
- Decision boundary comparisons (baseline vs PMFlow-trained vs PMFlow-swapped)
- t-SNE feature space analysis
- Training accuracy progression charts
- Cross-dataset generalization studies

---
*These implementations demonstrate how fundamental physics can enhance machine learning architectures and provide new approaches to physics-informed AI.*