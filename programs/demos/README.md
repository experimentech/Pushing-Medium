# PMFlow Demonstrations

This directory contains comprehensive demonstrations of the Pushing-Medium (PMFlow) physics model and its applications across multiple domains.

## Directory Structure

### 📚 **physics/** - Core Physics Demonstrations
Foundational demos showcasing the pushing-medium model in various physical scenarios:

- `demo1.py` - `demo7.py`: Progressive introduction to PMFlow concepts
- `pushing_medium.py`: Core PMFlow physics demonstration
- `lagrange_1.py`, `lagrange_2.py`: Lagrange point dynamics with PMFlow
- `grav_lensing.py`: Gravitational lensing using refractive index model
- `plasma_*.py`: Plasma physics demonstrations with PMFlow medium effects

### 🧠 **machine_learning/** - Neural Network & AI Applications
Machine learning implementations incorporating PMFlow dynamics:

#### **BNN/** - Bayesian Neural Networks
- `pmflow_bnn_always_plastic.py`: Always-plastic BNN with PMFlow blocks
- `BNN-Minimal.py`: Minimal BNN implementation
- `pmflow_bnn_prototype.py`: Full prototype BNN with PMFlow integration

#### **CNN/** - Convolutional Neural Networks  
- `mnist-*.py`: MNIST classification with PMFlow layers
- `pytorch_classifier_with_pushing-medium_flow_block.py`: Main PMFlow classifier
- `multi-dataset-training-benchmark.py`: Cross-dataset performance testing
- Visualization outputs: decision boundaries, t-SNE plots, accuracy charts

#### **nn_lib/** - Neural Network Library
Complete PyTorch package for PMFlow-BNN models:
- `pmflow_bnn/`: Core library modules (pmflow.py, factory.py, utils.py, etc.)
- `setup.py`: Package installation script
- `tests/`: Unit tests for library functionality

#### Training & Optimization
- `train_lagrange_gradient_descent.py`: ML training workflows for physics simulations

### 🎨 **visualization/** - Graphics & Rendering
Visual demonstrations and rendering applications:

#### **graphics/** - 3D Rendering & Ray Tracing
- `3d_PM_Raytracing_Minimal.py`: Minimal 3D raytracing with PMFlow
- `3d_raytracing_with_progress_bar.py`: Enhanced raytracing with progress tracking
- `pm_radiosity.py`: Radiosity rendering with pushing-medium effects
- `pm_radiosity_time_stepped.py`: Time-stepped radiosity simulation
- `raytracing0.py`: Basic raytracing implementation
- `pm_raytrace_3d.png`: Sample 3D raytracing output

#### Structure Analysis
- `skeleton_and_flow.py`: Flow structure visualization
- `skeleton_finder.py`: Structural analysis tools

### 📊 **benchmarks/** - Performance & Comparison Testing
Comprehensive testing and comparison frameworks:

#### **comparative/** - Side-by-Side Comparisons
- `nbody_rays_side_by_side.py`: N-body simulation comparisons
- `pushing_medium_nbody_rays.py`: PMFlow-enhanced N-body simulations

#### **showcase/** - Presentation Demos
- `pm_nbody_skeleton_demo.py`: N-body with skeleton visualization
- `pm_nbody_rays_with_skeleton.py`: Combined raytracing and structure analysis
- `pm_switchboard_nbody_rays.py`: Interactive N-body demonstration

#### Comprehensive Tests
- `full_demo.py`: Complete PMFlow demonstration suite
- `full_demo_2.py`: Extended comprehensive demo

### 🗃️ **Supporting Files**
- `data/`: Datasets and input files for demos
- `library/`: Shared utilities and helper functions
- Image files (`.png`): Result visualizations from various demos
- `tt_wave_propagation.mp4`: Video demonstration of wave propagation

## Getting Started

### Prerequisites
```

### Quick Start Recommendations

1. **New to PMFlow?** Start with `physics/demo1.py` through `physics/demo7.py`
2. **Interested in ML?** Try `machine_learning/CNN/mnist-flow.py`
3. **Want visualization?** Run `visualization/graphics/3d_PM_Raytracing_Minimal.py`
4. **Need benchmarks?** Use `benchmarks/full_demo.py`

### GPU Testing (Jetson Nano / CUDA)
For GPU acceleration testing:
1. Start with basic CUDA verification (see root notebook)
2. Run `machine_learning/CNN/pytorch_classifier_with_pushing-medium_flow_block.py`
3. Try `visualization/graphics/3d_raytracing_with_progress_bar.py` for compute-intensive rendering

## Demo Categories Explained

- **Physics**: Pure physics simulations showcasing PMFlow theory
- **Machine Learning**: Neural networks enhanced with PMFlow dynamics  
- **Visualization**: Graphics rendering and visual analysis tools
- **Benchmarks**: Performance testing and comparative analysis

## Legacy Individual Demos

The following standalone physics demonstrations are available in the root directory:

### Core Physics Equations
- **Cosmology**: Distance relations \(D_C(z), D_L, D_A\) with expanding substrate
- **Fluid Dynamics**: 1D substrate continuity and momentum equations
- **Frame Dragging**: Gravito-magnetic spin \(\omega_s(r) = 2GJ/(c^2r^3)\)
- **Lensing**: Fermat travel time with refractive index \(n(r)\)
- **Waves**: TT gravitational wave propagation

### Supporting Materials
- Result images: `Lagrange1.png`, `Lagrange2.png`, cosmology distances, fluid dynamics
- Video: `tt_wave_propagation.mp4` - wave propagation demonstration

## Dependencies
- PyTorch (with CUDA support for GPU demos)
- NumPy, Matplotlib, SciPy
- Additional requirements in individual demo files

## Contributing
See `CONTRIBUTING.md` in the repository root for guidelines on adding new demonstrations.

---
*For questions about specific demos, see individual file docstrings or the main project documentation.*bash
# Install the PMFlow library
pip install -e programs/demos/machine_learning/nn_lib/

# Or install from repository root  
pip install .

# Basic requirements
pip install torch torchvision numpy matplotlib scipy

