# Visualization Demonstrations

Graphics rendering and visual analysis tools showcasing PMFlow physics through advanced visualization techniques.

## **graphics/** - 3D Rendering & Ray Tracing

Advanced graphics demonstrations using PMFlow physics for realistic rendering:

### Ray Tracing Implementations

- **`3d_PM_Raytracing_Minimal.py`**: Minimal 3D raytracing with PMFlow refractive index effects
- **`3d_raytracing_with_progress_bar.py`**: Enhanced raytracing with progress tracking for long renders
- **`raytracing0.py`**: Basic raytracing implementation demonstrating core concepts

### Radiosity & Global Illumination

- **`pm_radiosity.py`**: Radiosity rendering incorporating pushing-medium effects on light transport
- **`pm_radiosity_time_stepped.py`**: Time-stepped radiosity simulation showing dynamic light evolution

### Result Visualizations
- **`pm_raytrace_3d.png`**: Sample 3D raytracing output demonstrating PMFlow light bending

## Structure Analysis & Flow Visualization

Tools for analyzing and visualizing the underlying structure of PMFlow systems:

- **`skeleton_and_flow.py`**: Combined skeleton extraction and flow field visualization
- **`skeleton_finder.py`**: Structural analysis tools for identifying key features in PMFlow simulations

## Key Features

### Physics-Based Rendering
- **Refractive Index Integration**: Light rays bend according to PMFlow gravitational refractive index
- **Dynamic Medium**: Time-evolving gravitational fields affect light propagation
- **Realistic Lensing**: Gravitational lensing effects emerge naturally from the physics

### GPU Acceleration
Perfect for testing on Jetson Nano and other CUDA-enabled devices:
- Parallel ray tracing computations
- GPU-accelerated radiosity calculations  
- Real-time structure analysis

### Visualization Techniques
- **3D Ray Tracing**: Volumetric rendering through gravitational media
- **Radiosity**: Global illumination with gravitational light transport
- **Flow Fields**: Vector field visualization of gravitational flow
- **Skeleton Extraction**: Structural analysis of gravitational configurations

## Usage Examples

```bash
# Basic 3D raytracing (CPU)
python graphics/3d_PM_Raytracing_Minimal.py

# GPU-accelerated raytracing with progress tracking
python graphics/3d_raytracing_with_progress_bar.py --gpu

# Interactive radiosity simulation
python graphics/pm_radiosity_time_stepped.py

# Flow structure analysis
python skeleton_and_flow.py
```

## Applications

### Scientific Visualization
- Gravitational field visualization
- Light path analysis in curved spacetime
- Multi-body gravitational system structure

### Computer Graphics Research
- Physics-informed rendering algorithms
- Realistic gravitational lensing effects
- Dynamic lighting with gravitational medium

### Educational Demonstrations
- Visual representation of abstract physics concepts
- Interactive exploration of gravitational effects
- Real-time physics simulation feedback

## Performance Notes

### CPU vs GPU
- CPU implementations: Good for learning and small-scale tests
- GPU implementations: Required for real-time interaction and high-resolution renders

### Jetson Nano Optimization
- Optimized for ARM-based CUDA
- Progressive quality settings for real-time interaction
- Memory-efficient implementations for limited GPU memory

---
*These visualization tools make abstract gravitational physics concepts tangible through advanced computer graphics techniques.*