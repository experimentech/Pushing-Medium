# Benchmark Demonstrations

Performance testing and comparative analysis frameworks for PMFlow implementations.

## **comparative/** - Side-by-Side Comparisons

Direct comparisons between PMFlow and traditional physics implementations:

- **`nbody_rays_side_by_side.py`**: N-body simulation comparisons showing PMFlow vs Newtonian dynamics side-by-side
- **`pushing_medium_nbody_rays.py`**: PMFlow-enhanced N-body simulations with ray tracing visualization

## **showcase/** - Presentation Demos

Polished demonstrations designed for presentations and public showcasing:

- **`pm_nbody_skeleton_demo.py`**: N-body simulation with skeleton structure visualization
- **`pm_nbody_rays_with_skeleton.py`**: Combined raytracing and structure analysis demonstration
- **`pm_switchboard_nbody_rays.py`**: Interactive N-body demonstration with parameter switching

## Comprehensive Test Suites

Full-scale demonstrations covering multiple PMFlow aspects:

- **`full_demo.py`**: Complete PMFlow demonstration suite covering core physics concepts
- **`full_demo_2.py`**: Extended comprehensive demo with additional advanced features

## Benchmark Categories

### Performance Benchmarks
- **Computational Speed**: PMFlow vs traditional gravity calculations
- **Memory Usage**: Resource efficiency comparisons
- **Scalability**: Performance with increasing particle counts
- **GPU Acceleration**: CUDA speedup analysis

### Physics Accuracy
- **Orbital Mechanics**: Precision of PMFlow orbital predictions
- **Multi-Body Dynamics**: Complex gravitational system accuracy
- **Energy Conservation**: Long-term stability analysis
- **Gravitational Lensing**: Light deflection precision

### Visualization Quality
- **Rendering Performance**: Graphics rendering speed comparisons
- **Visual Fidelity**: Quality of gravitational effects visualization
- **Real-Time Interaction**: Interactive performance metrics

## Usage

### Running Comparisons
```bash
# Side-by-side N-body comparison
python comparative/nbody_rays_side_by_side.py

# Interactive showcase demo
python showcase/pm_switchboard_nbody_rays.py

# Full comprehensive test
python full_demo.py
```

### Benchmark Parameters
Most benchmarks accept command-line parameters:
```bash
# Specify particle count for N-body simulations
python comparative/nbody_rays_side_by_side.py --particles 1000

# Enable GPU acceleration
python showcase/pm_nbody_rays_with_skeleton.py --gpu

# Set time duration for simulations
python full_demo.py --duration 100 --timestep 0.01
```

## Key Metrics

### Performance Metrics
- **Computation Time**: Time per simulation step
- **Memory Allocation**: Peak and average memory usage
- **GPU Utilization**: CUDA core usage efficiency
- **Throughput**: Particles processed per second

### Accuracy Metrics
- **Energy Drift**: Long-term energy conservation
- **Orbital Stability**: Deviation from expected orbits
- **Phase Space Volume**: Liouville theorem compliance
- **Convergence**: Solution stability with varying timesteps

### Visual Metrics
- **Frame Rate**: Real-time visualization performance
- **Rendering Quality**: Visual fidelity measurements
- **Interaction Latency**: Response time for parameter changes

## Comparative Studies

### PMFlow vs Newtonian
Direct comparison of:
- Computational complexity
- Physical accuracy
- Visual representation quality
- Educational value

### Implementation Variants
Testing different PMFlow implementations:
- Direct refractive index calculation
- Approximate methods for speed
- Hybrid approaches
- GPU vs CPU performance

## Applications

### Research Validation
- Verify PMFlow implementation correctness
- Compare with established physics simulations
- Validate new features and optimizations

### Performance Optimization
- Identify computational bottlenecks
- Test GPU acceleration effectiveness
- Optimize for specific hardware (Jetson Nano, etc.)

### Educational Assessment
- Demonstrate physics concept clarity
- Show computational trade-offs
- Illustrate real-world performance considerations

---
*These benchmarks provide quantitative validation of PMFlow implementations and guide optimization efforts for different use cases.*