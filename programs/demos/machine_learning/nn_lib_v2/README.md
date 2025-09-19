# PMFlow BNN v0.2.0 - Temporal Parallelism Enhancement

Enhanced implementation of PMFlow Biological Neural Networks with temporal parallelism, vectorized operations, and embarrassingly parallel scaling.

## New Features in v0.2.0

### Temporal Parallelism
- **ParallelPMField**: Vectorized PMFlow implementation with batch gradient calculations
- **TemporalPipelineBNN**: Pipeline overlapping for temporal dynamics
- **AdaptiveScheduler**: Hardware-aware optimization and scheduling

### Vectorized Operations
- **VectorizedLateralEI**: Memory-efficient lateral excitation-inhibition
- **Vectorized Plasticity**: Batch plasticity updates with Hebbian learning
- **Memory Optimization**: Chunked computation for large batch processing

### Multi-GPU Support
- **MultiGPUPMBNN**: Distributed PMFlow computation across multiple GPUs
- **Embarrassingly Parallel**: Independent PMFlow centers like gravitational point masses
- **Adaptive Scaling**: Linear scaling until bandwidth limits

### Hardware Optimization
- **Jetson Nano Support**: Optimized for NVIDIA Tegra X1 with memory constraints
- **Auto-Configuration**: Automatic hardware detection and optimization
- **Performance Profiling**: Built-in benchmarking and validation tools

## Physics Foundation

PMFlow neural networks implement the core Pushing-Medium gravitational equations:

```
Refractive Index: n(r) = 1 + Σμᵢ/|r-rᵢ|
Gradient: ∇ln(n) = -Σ(μᵢ/|r-rᵢ|³)(r-rᵢ) / n_total  
Flow Acceleration: a = -c²∇ln(n)
```

Where μᵢ = 2GMᵢ/c² represents gravitational strength parameters, and each PMFlow center acts as an independent gravitational point mass enabling embarrassingly parallel computation.

## Quick Start

```python
import torch
from pmflow_bnn import get_model_v2, get_performance_config, benchmark_temporal_parallelism

# Auto-detect hardware and get optimized configuration
config = get_performance_config('auto')
print(f"Detected hardware profile: {config}")

# Create temporal pipeline model
model = get_model_v2('temporal_pipeline', **config)

# Benchmark performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = benchmark_temporal_parallelism(model, [16, 32, 64], device)
print(f"Temporal parallelism results: {results}")

# Training example
x = torch.randn(32, 28*28)  # MNIST-like input
logits, (z, h) = model(x, T=5)  # 5 temporal steps
```

## Model Types

### TemporalPipelineBNN
Enhanced BNN with pipeline overlapping and adaptive scheduling:
```python
model = get_model_v2('temporal_pipeline', 
                    temporal_stages=3,
                    pipeline_overlap=True,
                    adaptive_scheduling=True)
```

### MultiGPUPMBNN  
Distributed computation across multiple GPUs:
```python
model = get_model_v2('multi_gpu', 
                    gpu_devices=[0, 1, 2, 3],
                    n_centers=128)
```

### PMBNNAlwaysPlasticV2
Continuous adaptation with vectorized plasticity:
```python
model = get_model_v2('always_plastic_v2',
                    plastic=True,
                    plasticity_lr=1e-3)
```

## Hardware Profiles

The library automatically optimizes for different hardware:

- **jetson_nano**: NVIDIA Tegra X1 with memory constraints (16-32 centers)
- **single_gpu**: Standard GPU with pipeline parallelism (64 centers)
- **multi_gpu**: Multiple GPUs with distributed computation (128+ centers)  
- **cpu**: CPU-optimized with reduced complexity (24 centers)

## Meaningful Evaluation (Not "Faster than MLP"!)

**IMPORTANT**: PMFlow BNN is not about being "faster than standard MLPs" - it provides entirely new capabilities that don't exist in conventional neural networks.

### Core PMFlow Capabilities to Evaluate:

1. **Embarrassingly Parallel Scaling** - The real performance metric
2. **Gravitational Center Dynamics** - Physics-based specialization  
3. **Biological Plasticity** - Adaptive learning mechanisms
4. **Temporal Parallelism** - Pipeline processing capabilities

### Proper Evaluation:
```python
from pmflow_bnn.evaluation import PMFlowEvaluator

# Create evaluator for meaningful metrics
evaluator = PMFlowEvaluator()

# Test 1: Embarrassingly parallel scaling (THE key metric)
scaling_results = evaluator.evaluate_embarrassingly_parallel_scaling(model)
print(f"Peak scaling efficiency: {scaling_results['peak_efficiency']:.1f}x")
print(f"Is embarrassingly parallel: {scaling_results['is_embarrassingly_parallel']}")

# Test 2: Gravitational center dynamics
dynamics_results = evaluator.evaluate_gravitational_dynamics(model, test_data, test_labels)
print(f"Center specialization ratio: {dynamics_results['specialization_ratio']:.2f}x")

# Test 3: Biological plasticity
plasticity_results = evaluator.evaluate_biological_plasticity(model, train_data, train_labels, shifting_datasets)
print(f"Plasticity score: {plasticity_results['plasticity_score']:.3f}")

# Generate comprehensive report
report = evaluator.generate_report()
print(report)

# Create meaningful visualizations
evaluator.visualize_results()
```

### What NOT to Compare:
❌ **PMFlow BNN vs Simple MLP speed** - Like comparing a spacecraft to a bicycle  
❌ **Raw forward pass time** - Misses the point entirely  
❌ **Parameter count comparisons** - Different architectures, different purposes  

### What TO Compare:
✅ **Scaling efficiency with batch size** - The embarrassingly parallel advantage  
✅ **Gravitational specialization dynamics** - Physics-based adaptation  
✅ **Biological plasticity capabilities** - Adaptive learning mechanisms  
✅ **Different PMFlow configurations** - Temporal stages, center counts, etc.

The value proposition is **NEW CAPABILITIES**, not raw speed!

## Architecture Overview

The v0.2.0 architecture implements three levels of parallelism:

1. **Spatial Parallelism**: Independent PMFlow centers (embarrassingly parallel)
2. **Temporal Parallelism**: Pipeline overlapping across time steps  
3. **Device Parallelism**: Multi-GPU distribution of gravitational centers

This mirrors the physical reality that gravitational fields from point masses superpose linearly, making PMFlow centers naturally independent and scalable.

## Installation

```bash
cd nn_lib_v2
pip install -e .

# For development
pip install -e ".[dev,benchmark]"

# For CUDA support  
pip install -e ".[cuda]"
```

## Backward Compatibility

v0.2.0 maintains API compatibility with v0.1.0 while providing enhanced performance. Existing code will work with minimal changes:

```python
# v0.1.0 style (still works)
from pmflow_bnn import PMBNN
model = PMBNN()

# v0.2.0 style (enhanced)  
from pmflow_bnn import get_model_v2
model = get_model_v2('temporal_pipeline')
```

## License

MIT License - see LICENSE file for details.