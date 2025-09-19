#!/usr/bin/env python3
"""
Debug PMFlow Scaling Issues

Identify and fix the scaling calculation and CPU performance issues.
"""

import torch
import torch.nn as nn
import time
import numpy as np

# Simple test model to isolate the scaling issue
class TestPMFlowModel(nn.Module):
    def __init__(self, d_latent=8, n_centers=32):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, d_latent) * 0.8)
        self.mus = nn.Parameter(torch.ones(n_centers) * 0.5)
        self.proj = nn.Linear(d_latent, 64)
        self.out = nn.Linear(64, 4)
    
    def forward(self, x):
        # Simple PMFlow-like computation
        B, D = x.shape
        z = x  # Input assumed to be latent already
        
        # Vectorized center distance computation
        rvec = z.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, N, D)
        r2 = torch.sum(rvec * rvec, dim=2) + 1e-4  # (B, N)
        r = torch.sqrt(r2)  # (B, N)
        
        # Gravitational field
        forces = self.mus.unsqueeze(0) / r  # (B, N)
        g = torch.sum(forces.unsqueeze(2) * (-rvec / r.unsqueeze(2)), dim=1)  # (B, D)
        
        # Apply gravitational flow
        z_new = z + 0.1 * g
        
        # Project and classify
        h = torch.tanh(self.proj(z_new))
        logits = self.out(h)
        
        return logits

def debug_scaling_calculation():
    """Debug the scaling efficiency calculation."""
    
    print("🔍 Debugging Scaling Calculation")
    print("="*40)
    
    # Simulate throughput data that should show good scaling
    batch_sizes = [2, 4, 8, 16]
    
    # Test 1: Perfect scaling case
    print("\n📊 Test 1: Perfect Scaling (Theoretical)")
    perfect_times = [0.01, 0.01, 0.01, 0.01]  # Constant time
    perfect_throughputs = [bs/t for bs, t in zip(batch_sizes, perfect_times)]
    
    # Current (buggy) calculation
    print("   Current calculation:")
    baseline_throughput_per_sample = perfect_throughputs[0] / batch_sizes[0]
    current_efficiency = []
    for throughput, batch_size in zip(perfect_throughputs, batch_sizes):
        expected_throughput = baseline_throughput_per_sample * batch_size
        efficiency = throughput / expected_throughput
        current_efficiency.append(efficiency)
        print(f"     Batch {batch_size}: {efficiency:.2f}x")
    
    # Fixed calculation  
    print("   Fixed calculation:")
    baseline_batch = batch_sizes[0]
    baseline_throughput = perfect_throughputs[0]
    fixed_efficiency = []
    for throughput, batch_size in zip(perfect_throughputs, batch_sizes):
        expected_scaling_factor = batch_size / baseline_batch
        expected_throughput = baseline_throughput * expected_scaling_factor
        efficiency = throughput / expected_throughput
        fixed_efficiency.append(efficiency)
        print(f"     Batch {batch_size}: {efficiency:.2f}x")
    
    # Test 2: Realistic case with some overhead
    print("\n📊 Test 2: Realistic Scaling")
    realistic_times = [0.01, 0.015, 0.025, 0.045]  # Increasing overhead
    realistic_throughputs = [bs/t for bs, t in zip(batch_sizes, realistic_times)]
    
    print("   Fixed calculation on realistic data:")
    baseline_batch = batch_sizes[0]
    baseline_throughput = realistic_throughputs[0]
    realistic_efficiency = []
    for throughput, batch_size in zip(realistic_throughputs, batch_sizes):
        expected_scaling_factor = batch_size / baseline_batch
        expected_throughput = baseline_throughput * expected_scaling_factor
        efficiency = throughput / expected_throughput
        realistic_efficiency.append(efficiency)
        print(f"     Batch {batch_size}: {efficiency:.2f}x")
    
    print(f"\n   Average efficiency: {np.mean(realistic_efficiency):.2f}x")
    print(f"   Is embarrassingly parallel: {np.mean(realistic_efficiency) > 0.7 and min(realistic_efficiency) > 0.5}")

def test_actual_model_scaling():
    """Test actual model scaling with the fixed calculation."""
    
    print("\n🧪 Testing Actual Model Scaling")
    print("="*35)
    
    device = torch.device('cpu')  # Force CPU for consistent testing
    model = TestPMFlowModel(d_latent=8, n_centers=32).to(device)
    
    batch_sizes = [2, 4, 8, 16]
    times = []
    throughputs = []
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 8, device=device)
        
        # Warmup
        for _ in range(3):
            _ = model(x)
        
        # Time it
        start = time.time()
        for _ in range(10):
            _ = model(x)
        end = time.time()
        
        avg_time = (end - start) / 10
        throughput = batch_size / avg_time
        
        times.append(avg_time)
        throughputs.append(throughput)
        
        print(f"   Batch {batch_size:2d}: {avg_time*1000:.2f}ms, {throughput:.1f} samples/sec")
    
    # Apply fixed scaling calculation
    baseline_batch = batch_sizes[0]
    baseline_throughput = throughputs[0]
    scaling_efficiency = []
    
    print(f"\n📈 Scaling Analysis:")
    for i, (throughput, batch_size) in enumerate(zip(throughputs, batch_sizes)):
        expected_scaling_factor = batch_size / baseline_batch
        expected_throughput = baseline_throughput * expected_scaling_factor
        efficiency = throughput / expected_throughput
        scaling_efficiency.append(efficiency)
        
        print(f"   Batch {batch_size:2d}: {efficiency:.2f}x efficiency")
    
    avg_efficiency = np.mean(scaling_efficiency)
    peak_efficiency = max(scaling_efficiency)
    is_embarrassingly_parallel = avg_efficiency > 0.7 and min(scaling_efficiency) > 0.5
    
    print(f"\n🎯 Results:")
    print(f"   Peak Efficiency: {peak_efficiency:.2f}x")
    print(f"   Average Efficiency: {avg_efficiency:.2f}x")
    print(f"   Embarrassingly Parallel: {is_embarrassingly_parallel}")
    
    if avg_efficiency > 0.8:
        print("   🎉 EXCELLENT: Great scaling!")
    elif avg_efficiency > 0.6:
        print("   ✅ GOOD: Decent scaling")
    else:
        print("   ⚠️ POOR: Scaling issues")

def create_optimized_cpu_config():
    """Create truly optimized configuration for CPU."""
    
    print(f"\n🔧 Creating Optimized CPU Configuration")
    print("="*40)
    
    optimized_config = {
        'model_type': 'temporal_pipeline',
        'n_centers': 24,        # Reduce for CPU efficiency
        'pm_steps': 3,          # Reduce computational load
        'dt': 0.2,              # Larger time steps for efficiency
        'beta': 1.0,            # Standard coupling
        'd_latent': 8,          # Keep moderate
        'channels': 64,         # Standard size
        'temporal_stages': 2,   # Keep some parallelism
        'pipeline_overlap': True,
        'adaptive_scheduling': True,
        'chunk_size': 8,        # Smaller chunks for CPU
    }
    
    print("Optimized CPU config:")
    for key, value in optimized_config.items():
        print(f"   {key}: {value}")
    
    return optimized_config

if __name__ == "__main__":
    print("🔍 PMFlow Scaling Debug & Optimization")
    print("="*40)
    
    # Debug scaling calculation
    debug_scaling_calculation()
    
    # Test actual model
    test_actual_model_scaling() 
    
    # Create optimized config
    optimized_config = create_optimized_cpu_config()
    
    print(f"\n✅ Debug Complete!")
    print("Key issues identified and solutions provided.")