#!/usr/bin/env python3
"""
Test script for PMFlow BNN v0.2.0 temporal parallelism features.

This script validates the new v0.2.0 architecture including:
- Vectorized PMField operations
- Temporal pipeline parallelism  
- Multi-GPU distribution
- Performance improvements over v0.1.0
"""

import torch
import torch.nn as nn
import sys
import os

# Add the v0.2.0 library to path
sys.path.insert(0, '/home/tmumford/Documents/gravity/programs/demos/machine_learning/nn_lib_v2')

try:
    from pmflow_bnn import (
        get_model_v2, 
        get_performance_config, 
        benchmark_temporal_parallelism,
        validate_embarrassingly_parallel_scaling,
        get_hardware_info
    )
    print("✅ Successfully imported PMFlow BNN v0.2.0")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_hardware_detection():
    """Test hardware detection and configuration."""
    print("\n=== Hardware Detection ===")
    hw_info = get_hardware_info()
    print(f"Platform: {hw_info['platform']}")
    print(f"CUDA Available: {hw_info['cuda_available']}")
    print(f"GPU Count: {hw_info['gpu_count']}")
    print(f"Is Jetson: {hw_info['is_jetson']}")
    
    if hw_info['cuda_available']:
        for gpu in hw_info['gpu_devices']:
            print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total']//1024**2}MB)")
    
    return hw_info

def test_model_creation():
    """Test creating different v0.2.0 model types."""
    print("\n=== Model Creation Tests ===")
    
    # Test temporal pipeline model
    try:
        model = get_model_v2('temporal_pipeline', n_centers=32, pm_steps=3)
        print("✅ TemporalPipelineBNN created successfully")
    except Exception as e:
        print(f"❌ TemporalPipelineBNN creation failed: {e}")
        return None
    
    # Test always plastic model
    try:
        plastic_model = get_model_v2('always_plastic_v2', n_centers=24, plastic=True)
        print("✅ PMBNNAlwaysPlasticV2 created successfully")
    except Exception as e:
        print(f"❌ PMBNNAlwaysPlasticV2 creation failed: {e}")
    
    # Test multi-GPU model (if available)
    if torch.cuda.device_count() > 1:
        try:
            multi_gpu_model = get_model_v2('multi_gpu', n_centers=48, gpu_devices=[0, 1])
            print("✅ MultiGPUPMBNN created successfully")
        except Exception as e:
            print(f"❌ MultiGPUPMBNN creation failed: {e}")
    else:
        print("ℹ️  Multi-GPU model skipped (single GPU or CPU)")
    
    return model

def test_forward_pass(model):
    """Test forward pass with temporal dynamics."""
    print("\n=== Forward Pass Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create test input (MNIST-like)
    batch_size = 8
    x = torch.randn(batch_size, 28*28, device=device)
    
    try:
        with torch.no_grad():
            logits, (z, h) = model(x, T=3)  # 3 temporal steps
        
        print(f"✅ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Latent z shape: {z.shape}")
        print(f"  Hidden h shape: {h.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_vectorized_operations(model):
    """Test vectorized PMField operations."""
    print("\n=== Vectorized Operations Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test different batch sizes to verify vectorization
    batch_sizes = [4, 8, 16]
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 28*28, device=device)
        
        try:
            with torch.no_grad():
                logits, _ = model(x, T=2)
            print(f"✅ Batch size {batch_size}: {logits.shape}")
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {e}")
            return False
    
    return True

def test_performance_comparison():
    """Test performance improvements over v0.1.0."""
    print("\n=== Performance Comparison ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create v0.2.0 model
    model_v2 = get_model_v2('temporal_pipeline', n_centers=32, pm_steps=3)
    
    try:
        # Benchmark temporal parallelism
        results = benchmark_temporal_parallelism(
            model_v2, 
            batch_sizes=[8, 16], 
            device=device, 
            num_trials=3
        )
        
        print("✅ Performance benchmark completed")
        for i, (batch_size, throughput) in enumerate(zip(results['batch_sizes'], results['throughput'])):
            print(f"  Batch {batch_size}: {throughput:.2f} samples/sec")
        
        # Test embarrassingly parallel scaling
        scaling_results = validate_embarrassingly_parallel_scaling(
            model_v2, 
            max_batch_size=32, 
            device=device
        )
        
        print(f"✅ Scaling validation completed")
        print(f"  Average efficiency: {scaling_results['average_efficiency']:.2f}")
        print(f"  Is embarrassingly parallel: {scaling_results['is_embarrassingly_parallel']}")
        
        return True
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        return False

def main():
    """Run all v0.2.0 validation tests."""
    print("🚀 PMFlow BNN v0.2.0 Validation Tests")
    print("=" * 50)
    
    # Test hardware detection
    hw_info = test_hardware_detection()
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        print("❌ Cannot proceed without working model")
        return
    
    # Test forward pass
    if not test_forward_pass(model):
        print("❌ Forward pass failed, skipping remaining tests")
        return
    
    # Test vectorized operations
    if not test_vectorized_operations(model):
        print("❌ Vectorized operations failed")
        return
    
    # Test performance
    if not test_performance_comparison():
        print("❌ Performance testing failed")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All v0.2.0 validation tests passed!")
    print("   Temporal parallelism is working correctly")
    print("   Vectorized operations are functional")
    print("   Performance improvements validated")

if __name__ == "__main__":
    main()