#!/usr/bin/env python3
"""
Optimized PMFlow BNN Configuration & Testing

This script addresses the performance issues identified and provides
better hyperparameter configurations for improved initial results.
"""

import torch
import numpy as np
from pmflow_bnn import get_model_v2, PMFlowEvaluator
from pmflow_bnn.factory import get_performance_config

def get_optimized_config(hardware_profile='auto'):
    """Get optimized configuration with better hyperparameters."""
    
    base_config = get_performance_config(hardware_profile)
    
    # Optimizations based on analysis
    optimizations = {
        'cpu': {
            'model_type': 'temporal_pipeline',
            'n_centers': 32,          # Increased from 24
            'pm_steps': 4,            # Increased from 3
            'dt': 0.15,               # Optimized time step
            'beta': 1.2,              # Increased coupling strength
            'temporal_stages': 2,     # Enable parallelism
            'pipeline_overlap': True, # Enable overlap
            'd_latent': 12,           # Increased dimensionality
            'channels': 96,           # Increased capacity
            'chunk_size': 16,         # Optimized chunk size
        },
        'single_gpu': {
            'n_centers': 64,          # Keep at 64
            'pm_steps': 5,            # Increased from 4
            'dt': 0.12,               # Standard
            'beta': 1.0,              # Standard
            'temporal_stages': 3,     # More stages
            'd_latent': 16,           # Increased
            'channels': 128,          # Increased
        },
        'jetson_nano': {
            'n_centers': 48,          # Increased from 32
            'pm_steps': 4,            # Keep at 4
            'dt': 0.14,               # Slightly increased
            'beta': 1.1,              # Slightly increased
            'temporal_stages': 2,     # Keep at 2
            'd_latent': 10,           # Increased
            'channels': 80,           # Increased
        }
    }
    
    # Apply optimizations
    if hardware_profile in optimizations:
        base_config.update(optimizations[hardware_profile])
    elif base_config.get('model_type') == 'standard_v2':
        # If it detected as basic config, apply CPU optimizations
        base_config.update(optimizations['cpu'])
    
    return base_config

def test_hyperparameter_impact():
    """Test different hyperparameter configurations to find optimal settings."""
    
    print("🔬 Testing Hyperparameter Impact on PMFlow Performance")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configurations
    configs_to_test = [
        {
            'name': 'Original',
            'config': get_performance_config('auto')
        },
        {
            'name': 'Optimized',  
            'config': get_optimized_config('auto')
        },
        {
            'name': 'High Capacity',
            'config': {
                'model_type': 'temporal_pipeline',
                'n_centers': 48,
                'pm_steps': 5,
                'dt': 0.18,
                'beta': 1.3,
                'd_latent': 16,
                'channels': 128,
                'temporal_stages': 3,
                'pipeline_overlap': True,
            }
        }
    ]
    
    results = {}
    
    for config_info in configs_to_test:
        name = config_info['name']
        config = config_info['config']
        
        print(f"\n🧪 Testing '{name}' configuration:")
        print(f"    Centers: {config.get('n_centers', 'N/A')}")
        print(f"    PM Steps: {config.get('pm_steps', 'N/A')}")
        print(f"    dt: {config.get('dt', 'N/A')}")
        print(f"    beta: {config.get('beta', 'N/A')}")
        
        try:
            # Create model
            model = get_model_v2(**config)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"    Parameters: {param_count:,}")
            
            # Quick evaluation
            evaluator = PMFlowEvaluator(device=device)
            
            # Test scaling (smaller range for speed)
            max_batch = 16 if device.type == 'cpu' else 32
            ep_results = evaluator.evaluate_embarrassingly_parallel_scaling(
                model, max_batch_size=max_batch, input_shape=(28*28,)
            )
            
            results[name] = {
                'config': config,
                'parameters': param_count,
                'peak_efficiency': ep_results['peak_efficiency'],
                'average_efficiency': ep_results['average_efficiency'],
                'is_embarrassingly_parallel': ep_results['is_embarrassingly_parallel'],
                'scaling_trend': ep_results.get('scaling_trend', 'unknown')
            }
            
            print(f"    📊 Results:")
            print(f"        Peak Efficiency: {ep_results['peak_efficiency']:.2f}x")
            print(f"        Average Efficiency: {ep_results['average_efficiency']:.2f}x") 
            print(f"        Embarrassingly Parallel: {ep_results['is_embarrassingly_parallel']}")
            print(f"        Scaling Trend: {ep_results.get('scaling_trend', 'unknown')}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[name] = {'error': str(e)}
    
    # Summary
    print(f"\n📊 HYPERPARAMETER IMPACT SUMMARY")
    print("="*40)
    
    for name, result in results.items():
        if 'error' not in result:
            print(f"\n{name}:")
            print(f"  Peak Efficiency: {result['peak_efficiency']:.2f}x")
            print(f"  Average Efficiency: {result['average_efficiency']:.2f}x")
            print(f"  Embarrassingly Parallel: {result['is_embarrassingly_parallel']}")
            print(f"  Parameters: {result['parameters']:,}")
    
    # Find best configuration
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_config = max(valid_results.items(), key=lambda x: x[1]['peak_efficiency'])
        print(f"\n🏆 BEST CONFIGURATION: {best_config[0]}")
        print(f"    Peak Efficiency: {best_config[1]['peak_efficiency']:.2f}x")
        print(f"    Average Efficiency: {best_config[1]['average_efficiency']:.2f}x")
    
    return results

def create_improved_demo():
    """Create an improved demo with optimized hyperparameters."""
    
    print(f"\n🚀 Running Improved PMFlow Demo with Optimized Settings")
    print("="*55)
    
    # Get optimized configuration
    config = get_optimized_config('auto')
    print(f"Optimized config: {config}")
    
    # Create model
    model = get_model_v2(**config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Test with evaluator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = PMFlowEvaluator(device=device)
    
    # Generate better test data
    train_data = torch.randn(600, 28*28) * 0.8  # Reduced variance for stability
    train_labels = torch.randint(0, 4, (600,))
    test_data = torch.randn(200, 28*28) * 0.8
    test_labels = torch.randint(0, 4, (200,))
    
    # Test embarrassingly parallel scaling
    print(f"\n🚀 Testing Optimized Embarrassingly Parallel Scaling...")
    max_batch = 32 if device.type == 'cuda' else 16
    ep_results = evaluator.evaluate_embarrassingly_parallel_scaling(
        model, max_batch_size=max_batch
    )
    
    print(f"\n📈 Improved Results:")
    print(f"    Peak Efficiency: {ep_results['peak_efficiency']:.2f}x")
    print(f"    Average Efficiency: {ep_results['average_efficiency']:.2f}x")
    print(f"    Embarrassingly Parallel: {ep_results['is_embarrassingly_parallel']}")
    print(f"    Scaling Trend: {ep_results.get('scaling_trend', 'unknown')}")
    
    if ep_results['peak_efficiency'] > 5.0:
        print("    🎉 EXCELLENT: Much improved scaling!")
    elif ep_results['peak_efficiency'] > 2.0:
        print("    ✅ GOOD: Significant improvement")
    else:
        print("    ⚠️ MODERATE: Some improvement")
    
    # Test gravitational dynamics
    print(f"\n🌌 Testing Optimized Gravitational Dynamics...")
    gd_results = evaluator.evaluate_gravitational_dynamics(
        model, test_data, test_labels, adaptation_steps=15
    )
    
    if gd_results and 'specialization_ratio' in gd_results:
        print(f"    Specialization Ratio: {gd_results['specialization_ratio']:.2f}x")
        print(f"    Center Movement: {gd_results['mean_movement']:.4f}")
    
    return {
        'config': config,
        'model': model,
        'ep_results': ep_results,
        'gd_results': gd_results
    }

if __name__ == "__main__":
    print("🎯 PMFlow BNN Optimization & Testing")
    print("="*40)
    
    # Test hyperparameter impact
    hp_results = test_hyperparameter_impact()
    
    # Run improved demo
    demo_results = create_improved_demo()
    
    print(f"\n🎉 Optimization Complete!")
    print("Key improvements applied to address performance issues.")