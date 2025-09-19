#!/usr/bin/env python3
"""
PMFlow BNN v0.2.0 Comprehensive Library Testing

This script thoroughly tests the library for logic and syntax errors that could cause failures.
Tests all modules, functions, and edge cases before deployment.
"""

import sys
import os
import traceback
import importlib
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

def test_module_imports():
    """Test all module imports for syntax errors."""
    print("🔍 Testing Module Imports...")
    
    modules_to_test = [
        'pmflow_bnn',
        'pmflow_bnn.pmflow',
        'pmflow_bnn.bnn',
        'pmflow_bnn.factory',
        'pmflow_bnn.evaluation',
        'pmflow_bnn.utils',
        'pmflow_bnn.baselines',
        'pmflow_bnn.version'
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            results[module_name] = "✅ Success"
            print(f"   ✅ {module_name}")
        except Exception as e:
            results[module_name] = f"❌ Error: {str(e)}"
            print(f"   ❌ {module_name}: {str(e)}")
    
    return results

def test_factory_functions():
    """Test factory functions with various configurations."""
    print("\n🏭 Testing Factory Functions...")
    
    from pmflow_bnn import get_model_v2, get_performance_config
    
    test_cases = [
        {'model_type': 'temporal_pipeline'},
        {'model_type': 'always_plastic_v2'},
        {'model_type': 'standard_v2'},
        {'model_type': 'temporal_pipeline', 'n_centers': 16, 'pm_steps': 3},
        {'model_type': 'always_plastic_v2', 'plastic': True, 'plasticity_lr': 1e-3},
    ]
    
    if torch.cuda.device_count() > 1:
        test_cases.append({'model_type': 'multi_gpu'})
    
    results = {}
    
    for i, config in enumerate(test_cases):
        try:
            model = get_model_v2(**config)
            param_count = sum(p.numel() for p in model.parameters())
            results[f"config_{i+1}"] = f"✅ Success: {param_count:,} params"
            print(f"   ✅ Config {i+1}: {config['model_type']} - {param_count:,} params")
        except Exception as e:
            results[f"config_{i+1}"] = f"❌ Error: {str(e)}"
            print(f"   ❌ Config {i+1}: {str(e)}")
    
    # Test hardware configurations
    hardware_profiles = ['auto', 'cpu', 'single_gpu', 'jetson_nano']
    if torch.cuda.device_count() > 1:
        hardware_profiles.append('multi_gpu')
    
    print(f"\n   Testing hardware configurations:")
    for profile in hardware_profiles:
        try:
            config = get_performance_config(profile)
            results[f"hw_{profile}"] = f"✅ Success: {config['model_type']}"
            print(f"     ✅ {profile}: {config['model_type']}")
        except Exception as e:
            results[f"hw_{profile}"] = f"❌ Error: {str(e)}"
            print(f"     ❌ {profile}: {str(e)}")
    
    return results

def test_model_forward_passes():
    """Test forward passes with different inputs and configurations."""
    print("\n🧠 Testing Model Forward Passes...")
    
    from pmflow_bnn import get_model_v2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_configs = [
        {'model_type': 'temporal_pipeline', 'n_centers': 16, 'pm_steps': 3},
        {'model_type': 'standard_v2', 'n_centers': 12, 'pm_steps': 2},
    ]
    
    input_shapes = [
        (1, 28*28),     # Single sample
        (4, 28*28),     # Small batch
        (16, 28*28),    # Medium batch
        (32, 28*28),    # Large batch (if memory allows)
    ]
    
    results = {}
    
    for config_idx, config in enumerate(test_configs):
        print(f"   Testing {config['model_type']}...")
        
        try:
            model = get_model_v2(**config).to(device)
            model.eval()
            
            for shape in input_shapes:
                try:
                    with torch.no_grad():
                        x = torch.randn(shape, device=device)
                        
                        # Test standard forward pass
                        output = model(x)
                        if isinstance(output, tuple):
                            logits, hidden = output
                        else:
                            logits = output
                        
                        # Test with different temporal steps
                        if hasattr(model, 'forward') and 'T=' in str(model.forward.__code__.co_varnames):
                            output_t3 = model(x, T=3)
                            output_t5 = model(x, T=5)
                        
                        key = f"config_{config_idx+1}_shape_{shape[0]}"
                        results[key] = f"✅ Success: {logits.shape}"
                        print(f"     ✅ Batch {shape[0]}: Output shape {logits.shape}")
                        
                except Exception as e:
                    key = f"config_{config_idx+1}_shape_{shape[0]}"
                    results[key] = f"❌ Error: {str(e)}"
                    print(f"     ❌ Batch {shape[0]}: {str(e)}")
            
        except Exception as e:
            key = f"config_{config_idx+1}_model"
            results[key] = f"❌ Model creation failed: {str(e)}"
            print(f"   ❌ Model creation failed: {str(e)}")
    
    return results

def test_evaluation_framework():
    """Test the evaluation framework thoroughly."""
    print("\n📊 Testing Evaluation Framework...")
    
    from pmflow_bnn import PMFlowEvaluator, get_model_v2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    try:
        # Create evaluator
        evaluator = PMFlowEvaluator(device=device)
        results['evaluator_creation'] = "✅ Success"
        print("   ✅ PMFlowEvaluator created")
        
        # Create test model
        model = get_model_v2('temporal_pipeline', n_centers=16, pm_steps=3).to(device)
        results['test_model'] = "✅ Success"
        print("   ✅ Test model created")
        
        # Generate test data
        test_data = torch.randn(100, 28*28)
        test_labels = torch.randint(0, 4, (100,))
        shifting_data = [
            (torch.randn(50, 28*28), torch.randint(0, 4, (50,))),
            (torch.randn(50, 28*28), torch.randint(0, 4, (50,))),
        ]
        results['test_data'] = "✅ Success"
        print("   ✅ Test data generated")
        
        # Test embarrassingly parallel scaling (small scale)
        try:
            ep_results = evaluator.evaluate_embarrassingly_parallel_scaling(
                model, max_batch_size=8, input_shape=(28*28,)
            )
            results['parallel_scaling'] = f"✅ Success: {ep_results['peak_efficiency']:.2f}x peak"
            print(f"   ✅ Parallel scaling: {ep_results['peak_efficiency']:.2f}x peak efficiency")
        except Exception as e:
            results['parallel_scaling'] = f"❌ Error: {str(e)}"
            print(f"   ❌ Parallel scaling: {str(e)}")
        
        # Test gravitational dynamics
        try:
            gd_results = evaluator.evaluate_gravitational_dynamics(
                model, test_data, test_labels, adaptation_steps=5
            )
            if gd_results:
                results['gravitational_dynamics'] = f"✅ Success: {gd_results['mean_movement']:.4f} movement"
                print(f"   ✅ Gravitational dynamics: {gd_results['mean_movement']:.4f} movement")
            else:
                results['gravitational_dynamics'] = "⚠️ No gravitational centers detected"
                print("   ⚠️ Gravitational dynamics: No centers detected")
        except Exception as e:
            results['gravitational_dynamics'] = f"❌ Error: {str(e)}"
            print(f"   ❌ Gravitational dynamics: {str(e)}")
        
        # Test biological plasticity (simplified)
        try:
            bp_results = evaluator.evaluate_biological_plasticity(
                model, test_data[:50], test_labels[:50], shifting_data
            )
            results['biological_plasticity'] = f"✅ Success: {bp_results['plasticity_score']:.3f} score"
            print(f"   ✅ Biological plasticity: {bp_results['plasticity_score']:.3f} score")
        except Exception as e:
            results['biological_plasticity'] = f"❌ Error: {str(e)}"
            print(f"   ❌ Biological plasticity: {str(e)}")
        
    except Exception as e:
        results['framework_error'] = f"❌ Framework error: {str(e)}"
        print(f"   ❌ Framework error: {str(e)}")
    
    return results

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n⚠️  Testing Edge Cases...")
    
    from pmflow_bnn import get_model_v2, PMFlowEvaluator
    
    results = {}
    
    # Test invalid configurations
    invalid_configs = [
        {'model_type': 'nonexistent_model'},
        {'model_type': 'temporal_pipeline', 'n_centers': 0},
        {'model_type': 'temporal_pipeline', 'pm_steps': 0},
        {'model_type': 'temporal_pipeline', 'n_centers': -1},
    ]
    
    for i, config in enumerate(invalid_configs):
        try:
            model = get_model_v2(**config)
            results[f"invalid_{i+1}"] = "❌ Should have failed but didn't"
            print(f"   ❌ Invalid config {i+1} should have failed: {config}")
        except Exception as e:
            results[f"invalid_{i+1}"] = "✅ Correctly rejected"
            print(f"   ✅ Invalid config {i+1} correctly rejected: {str(e)[:50]}")
    
    # Test very small inputs
    try:
        model = get_model_v2('temporal_pipeline', n_centers=8, pm_steps=2)
        x = torch.randn(1, 28*28)
        output = model(x)
        results['small_input'] = "✅ Success"
        print(f"   ✅ Small input handled correctly")
    except Exception as e:
        results['small_input'] = f"❌ Error: {str(e)}"
        print(f"   ❌ Small input failed: {str(e)}")
    
    # Test memory constraints
    device = torch.device('cpu')  # Force CPU for memory test
    try:
        model = get_model_v2('temporal_pipeline', n_centers=64, pm_steps=5).to(device)
        x = torch.randn(64, 28*28, device=device)  # Large batch
        with torch.no_grad():
            output = model(x)
        results['memory_test'] = "✅ Success"
        print(f"   ✅ Memory constraints handled")
    except Exception as e:
        results['memory_test'] = f"⚠️ Memory issue: {str(e)[:50]}"
        print(f"   ⚠️ Memory test: {str(e)[:50]}")
    
    return results

def test_performance_benchmarks():
    """Test performance benchmarking capabilities."""
    print("\n⚡ Testing Performance Benchmarks...")
    
    from pmflow_bnn import benchmark_temporal_parallelism, get_model_v2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    try:
        model = get_model_v2('temporal_pipeline', n_centers=16, pm_steps=3).to(device)
        
        # Test benchmark function
        benchmark_results = benchmark_temporal_parallelism(
            model, batch_sizes=[2, 4, 8], device=device, num_trials=3
        )
        
        results['benchmark_function'] = f"✅ Success: {len(benchmark_results['batch_sizes'])} sizes tested"
        print(f"   ✅ Benchmark function: {len(benchmark_results['batch_sizes'])} batch sizes tested")
        
        # Check benchmark results structure
        required_keys = ['batch_sizes', 'forward_times', 'throughput', 'memory_usage']
        missing_keys = [key for key in required_keys if key not in benchmark_results]
        
        if not missing_keys:
            results['benchmark_structure'] = "✅ All required keys present"
            print(f"   ✅ Benchmark structure complete")
        else:
            results['benchmark_structure'] = f"❌ Missing keys: {missing_keys}"
            print(f"   ❌ Missing benchmark keys: {missing_keys}")
        
    except Exception as e:
        results['benchmark_error'] = f"❌ Error: {str(e)}"
        print(f"   ❌ Benchmark error: {str(e)}")
    
    return results

def generate_test_report(all_results: Dict[str, Dict]) -> str:
    """Generate comprehensive test report."""
    
    total_tests = sum(len(results) for results in all_results.values())
    passed_tests = sum(
        sum(1 for result in results.values() if result.startswith("✅"))
        for results in all_results.values()
    )
    
    report = f"""
PMFlow BNN v0.2.0 Library Test Report
=====================================

📊 SUMMARY:
   Total Tests: {total_tests}
   Passed: {passed_tests}
   Failed: {total_tests - passed_tests}
   Success Rate: {passed_tests/total_tests*100:.1f}%

"""
    
    for category, results in all_results.items():
        report += f"\n📋 {category.upper().replace('_', ' ')}:\n"
        for test, result in results.items():
            report += f"   {test}: {result}\n"
    
    # Overall assessment
    success_rate = passed_tests / total_tests
    if success_rate >= 0.9:
        report += f"\n🎉 OVERALL ASSESSMENT: EXCELLENT - Library ready for deployment!"
    elif success_rate >= 0.8:
        report += f"\n✅ OVERALL ASSESSMENT: GOOD - Minor issues to address"
    elif success_rate >= 0.7:
        report += f"\n⚠️ OVERALL ASSESSMENT: MODERATE - Some issues need fixing"
    else:
        report += f"\n❌ OVERALL ASSESSMENT: POOR - Major issues require attention"
    
    return report

def main():
    """Run comprehensive library testing."""
    
    print("🧪 PMFlow BNN v0.2.0 Comprehensive Library Testing")
    print("="*55)
    print("Testing for logic/syntax errors before deployment...\n")
    
    all_results = {}
    
    # Run all test categories
    all_results['module_imports'] = test_module_imports()
    all_results['factory_functions'] = test_factory_functions()
    all_results['model_forward_passes'] = test_model_forward_passes()
    all_results['evaluation_framework'] = test_evaluation_framework()
    all_results['edge_cases'] = test_edge_cases()
    all_results['performance_benchmarks'] = test_performance_benchmarks()
    
    # Generate and display report
    report = generate_test_report(all_results)
    print(f"\n{report}")
    
    # Save report to file
    with open('library_test_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n📄 Test report saved to: library_test_report.txt")
    
    return all_results

if __name__ == "__main__":
    results = main()