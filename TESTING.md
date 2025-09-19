# Testing Documentation

## Running the Test Suite

The Pushing-Medium gravitational model has a comprehensive test suite with **59 tests** covering all aspects of the theory from basic GR equivalence to advanced galaxy dynamics.

### Quick Start

```bash
# Run all tests
pytest tests -v

# Run specific test categories
pytest tests/test_passed_benchmarks.py -v    # Core GR comparison tests
pytest tests/test_galaxy_rotation.py -v     # Galaxy dynamics tests
pytest tests/test_sparc_loader.py -v        # SPARC data integration tests

# Run with quiet output
pytest tests -q

# Run with coverage
pytest tests --cov
```

### Test Organization

The test suite is organized into 8 main categories:

#### 1. **Calibration & Validation** (4 tests)
- `test_calibration_mu.py` - Calibration parameter validation
- `test_calibration_values.py` - Expected value verification

#### 2. **Classical GR Benchmarks** (10 tests) 
- `test_passed_benchmarks.py` - Core GR comparison tests
- `test_gr_classical.py` - Additional GR validation

**Key tests:**
- Light bending by point mass
- Shapiro time delay
- Gravitational redshift
- Perihelion precession
- Frame drag (Lense-Thirring)
- Einstein radius (strong lensing)
- Gravitational wave speed
- Quadrupole radiation power
- Circular orbit energy

#### 3. **Light Deflection & Lensing** (6 tests)
- `test_deflection_convergence.py` - Numerical convergence
- `test_fermat_helper.py` - Fermat principle validation
- `test_iterative_deflection.py` - Iterative methods
- `test_weak_bending_numeric.py` - Weak field lensing
- `test_strong_field_trend.py` - Strong field behavior

#### 4. **Galaxy Dynamics & Rotation** (10 tests)
- `test_galaxy_rotation.py` - Rotation curve physics
- `test_fitting.py` - Parameter fitting
- `test_halo_fitting.py` - Dark matter halo comparison
- `test_scaling_relations.py` - Observational scaling laws

#### 5. **Core Physics & Medium Properties** (15 tests)
- `test_pm_core.py` - Core pushing-medium physics
- `test_pm_vs_gr.py` - Direct PM vs GR comparison
- `test_index_from_density.py` - Medium property calculations
- `test_plummer_helpers.py` - Stellar distribution models

#### 6. **Model Comparison & SPARC Data** (10 tests)
- `test_model_comparison.py` - Multi-model comparison framework
- `test_sparc_loader.py` - SPARC database integration
- `test_population_fitting.py` - Population-scale analysis

#### 7. **Moving Lens & Frame Effects** (3 tests)
- `test_moving_lens_deflection.py` - Moving source effects
- `test_moving_lens_numeric.py` - Numerical moving lens

#### 8. **PPN Parameters & Relativity** (2 tests)
- `test_ppn_parameters.py` - Post-Newtonian parameter validation

### Current Status

**✅ All 59 tests pass with 100% success rate**

Last verified: September 19, 2025

### Automated Testing

Tests run automatically via GitHub Actions on:
- Push to main branch
- Pull requests to main branch

See `.github/workflows/python-tests.yml` for CI configuration.

### Dependencies

Required for testing:
```bash
pip install pytest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install .  # Install the pushing-medium package
```

### Test Data

Test data is located in:
- `tests/data/mock_sparc.csv` - Mock SPARC galaxy data
- `tests/data/sparc_sample.csv` - Sample real SPARC data
- `tests/fixtures/` - Test fixtures and calibration data

### Generating Test Reports

Use the comprehensive test report generator:

```bash
python test_results_table.py
```

This generates a detailed breakdown of all test categories and results.

### Troubleshooting

**Import Errors:**
- Ensure you're in the gravity repository root directory
- Install the package: `pip install .`
- Check Python path includes the repository

**Missing Test Data:**
- Verify `tests/data/` directory exists with CSV files
- Check file permissions

**PyTorch Issues:**
- Install CPU version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- For GPU testing, install CUDA version

---

## Writing New Tests

### Test Structure

Follow pytest conventions:
- Test files: `test_*.py` in `tests/` directory
- Test functions: `def test_*():`
- Use descriptive names: `test_light_bending_matches_gr()`

### Example Test

```python
def test_new_physics_feature():
    """Test description of what this validates"""
    from pushing_medium import some_function
    
    result = some_function(test_input)
    expected = known_good_value
    
    assert abs(result - expected) < tolerance
```

### Categories to Test

When adding new features, ensure tests cover:
1. **GR equivalence** - Does it match GR in appropriate limits?
2. **Physical consistency** - Are results physically reasonable?
3. **Numerical stability** - Convergence with resolution/precision?
4. **Edge cases** - Boundary conditions and limits
5. **Integration** - Works with existing components?

---

*This documentation ensures the comprehensive test suite remains accessible and well-understood for ongoing development and validation of the Pushing-Medium gravitational model.*