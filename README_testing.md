# Testing Framework for Nano-Genie

This document describes the comprehensive testing framework for the nano-genie video generation system.

## Overview

The testing framework includes:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Full pipeline testing
- **Configuration Tests**: Verification of saved configurations
- **Model Tests**: Neural network model validation
- **Pipeline Tests**: End-to-end workflow testing

## Test Structure

```
tests/
├── __init__.py
├── test_video_tokenizer.py    # Video tokenizer tests
├── test_lam.py                # LAM (Latent Action Model) tests
├── test_dynamics.py           # Dynamics model tests
└── test_full_pipeline.py      # Full pipeline tests
```

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run with coverage reporting
python run_tests.py --coverage
```

### Specific Test Categories

```bash
# Run specific model tests
python run_tests.py --type video_tokenizer
python run_tests.py --type lam
python run_tests.py --type dynamics
python run_tests.py --type pipeline

# Run by test type
python run_tests.py --type unit
python run_tests.py --type integration

# Run by device requirement
python run_tests.py --type cpu
python run_tests.py --type gpu
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_video_tokenizer.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

## Test Categories

### 1. Video Tokenizer Tests (`test_video_tokenizer.py`)

Tests for the video tokenizer model and main script:

- **Model Initialization**: Verify model creation and parameters
- **Forward Pass**: Test encoding/decoding functionality
- **Vector Quantization**: Test VQ-VAE components
- **Device Compatibility**: CPU/GPU testing
- **Gradient Flow**: Verify backpropagation
- **Model Save/Load**: Checkpoint functionality
- **Configuration Saving**: Verify `run_config.json` creation
- **Dataset Loading**: Pong dataset functionality

### 2. LAM Tests (`test_lam.py`)

Tests for the Latent Action Model:

- **Model Initialization**: Verify LAM architecture
- **Forward Pass**: Test action encoding/decoding
- **Action Diversity**: Ensure diverse action generation
- **Loss Components**: Verify all loss terms
- **Single Transitions**: Test encode/decode for frame pairs
- **Device Compatibility**: CPU/GPU testing
- **Configuration Saving**: Verify configuration persistence

### 3. Dynamics Model Tests (`test_dynamics.py`)

Tests for the dynamics model:

- **Model Initialization**: Verify dynamics architecture
- **Forward Pass**: Test latent prediction
- **Training Mode**: Test masking functionality
- **Full Integration**: Test with video tokenizer and LAM
- **Latent Consistency**: Verify latent space properties
- **Device Compatibility**: CPU/GPU testing
- **Checkpoint Loading**: Test pre-trained model loading

### 4. Full Pipeline Tests (`test_full_pipeline.py`)

Tests for the complete training pipeline:

- **Checkpoint Discovery**: Test automatic checkpoint finding
- **Command Execution**: Test subprocess management
- **Error Handling**: Test graceful failure handling
- **Directory Validation**: Verify directory structure
- **Configuration Creation**: Test config file generation
- **Hyperparameter Consistency**: Verify parameter matching
- **Pipeline Logging**: Test progress reporting
- **Completion Summary**: Test final results reporting

## Configuration File Testing

All main scripts now save comprehensive configuration files (`run_config.json`) that include:

### Video Tokenizer Configuration
```json
{
  "timestamp": "2024-01-01_12:00:00",
  "device": "cuda",
  "model_architecture": {
    "frame_size": [64, 64],
    "patch_size": 8,
    "embed_dim": 128,
    "num_heads": 4,
    "hidden_dim": 512,
    "num_blocks": 2,
    "latent_dim": 16,
    "dropout": 0.1,
    "codebook_size": 256,
    "beta": 0.01
  },
  "training_parameters": {
    "batch_size": 32,
    "n_updates": 10000,
    "learning_rate": 1e-4,
    "log_interval": 250,
    "context_length": 4,
    "dataset": "PONG"
  },
  "checkpoint_info": {
    "checkpoint_path": null,
    "start_iteration": 0
  },
  "directories": {
    "run_dir": "/path/to/results",
    "checkpoints_dir": "/path/to/checkpoints",
    "visualizations_dir": "/path/to/visualizations"
  }
}
```

### LAM Configuration
```json
{
  "timestamp": "2024-01-01_12:00:00",
  "device": "cuda",
  "model_architecture": {
    "frame_size": [64, 64],
    "n_actions": 4,
    "patch_size": 8,
    "embed_dim": 128,
    "num_heads": 4,
    "hidden_dim": 512,
    "num_blocks": 2,
    "action_dim": 16,
    "dropout": 0.1
  },
  "training_parameters": {
    "batch_size": 32,
    "n_updates": 1000,
    "learning_rate": 3e-3,
    "log_interval": 100,
    "seq_length": 8,
    "dataset": "PONG"
  },
  "directories": {
    "run_dir": "/path/to/results",
    "checkpoints_dir": "/path/to/checkpoints",
    "visualizations_dir": "/path/to/visualizations"
  }
}
```

### Dynamics Model Configuration
```json
{
  "timestamp": "2024-01-01_12:00:00",
  "device": "cuda",
  "model_architecture": {
    "frame_size": [64, 64],
    "patch_size": 8,
    "embed_dim": 128,
    "num_heads": 4,
    "hidden_dim": 512,
    "num_blocks": 2,
    "latent_dim": 16,
    "dropout": 0.1,
    "height": 64,
    "width": 64,
    "channels": 3
  },
  "training_parameters": {
    "batch_size": 32,
    "n_updates": 10000,
    "learning_rate": 1e-4,
    "log_interval": 250,
    "context_length": 4,
    "dataset": "PONG"
  },
  "pretrained_models": {
    "video_tokenizer_path": "/path/to/vt_checkpoint.pth",
    "lam_path": "/path/to/lam_checkpoint.pth"
  },
  "checkpoint_info": {
    "checkpoint_path": null,
    "start_iteration": 0
  },
  "directories": {
    "run_dir": "/path/to/results",
    "checkpoints_dir": "/path/to/checkpoints",
    "visualizations_dir": "/path/to/visualizations"
  }
}
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for full workflows
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.gpu`: Tests that require GPU
- `@pytest.mark.cpu`: Tests that run on CPU only

## Running Specific Test Types

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only fast tests
pytest -m "not slow"

# Run only CPU tests
pytest -m cpu

# Run only GPU tests (if available)
pytest -m gpu
```

## Coverage Reporting

Generate coverage reports to see test coverage:

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate terminal coverage report
pytest --cov=src --cov-report=term

# Generate both
pytest --cov=src --cov-report=html --cov-report=term
```

The HTML report will be generated in `htmlcov/index.html`.

## Continuous Integration

The tests are designed to work in CI environments:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: python run_tests.py --coverage
```

## Debugging Tests

### Verbose Output
```bash
pytest -v tests/test_video_tokenizer.py
```

### Stop on First Failure
```bash
pytest -x tests/
```

### Show Local Variables on Failure
```bash
pytest -l tests/
```

### Run with Debugger
```bash
pytest --pdb tests/
```

## Test Data

Tests use mock data to avoid requiring large datasets:

- **Mock Video Frames**: Random tensors simulating video data
- **Mock HDF5 Files**: Simulated Pong dataset files
- **Mock Checkpoints**: Simulated model checkpoints
- **Temporary Directories**: Isolated test environments

## Best Practices

1. **Isolation**: Each test runs in isolation with fresh data
2. **Cleanup**: Tests clean up after themselves
3. **Mocking**: External dependencies are mocked
4. **Assertions**: Clear, specific assertions
5. **Documentation**: Tests are well-documented
6. **Coverage**: Aim for high test coverage

## Adding New Tests

When adding new functionality:

1. **Create test file**: `tests/test_new_feature.py`
2. **Add unit tests**: Test individual components
3. **Add integration tests**: Test full workflows
4. **Update configuration tests**: Test config saving
5. **Add markers**: Categorize tests appropriately
6. **Update documentation**: Document new tests

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Missing Dependencies**: Install required packages
3. **GPU Tests Failing**: Check CUDA availability
4. **Slow Tests**: Use `-m "not slow"` to skip
5. **Memory Issues**: Reduce batch sizes in tests

### Getting Help

- Check test output for specific error messages
- Run tests with `-v` for verbose output
- Use `--pdb` to debug failing tests
- Check coverage reports for untested code

## Performance

Test execution times (approximate):

- **Unit Tests**: 10-30 seconds
- **Integration Tests**: 1-5 minutes
- **Full Test Suite**: 2-10 minutes
- **With Coverage**: +50% time

## Future Enhancements

Planned improvements:

1. **Performance Benchmarks**: Add timing tests
2. **Memory Usage Tests**: Monitor memory consumption
3. **Stress Tests**: Test with large datasets
4. **Regression Tests**: Track performance over time
5. **Visual Tests**: Test visualization outputs
6. **API Tests**: Test external interfaces 