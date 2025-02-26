# DP5 Tests

This directory contains tests for the DP5 software.

## Running Tests

You can run all tests using the `run_tests.py` script in the project root:

```bash
./run_tests.py
```

Or you can use pytest directly:

```bash
pytest -v tests/
```

To run a specific test file:

```bash
pytest -v tests/test_dp5_examples.py
```

To run a specific test:

```bash
pytest -v tests/test_dp5_examples.py::TestDP5Examples::test_example_runs
```

## Test Structure

- `test_dp5_examples.py`: Tests that DP5 runs correctly on example data
- `test_sgnn_model.py`: Tests for the SGNN model functionality

## Adding New Tests

To add new tests, create a new test file in this directory. The file name should start with `test_` and end with `.py`. The test functions should also start with `test_`.

## Requirements

The tests require pytest to be installed:

```bash
pip install pytest
``` 