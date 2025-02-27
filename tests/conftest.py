import pytest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define fixtures that can be reused across tests
@pytest.fixture(scope="session")
def examples_dir():
    """Return the path to the examples directory"""
    examples_path = "/scratch/jbr46/structure_reassignment_examples"
    if not os.path.isdir(examples_path):
        pytest.skip(f"Examples directory {examples_path} not found")
    return examples_path

@pytest.fixture(scope="session")
def config_path(examples_dir):
    """Return the path to the config file"""
    config_path = os.path.join(examples_dir, "config.toml")
    if not os.path.isfile(config_path):
        pytest.skip(f"Config file {config_path} not found")
    return config_path 