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

def pytest_addoption(parser):
    parser.addoption(
        "--model-filter", 
        action="store", 
        default="all", 
        help="Filter tests by model: 'cascade', 'sgnn', or 'all'"
    )

def pytest_collection_modifyitems(config, items):
    model_filter = config.getoption("--model-filter")
    if model_filter == "all":
        # Run all tests
        return
    
    skip_model = pytest.mark.skip(reason=f"Test doesn't use the {model_filter} model")
    
    for item in items:
        # Check if this is a parameterized test with model_name
        if hasattr(item, 'callspec') and hasattr(item.callspec, 'params') and "model_name" in item.callspec.params:
            # This is a test with a model parameter
            if item.callspec.params["model_name"] != model_filter:
                item.add_marker(skip_model) 