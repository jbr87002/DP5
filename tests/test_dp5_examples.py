import os
import sys
import pytest
import subprocess
import time
from pathlib import Path

# Path to the examples directory
EXAMPLES_DIR = "/scratch/jbr46/structure_reassignment_examples"
CONFIG_PATH = os.path.join(EXAMPLES_DIR, "config.toml")

# Models to test
MODELS = ["cascade", "sgnn"]

def run_dp5_command(directory, command):
    """Run a dp5 command in the specified directory and return the result"""
    original_dir = os.getcwd()
    try:
        os.chdir(directory)
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        return result
    finally:
        os.chdir(original_dir)

# Module-level variables to track if SGNN and Cascade are available
sgnn_available = False
cascade_available = False

class TestDP5Examples:
    """Test class for running dp5 on example data"""
    
    def setup_method(self):
        """Setup method that runs before each test"""
        if not os.path.isdir(EXAMPLES_DIR):
            pytest.skip(f"Examples directory {EXAMPLES_DIR} not found")
            
        if not os.path.isfile(CONFIG_PATH):
            pytest.skip(f"Config file {CONFIG_PATH} not found")
    
    def test_sgnn_available(self):
        """Test that SGNN model is available"""
        # Import the module that defines SGNN_AVAILABLE
        from dp5.neural_net.nn_utils import SGNN_AVAILABLE
        
        # Check if SGNN is available
        assert SGNN_AVAILABLE, "SGNN model is not available. Make sure all dependencies are installed."
        
        # If we get here, SGNN is available and working
        global sgnn_available
        sgnn_available = True
    
    def test_cascade_available(self):
        """Test that Cascade model is available"""
        from dp5.neural_net.nn_utils import CASCADE_AVAILABLE
        
        assert CASCADE_AVAILABLE, "Cascade model is not available. Make sure all dependencies are installed."
        
        global cascade_available
        cascade_available = True

    @pytest.mark.parametrize("example_num", range(1, 25))
    @pytest.mark.parametrize("model_name", MODELS)
    def test_example_runs(self, example_num, model_name):
        """Test that dp5 runs successfully on each example with specified model"""
        # Skip SGNN tests if SGNN is not available
        if model_name == "sgnn" and not sgnn_available and pytest in sys.modules:
            pytest.skip("Skipping SGNN tests because SGNN model is not available")
        
        if model_name == "cascade" and not cascade_available and pytest in sys.modules:
            pytest.skip("Skipping Cascade tests because Cascade model is not available")
            
        directory = os.path.join(EXAMPLES_DIR, f"S{example_num}")
        
        if not os.path.isdir(directory):
            pytest.skip(f"Example directory S{example_num} not found")
        
        print(f"\n{'='*80}")
        print(f"RUNNING: S{example_num} with model {model_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        if example_num == 13:
            # Modified command for S13
            command = f'pydp4 -n S13a_NMR -i sdf -s S13a_.sdf S13b_.sdf -c {CONFIG_PATH} -w cw --model {model_name} --remove'
        else:
            command = f'pydp4 -n S{example_num}_NMR -i sdf -s S{example_num}a_.sdf S{example_num}b_.sdf -c {CONFIG_PATH} -w cw --model {model_name} --remove'
        
        print(f"COMMAND: {command}")
        
        result = run_dp5_command(directory, command)
        
        # Combine stdout and stderr for checking and display
        all_output = result.stdout + result.stderr
        
        # Print a summary of the output (first few lines and last few lines)
        output_lines = all_output.splitlines()
        if len(output_lines) > 20:
            print("Output (truncated):")
            print('\n'.join(output_lines[:10]))
            print("...")
            print('\n'.join(output_lines[-10:]))
        else:
            print("Output:")
            print(all_output)
        
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        
        # Check if the command was successful
        assert result.returncode == 0, f"DP5 failed on example S{example_num} with model {model_name}. Error: {result.stderr}"
        
        # Check for "Program terminated normally" in the combined output
        assert "Program terminated normally" in all_output, f"DP5 did not terminate normally on example S{example_num} with model {model_name}"
        
        # Only check for fallback if using SGNN model
        if model_name == "sgnn":
            assert "Falling back to cascade model" not in all_output, f"SGNN model is not being used for example S{example_num}, falling back to cascade model"
        
if __name__ == "__main__":
    # Run the tests
    pytest.main(["-v", __file__]) 