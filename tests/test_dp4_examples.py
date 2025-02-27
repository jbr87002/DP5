import os
import sys
import pytest
import subprocess
import glob
import time
import pickle
import numpy as np
from pathlib import Path

# Path to the examples directory
EXAMPLES_DIR = "/scratch/jbr46/dp4_examples"
CONFIG_PATH = os.path.join(EXAMPLES_DIR, "config.toml")

# Directories to skip (known to be problematic)
SKIP_DIRS = ["BYH2"]

ACCEPTABLE_ERRORS = {
    'AT3': 1000,
    'TS4': 1000,
    'IP1': 12,
    'KE3': 11,
    'KE1': 10,
    'NL1B': 10,
    'AT1': 10,
    'KE2': 9,
    'JB11': 8,
    'NP3B': 7,
    'NP1': 7,
    'TS3A': 7,
    'NL2B': 7,
    'TP2': 7,
    'JB7': 6,
    'NP2': 6,
    'JB10': 6,
    'JB1': 6,
    'NP5': 6,
    'NL2A': 6,
    'JB2': 6,
    'NP4': 6,
    'NP3A': 6,
    'JB12': 6,
    'AT2': 6,
    'IP5': 5,
    'BYH1': 5,
    'JB13A': 5,
    'JB9': 5,
    'JB13B': 5,
    'JB6': 5,
    'JB8': 5,
    'IP2': 5,
    'JB5': 5,
    'TP3': 4,
    'IP3': 4,
    'TP1': 4,
    'JB4': 4,
    'TS3B': 4,
    'OD1': 4,
    'TS1': 4,
    'IP4': 4,
    'TS2': 4,
    'JB3': 3,
    'NL1A': 1000
}

# Models to test
MODELS = ["cascade", "sgnn"]

def run_dp4_command(directory, command):
    """Run a dp4 command in the specified directory and return the result"""
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

def get_mae(directory):
    """Get the MAE for a directory"""
    data_dic_path = os.path.join(directory, 'dp4/data_dic.p')
    data_dic = pickle.load(open(data_dic_path, 'rb'))
    error_list = data_dic['Cerrors'][0]
    print(error_list)
    mae = np.nanmean(error_list)
    return mae

def find_sdf_files(directory):
    """Find SDF files in a directory"""
    sdf_files = glob.glob(os.path.join(directory, "*.sdf"))
    return [os.path.basename(f) for f in sdf_files]

def check_directory_structure():
    """Check the directory structure and print information about it"""
    if not os.path.isdir(EXAMPLES_DIR):
        print(f"Examples directory {EXAMPLES_DIR} not found")
        return False
        
    if not os.path.isfile(CONFIG_PATH):
        print(f"Config file {CONFIG_PATH} not found")
        return False
    
    example_dirs = [d for d in os.listdir(EXAMPLES_DIR) 
                   if os.path.isdir(os.path.join(EXAMPLES_DIR, d)) 
                   and not d.startswith('.')
                   and d not in SKIP_DIRS]
    
    if not example_dirs:
        print("No example directories found")
        return False
    
    print(f"Found {len(example_dirs)} example directories:")
    for dir_name in example_dirs:
        dir_path = os.path.join(EXAMPLES_DIR, dir_name)
        sdf_files = find_sdf_files(dir_path)
        
        if sdf_files:
            print(f"  {dir_name}: {', '.join(sdf_files)}")
        else:
            print(f"  {dir_name}: No SDF files found")
    
    return True

def check_command_exists(command):
    """Check if a command exists in the PATH"""
    try:
        result = subprocess.run(
            f"which {command}", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

# Module-level variables to track if SGNN and Cascade are available
sgnn_available = False
cascade_available = False

class TestDP4Examples:
    """Test class for running dp4 on example data"""
    
    def setup_method(self):
        """Setup method that runs before each test"""
        if not os.path.isdir(EXAMPLES_DIR):
            pytest.skip(f"Examples directory {EXAMPLES_DIR} not found")
            
        if not os.path.isfile(CONFIG_PATH):
            pytest.skip(f"Config file {CONFIG_PATH} not found")
            
        # Check if pydp4 command exists
        if not check_command_exists("pydp4"):
            pytest.skip("pydp4 command not found in PATH. Make sure it's installed and in your PATH.")
    
    def test_sgnn_available(self):
        """Test that SGNN model is available"""
        try:
            from dp5.neural_net.nn_utils import SGNN_AVAILABLE
                
            assert SGNN_AVAILABLE, "SGNN model is not available. Make sure all dependencies are installed."
            
            # If we get here, SGNN is available and working
            global sgnn_available
            sgnn_available = True
        except Exception as e:
            # fail the test if SGNN is not available
            pytest.fail(f"SGNN test failed: {str(e)}")
    
    def test_cascade_available(self):
        """Test that Cascade model is available"""
        try:
            from dp5.neural_net.nn_utils import CASCADE_AVAILABLE
            
            assert CASCADE_AVAILABLE, "Cascade model is not available. Make sure all dependencies are installed."

            global cascade_available
            cascade_available = True
        except Exception as e:
            # fail the test if Cascade is not available
            pytest.fail(f"Cascade test failed: {str(e)}")
    
    @pytest.mark.parametrize("dir_name", 
                             [d for d in os.listdir(EXAMPLES_DIR) 
                              if os.path.isdir(os.path.join(EXAMPLES_DIR, d)) 
                              and not d.startswith('.')
                              and d not in SKIP_DIRS])
    @pytest.mark.parametrize("model_name", MODELS)
    def test_example_runs(self, dir_name, model_name):
        """Test that dp4 runs successfully on each example with specified model"""
        # Skip SGNN tests if SGNN is not available
        if model_name == "sgnn" and not sgnn_available and pytest in sys.modules:
            pytest.skip("Skipping SGNN tests because SGNN model is not available")
        
        if model_name == "cascade" and not cascade_available and pytest in sys.modules:
            pytest.skip("Skipping Cascade tests because Cascade model is not available")
            
        directory = os.path.join(EXAMPLES_DIR, dir_name)
        
        # Find SDF files in the directory
        sdf_file = f'{dir_name}_.sdf'

        # Check if the SDF file exists
        if not os.path.exists(os.path.join(directory, sdf_file)):
            pytest.skip(f"SDF file {sdf_file} not found in {directory}")
        
        command = f'pydp4 -n {dir_name}NMR -i sdf -s {sdf_file} -c {CONFIG_PATH} -w gsw --model {model_name} --remove'
        
        print(f"\n{'='*80}")
        print(f"RUNNING: {dir_name} with model {model_name}")
        print(f"COMMAND: {command}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        result = run_dp4_command(directory, command)
        
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
        
        assert result.returncode == 0, f"DP4 failed on example {dir_name} with model {model_name}. Error: {result.stderr}"
        
        assert "Program terminated normally" in all_output, f"DP4 did not terminate normally on example {dir_name} with model {model_name}"

        # Check if the data dictionary file exists
        data_dic_path = os.path.join(directory, 'dp4/data_dic.p')
        assert os.path.exists(data_dic_path), f"Data dictionary file {data_dic_path} not found in {directory}"

        # Get the MAE
        mae = get_mae(directory)
        print(f"MAE for {dir_name} is {mae}")
        assert mae < ACCEPTABLE_ERRORS[dir_name], f"MAE for {dir_name} is too high: {mae}"
        
        # Only check for fallback if using SGNN model
        if model_name == "sgnn":
            assert "Falling back to cascade model" not in all_output, f"SGNN model is not being used for example {dir_name}, falling back to cascade model"
        
if __name__ == "__main__":
    # Print information about the directory structure
    if check_directory_structure():
        # Check if pydp4 command exists
        if not check_command_exists("pydp4"):
            print("WARNING: pydp4 command not found in PATH. Make sure it's installed and in your PATH.")
            sys.exit(1)
        
        # Run the tests
        pytest.main(["-v", __file__]) 