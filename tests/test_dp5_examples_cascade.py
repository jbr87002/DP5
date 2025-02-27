import os
import sys
import pytest
import subprocess
import tempfile
import shutil
from pathlib import Path

# Path to the examples directory
EXAMPLES_DIR = "/scratch/jbr46/structure_reassignment_examples"
CONFIG_PATH = os.path.join(EXAMPLES_DIR, "config.toml")

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

class TestDP5Examples:
    """Test class for running dp5 on example data"""
    
    def setup_method(self):
        """Setup method that runs before each test"""
        if not os.path.isdir(EXAMPLES_DIR):
            pytest.skip(f"Examples directory {EXAMPLES_DIR} not found")
            
        if not os.path.isfile(CONFIG_PATH):
            pytest.skip(f"Config file {CONFIG_PATH} not found")
    
    @pytest.mark.parametrize("example_num", range(1, 25))
    def test_example_runs(self, example_num):
        """Test that dp5 runs successfully on each example"""
        directory = os.path.join(EXAMPLES_DIR, f"S{example_num}")
        
        if not os.path.isdir(directory):
            pytest.skip(f"Example directory S{example_num} not found")
        
        if example_num == 13:
            # Modified command for S13
            command = f'pydp4 -n S13a_NMR -i sdf -s S13a_.sdf S13b_.sdf -c {CONFIG_PATH} -w cw --model cascade --remove'
        else:
            command = f'pydp4 -n S{example_num}_NMR -i sdf -s S{example_num}a_.sdf S{example_num}b_.sdf -c {CONFIG_PATH} -w cw --model cascade --remove'
        
        result = run_dp5_command(directory, command)
        
        assert result.returncode == 0, f"DP5 failed on example S{example_num} with error: {result.stderr}"

        assert "Program terminated normally" in result.stdout, f"DP5 did not terminate normally on example S{example_num}"
        
        # Additional check: verify that output files were created
        # This can be expanded based on what files you expect to be created

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 