#!/usr/bin/env python
"""
Example script demonstrating how to load NMR shifts from JSON files
instead of recalculating them with the neural network.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import dp5
sys.path.insert(0, str(Path(__file__).parent.parent))

from dp5.load_config import run_workflow

def main():
    # Example SMILES strings
    smiles_file = "examples/example_smiles.txt"
    
    # Create the file if it doesn't exist
    if not os.path.exists(smiles_file):
        with open(smiles_file, "w") as f:
            f.write("CC(=O)OC1=CC=CC=C1C(=O)O\n")  # Aspirin
            f.write("CC1=C(C=C(C=C1)S(=O)(=O)N)CC(=O)NC2=CC=C(C=C2)OCC3=CC=CC=C3\n")  # Glipizide
    
    # nmr file
    nmr_file = "examples/example_nmr.txt"
    
    # Output directory
    output_dir = "examples/nmr_shifts_output"
    
    # First, check if we've already calculated shifts
    shifts_dir = Path(output_dir)
    carbon_file = shifts_dir / "carbon_shifts.json"
    proton_file = shifts_dir / "proton_shifts.json"
    
    if carbon_file.exists() and proton_file.exists():
        print(f"Found existing NMR shift files in {output_dir}")
        print("Will load shifts from cache instead of recalculating")
        
        # Run the workflow with shifts_from_cache=True
        data = run_workflow(
            structure_files=[smiles_file],
            nmr_files=[nmr_file],
            output_path=output_dir,
            input_type="smiles",
            workflow="s",  # Just do conformational search, no DFT
            save_checkpoints=True,
            model='sgnn',
            shifts_from_cache=True
        )
        
if __name__ == "__main__":
    main() 