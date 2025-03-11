#!/usr/bin/env python
"""
Example script demonstrating how to load from a checkpoint and continue processing.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import dp5
sys.path.insert(0, str(Path(__file__).parent.parent))

from dp5.load_config import run_workflow
from dp5.run.data_structures import Molecules

def main():
    # Output directory from previous run
    output_dir = "examples/nmr_shifts_output"
    checkpoint_file = Path(output_dir) / "molecules_final.pkl"
    
    if not checkpoint_file.exists():
        print(f"Checkpoint file {checkpoint_file} not found.")
        print("Please run calculate_nmr_shifts.py first.")
        return
    
    print(f"Loading from checkpoint: {checkpoint_file}")
    
    # Method 1: Load directly using Molecules.load()
    data = Molecules.load(checkpoint_file)
    print("Loaded Molecules object directly.")
    print(f"Number of molecules: {len(data.mols)}")
    
    # Access data directly
    for i, mol in enumerate(data.mols):
        print(f"\nMolecule {i+1}:")
        if hasattr(mol, 'C_shifts'):
            print(f"  Carbon shifts: {mol.C_shifts}")
        if hasattr(mol, 'H_shifts'):
            print(f"  Proton shifts: {mol.H_shifts}")
    
    # Method 2: Use run_workflow with load_checkpoint
    # This is useful if you want to continue processing with different parameters
    print("\nResuming workflow from checkpoint...")
    
    # Example NMR file (you would provide your own)
    nmr_file = "examples/example_nmr.txt"
    
    # Create a dummy NMR file if it doesn't exist
    if not Path(nmr_file).exists():
        with open(nmr_file, "w") as f:
            f.write("# This is a placeholder NMR file\n")
            f.write("# In a real scenario, you would provide actual NMR data\n")
            f.write("C,130.5,128.2,133.7\n")
            f.write("H,7.2,7.5,8.1,2.1\n")
    
    # Resume workflow with NMR data for DP4/DP5 analysis
    # Note: This would normally fail without the checkpoint because we're skipping conformer search
    data = run_workflow(
        structure_files=["examples/example_smiles.txt"],  # Same structure file
        nmr_files=nmr_file,  # Now providing NMR data
        output_path=output_dir,
        input_type="smiles",
        workflow="ws",  # DP4 (s) and DP5 (w) analysis
        load_checkpoint=checkpoint_file,  # Load from checkpoint
        save_checkpoints=True  # Continue saving checkpoints
    )
    
    print("\nWorkflow completed with DP4/DP5 analysis.")

if __name__ == "__main__":
    main() 