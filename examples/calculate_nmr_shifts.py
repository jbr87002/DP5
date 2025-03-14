#!/usr/bin/env python
"""
Example script demonstrating how to calculate NMR shifts without NMR data
and how to use checkpoints to save/load calculation state.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import dp5
sys.path.insert(0, str(Path(__file__).parent.parent))

from dp5.load_config import run_workflow

def main():
    # Example SMILES strings
    smiles_file = "/scratch/jbr46/np_atlas_nmr/np_atlas.smi"
    
    # Create the file if it doesn't exist
    if not os.path.exists(smiles_file):
        with open(smiles_file, "w") as f:
            f.write("CC(=O)OC1=CC=CC=C1C(=O)O\n")  # Aspirin
            f.write("CC1=C(C=C(C=C1)S(=O)(=O)N)CC(=O)NC2=CC=C(C=C2)OCC3=CC=CC=C3\n")  # Glipizide
    
    # Output directory
    output_dir = "/scratch/jbr46/np_atlas_nmr"
    
    # Run the workflow without NMR data, just calculating shifts
    # m = conformational search, n = DFT NMR (disabled here)
    data = run_workflow(
        structure_files=[smiles_file],
        nmr_files=None,  # No NMR data
        output_path=output_dir,
        input_type="smiles",
        workflow="N",  # Just calculate NMR shifts
        skip_nmr=True,  # Skip NMR processing
        model='sgnn'
    )
    
    print("\nNMR shifts have been calculated and saved to JSON files in:", output_dir)
    print("- carbon_shifts.json: Contains all carbon NMR shifts")
    print("- proton_shifts.json: Contains all proton NMR shifts")
    print("The Molecules object has been saved to checkpoints for later use.")
    
    # Example of loading from a checkpoint
    checkpoint_file = Path(output_dir) / "molecules_final.pkl"
    if checkpoint_file.exists():
        print(f"\nTo load from the checkpoint in another script:")
        print(f"from dp5.run.data_structures import Molecules")
        print(f"data = Molecules.load('{checkpoint_file}')")
        print(f"# Now you can access the data directly")
        print(f"# For example: data.mols[0].C_shifts")
        
        # Example of reading the JSON files
        print(f"\nTo read the JSON files in another script:")
        print(f"import json")
        print(f"with open('{output_dir}/carbon_shifts.json', 'r') as f:")
        print(f"    carbon_shifts = json.load(f)")
        print(f"with open('{output_dir}/proton_shifts.json', 'r') as f:")
        print(f"    proton_shifts = json.load(f)")
        print(f"# Access shifts by InChI key")
        print(f"# For example: carbon_shifts['INCHIKEY']['shifts']")
        print(f"# The molecule name is stored in: carbon_shifts['INCHIKEY']['name']")
        print(f"# The SMILES is stored in: carbon_shifts['INCHIKEY']['smiles']")

if __name__ == "__main__":
    main() 