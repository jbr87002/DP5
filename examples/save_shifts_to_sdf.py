#!/usr/bin/env python
"""
Example script demonstrating how to save and load NMR shifts in SDF format.
"""

import os
import sys
from pathlib import Path
import time

# Add the parent directory to the path so we can import dp5
sys.path.insert(0, str(Path(__file__).parent.parent))

from dp5.load_config import run_workflow
from dp5.run.data_structures import Molecules
from rdkit import Chem

def main():
    # Example SMILES strings
    smiles_file = "examples/example_smiles.txt"
    
    # Create the file if it doesn't exist
    if not os.path.exists(smiles_file):
        with open(smiles_file, "w") as f:
            f.write("CC(=O)OC1=CC=CC=C1C(=O)O\n")  # Aspirin
            f.write("CC1=C(C=C(C=C1)S(=O)(=O)N)CC(=O)NC2=CC=C(C=C2)OCC3=CC=CC=C3\n")  # Glipizide
    
    # Output directory
    output_dir = "examples/nmr_shifts_output"
    
    # Run the workflow to calculate NMR shifts
    print("\nCalculating NMR shifts and saving to SDF...")
    data = run_workflow(
        structure_files=[smiles_file],
        nmr_files=None,  # No NMR data
        output_path=output_dir,
        input_type="smiles",
        workflow="m",  # Just do conformational search, no DFT
        skip_nmr=True,  # Skip NMR processing
        model='sgnn'
    )
    
    # Save the shifts to SDF format
    sdf_file = data.save_nmr_shifts_sdf()
    print(f"\nNMR shifts saved to SDF file: {sdf_file}")
    
    # Demonstrate how to read the SDF file directly with RDKit
    print("\nReading SDF file with RDKit to show the properties:")
    sdf_supplier = Chem.SDMolSupplier(sdf_file)
    
    for i, mol in enumerate(sdf_supplier):
        if mol is None:
            continue
            
        print(f"\nMolecule {i+1}: {mol.GetProp('_Name')}")
        print(f"InChI Key: {mol.GetProp('INCHIKEY')}")
        print(f"SMILES: {mol.GetProp('SMILES')}")
        
        # Print all properties that start with C_SHIFT or H_SHIFT
        print("\nCarbon Shifts:")
        carbon_shifts = []
        for prop_name in mol.GetPropNames():
            if prop_name.startswith("C_SHIFT_"):
                atom_idx = int(prop_name.split("_")[-1])
                shift_value = float(mol.GetProp(prop_name))
                carbon_shifts.append((atom_idx, shift_value))
        
        # Sort by atom index
        for atom_idx, shift in sorted(carbon_shifts):
            print(f"  Atom {atom_idx}: {shift:.2f} ppm")
        
        print("\nProton Shifts:")
        proton_shifts = []
        for prop_name in mol.GetPropNames():
            if prop_name.startswith("H_SHIFT_"):
                atom_idx = int(prop_name.split("_")[-1])
                shift_value = float(mol.GetProp(prop_name))
                proton_shifts.append((atom_idx, shift_value))
        
        # Sort by atom index
        for atom_idx, shift in sorted(proton_shifts):
            print(f"  Atom {atom_idx}: {shift:.2f} ppm")
    
    # Now demonstrate loading the shifts back into the Molecules object
    print("\nLoading shifts from SDF back into a new Molecules object...")
    
    # Create a new Molecules object
    new_data = run_workflow(
        structure_files=[smiles_file],
        nmr_files=None,
        output_path=output_dir + "_reload",
        input_type="smiles",
        workflow="m",  # Just do conformational search, no DFT
        skip_nmr=True,
        model='sgnn',
        shifts_from_cache=True  # This will try to load from cache
    )
    
    print("\nShifts successfully loaded from SDF file!")
    
    # Compare a few shifts between the original and loaded data
    print("\nComparing shifts from original calculation vs. loaded from SDF:")
    for i, (orig_mol, new_mol) in enumerate(zip(data.mols, new_data.mols)):
        print(f"\nMolecule {i+1}: {orig_mol.base_name}")
        
        if hasattr(orig_mol, 'C_shifts') and hasattr(new_mol, 'C_shifts'):
            print("Carbon shifts (first 3):")
            for j in range(min(3, len(orig_mol.C_shifts))):
                print(f"  {orig_mol.C_labels[j]}: {orig_mol.C_shifts[j]:.2f} ppm (original) vs {new_mol.C_shifts[j]:.2f} ppm (loaded)")

if __name__ == "__main__":
    main() 