#!/usr/bin/env python
"""
Example script demonstrating how to use pre-calculated NMR shift data.

This script shows how to:
1. Use pre-calculated data from an SDF file
2. Skip the expensive conformational search and neural network prediction steps
3. Go directly to NMR analysis using the pre-calculated shifts
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import dp5
sys.path.insert(0, str(Path(__file__).parent.parent))

from dp5.load_config import run_workflow
from rdkit import Chem

def main():
    # Example SMILES strings
    smiles_file = "examples/example_smiles.txt"
    
    # Create the file if it doesn't exist
    if not os.path.exists(smiles_file):
        with open(smiles_file, "w") as f:
            f.write("CC(=O)OC1=CC=CC=C1C(=O)O\n")  # Aspirin
            f.write("CC1=C(C=C(C=C1)S(=O)(=O)N)CC(=O)NC2=CC=C(C=C2)OCC3=CC=CC=C3\n")  # Glipizide
    
    # Example NMR file
    nmr_file = "examples/example_nmr.txt"
    
    # Create a dummy NMR file if it doesn't exist
    if not os.path.exists(nmr_file):
        with open(nmr_file, "w") as f:
            f.write("# This is a placeholder NMR file\n")
            f.write("# In a real scenario, you would provide actual NMR data\n")
            f.write("C,130.5,128.2,133.7\n")
            f.write("H,7.2,7.5,8.1,2.1\n")
    
    # Output directory
    output_dir = "examples/precalculated_example"
    
    # First, let's generate some pre-calculated data if it doesn't exist
    precalc_sdf = Path("examples/precalculated_shifts.sdf")
    
    if not precalc_sdf.exists():
        print("\nGenerating pre-calculated data...")
        print("This would normally be done once and then reused many times.")
        
        # Run the workflow to calculate NMR shifts and save to SDF
        data = run_workflow(
            structure_files=[smiles_file],
            nmr_files=None,  # No NMR data for this step
            output_path="examples/temp_calc",
            input_type="smiles",
            workflow="m",  # Just do conformational search, no DFT
            skip_nmr=True,  # Skip NMR processing
            model='sgnn'
        )
        
        # Save the shifts to our pre-calculated SDF file
        sdf_file = data.save_nmr_shifts_sdf(precalc_sdf.parent)
        
        # Rename the file to our desired name
        os.rename(sdf_file, precalc_sdf)
        print(f"Pre-calculated data saved to {precalc_sdf}")
    
    # Now demonstrate using the pre-calculated data
    print("\nUsing pre-calculated data for analysis...")
    print("This is the fast path that skips expensive calculations.")
    
    # Time the fast path
    start_time = time.time()
    
    # Run the workflow using pre-calculated data
    data = run_workflow(
        structure_files=[smiles_file],
        nmr_files=nmr_file,  # Now we provide NMR data for analysis
        output_path=output_dir,
        input_type="smiles",
        workflow="sw",  # DP4 (s) and DP5 (w) analysis
        use_precalculated=True,  # Use pre-calculated data
        precalculated_sdf=str(precalc_sdf)  # Specify the pre-calculated SDF file
    )
    
    fast_time = time.time() - start_time
    print(f"\nAnalysis with pre-calculated data completed in {fast_time:.2f} seconds")
    
    # For comparison, let's time the standard workflow
    print("\nFor comparison, running the standard workflow...")
    print("This would normally be much slower due to conformational search and neural network prediction.")
    
    # Time the standard workflow
    start_time = time.time()
    
    # Run the workflow without using pre-calculated data
    data_standard = run_workflow(
        structure_files=[smiles_file],
        nmr_files=nmr_file,
        output_path=output_dir + "_standard",
        input_type="smiles",
        workflow="msw",  # Conformational search (m), DP4 (s) and DP5 (w) analysis
        use_precalculated=False
    )
    
    standard_time = time.time() - start_time
    print(f"\nStandard workflow completed in {standard_time:.2f} seconds")
    print(f"Using pre-calculated data was {standard_time/fast_time:.1f}x faster!")
    
    # Show how to use pre-calculated data with SDF input directly
    print("\nYou can also use SDF files directly with pre-calculated shifts:")
    print("data = run_workflow(")
    print(f"    structure_files=['{precalc_sdf}'],")
    print("    nmr_files=nmr_file,")
    print("    input_type='sdf',")
    print("    workflow='sw',  # Just DP4 (s) and DP5 (w) analysis")
    print("    use_precalculated=True")
    print(")")

if __name__ == "__main__":
    main() 