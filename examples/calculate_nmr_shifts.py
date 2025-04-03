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
    smiles_file = "/scratch/jbr46/sarolt_compounds/input.smi"
    
    output_dir = "/scratch/jbr46/sarolt_compounds"
    # precalculated_sdf = "/scratch/jbr46/np_atlas_nmr/nmr_shifts.sdf"
    precalculated_sdf = None

    # Run the workflow without NMR data, just calculating shifts
    # N = neural network NMR prediction
    data = run_workflow(
        structure_files=[smiles_file],
        nmr_files=None,  # No NMR data
        output_path=output_dir,
        input_type="smiles",
        workflow="N",  # Just calculate NMR shifts
        model='cascade',
        precalculated_sdf=precalculated_sdf
    )
    
if __name__ == "__main__":
    main() 