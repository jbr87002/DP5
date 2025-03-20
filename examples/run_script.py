import os
import sys
import time
from pathlib import Path
import shutil

# Add the parent directory to the path so we can import dp5
sys.path.insert(0, str(Path(__file__).parent.parent))

from dp5.load_config import run_workflow
from rdkit import Chem

nmr_file = "/scratch/jbr46/np_atlas_nmr/NMR_example"
precalculated_sdf = "/scratch/jbr46/np_atlas_nmr/nmr_shifts.sdf"
output_path = "/scratch/jbr46/np_atlas_nmr/dp5_precalc_testing"
data = run_workflow(
    structure_files=['/scratch/jbr46/np_atlas_nmr/np_atlas_100.smi'],
    nmr_files=[nmr_file],
    output_path=output_path,
    input_type='smiles',
    workflow='w',  # DP4 analysis,
    remove_previous=True,
    precalculated_sdf=precalculated_sdf,
    model='cascade'
)
