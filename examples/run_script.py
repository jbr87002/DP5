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
data = run_workflow(
    structure_files=['/scratch/jbr46/np_atlas_nmr/np_atlas_1000_10.smi'],
    nmr_files=[nmr_file],
    input_type='smiles',
    workflow='s',  # DP4 analysis
    precalculated_sdf=precalculated_sdf,
    model='sgnn'
)
