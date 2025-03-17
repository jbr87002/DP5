import logging
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from rdkit import Chem
from tqdm import tqdm

def build_sdf_index(sdf_path: str) -> Tuple[Dict[str, int], Dict[str, int], Optional[Chem.SDMolSupplier]]:
    """
    Build indices of InChI keys and NPA numbers in the SDF file for faster lookup.
    
    Args:
        sdf_path: Path to the SDF file
        
    Returns:
        Tuple containing:
        - Dictionary mapping InChI keys to molecule indices in the SDF file
        - Dictionary mapping NPA numbers to molecule indices in the SDF file
        - SDMolSupplier for the SDF file (or None if the file doesn't exist)
    """
    logger = logging.getLogger(__name__)
    
    if not sdf_path or not Path(sdf_path).exists():
        logger.warning(f"SDF file not found: {sdf_path}")
        return {}, {}, None
    
    logger.info(f"Building indices of InChI keys and NPA numbers in SDF file: {sdf_path}")
    
    inchi_key_index = {}
    npa_index = {}
    
    with Chem.SDMolSupplier(str(sdf_path)) as sdf_reader:
        for mol_idx, mol in enumerate(tqdm(sdf_reader, desc="Indexing SDF file", unit="molecule")):
            if mol is None:
                continue
            
            # Get InChI key
            if mol.HasProp("INCHIKEY"):
                inchi_key = mol.GetProp("INCHIKEY")
            else:
                # Generate InChI key if not present
                try:
                    inchi_key = Chem.MolToInchiKey(mol)
                except Exception as e:
                    logger.warning(f"Error computing InChI key for molecule in SDF: {e}")
                    continue
            inchi_key_index[inchi_key] = mol_idx
            
            # Get npaid if present
            if mol.HasProp("_Name"):
                name = mol.GetProp("_Name")
                if name.startswith("NPA") and any(c.isdigit() for c in name):
                    npa_index[name] = mol_idx
    
    logger.info(f"Found {len(inchi_key_index)} molecules with InChI keys in precalculated SDF file")
    logger.info(f"Found {len(npa_index)} molecules with NPA numbers in precalculated SDF file")

    json.dump(inchi_key_index, open(f"{Path(sdf_path).with_suffix('.inchi_key_index.json')}", 'w'))
    json.dump(npa_index, open(f"{Path(sdf_path).with_suffix('.npa_index.json')}", 'w'))
    
    return inchi_key_index, npa_index

def get_sdf_indices(sdf_path: str) -> Tuple[Dict[str, int], Dict[str, int], Optional[Chem.SDMolSupplier]]:
    """
    Build indices of InChI keys and NPA numbers in the SDF file for faster lookup.
    
    Args:
        sdf_path: Path to the SDF file
        
    Returns:
        Tuple containing:
        - Dictionary mapping InChI keys to molecule indices in the SDF file
        - Dictionary mapping NPA numbers to molecule indices in the SDF file
        - SDMolSupplier for the SDF file (or None if the file doesn't exist)
    """
    # load indices for precalculated SDF file
    inchi_key_index_path = Path(sdf_path).with_suffix('.inchi_key_index.json')
    npa_index_path = Path(sdf_path).with_suffix('.npa_index.json')
    if not Path(inchi_key_index_path).exists() or not Path(npa_index_path).exists():
        inchi_key_index, npa_index = build_sdf_index(sdf_path)
    else:
        inchi_key_index = json.load(open(inchi_key_index_path))
        npa_index = json.load(open(npa_index_path))
    
    # Create a new SDMolSupplier for actual use
    sdf_supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    
    return inchi_key_index, npa_index, sdf_supplier

def get_inchi_key(mol) -> Optional[str]:
    """
    Get the InChI key for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        InChI key string or None if it couldn't be computed
    """
    try:
        return Chem.MolToInchiKey(mol)
    except Exception as e:
        logging.warning(f"Error computing InChI key: {e}")
        return None 