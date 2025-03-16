import logging
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
    
    # Use SDMolSupplier to read the SDF file
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
                    inchi_key_index[inchi_key] = mol_idx
                except Exception as e:
                    logger.warning(f"Error computing InChI key for molecule in SDF: {e}")
            
            # Get NPA number if present
            if mol.HasProp("_Name"):
                name = mol.GetProp("_Name")
                if name.startswith("NPA") and any(c.isdigit() for c in name):
                    npa_index[name] = mol_idx
    
    logger.info(f"Found {len(inchi_key_index)} molecules with InChI keys in precalculated SDF file")
    logger.info(f"Found {len(npa_index)} molecules with NPA numbers in precalculated SDF file")
    
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