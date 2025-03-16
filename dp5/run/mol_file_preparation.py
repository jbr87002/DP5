from pathlib import Path
import logging
from typing import List, Union, Dict

from rdkit import Chem
from rdkit.Chem import AllChem, EnumerateStereoisomers
from tqdm import tqdm
from dp5.run.utils import build_sdf_index, get_inchi_key


logger = logging.getLogger(__name__)


def write_to_sdf(mol: Chem.rdchem.Mol, relative_path: Path, output_folder: str):
    """
    Writes rdkit Mol object to a specified path

    arguments:
    - mol: RDKit Mol object
    - relative_path: path to write the file relative to the output folder

    returns:
    - input_file: relative path to file from current working directory
    """
    file_path = Path(output_folder) / relative_path
    writer = Chem.SDWriter(str(file_path))
    writer.write(mol)
    return file_path


def cleanup_3d(mol, ignore_sanitise_error=False):
    """
    Generates a 3D conformer of a molecule.

    arguments:
    - mol: RDKit mol object
    - ignore_sanitise_error: If True, return None on error instead of raising an exception

    returns:
    - a reasonable conformer, or None if there was an error and ignore_sanitise_error is True
    """
    try:
        mol = AllChem.AddHs(mol, addCoords=True)
        cid = AllChem.EmbedMolecule(
            mol,
            randomSeed=0,
            forceTol=0.0135,
        )
        if cid == -1:
            if ignore_sanitise_error:
                logger.warning("Molecule could not be sanitised")
                return None
            else:
                raise ValueError("Molecule could not be sanitised")
        
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.rdmolops.AssignStereochemistryFrom3D(mol)
        return mol
    except Exception as e:
        if ignore_sanitise_error:
            logger.warning(f"Error optimising molecule: {e}")
            return None
        else:
            raise ValueError(f"Error optimising molecule: {e}")


def read_sdf(input_file: str):

    fullf = Path(input_file).resolve()
    mol = Chem.MolFromMolFile(str(fullf), removeHs=False, sanitize=True)

    return mol


def read_textfile(input_file: str, input_type: str):
    """
    Reads a text file containing SMILES, SMARTS, or InChI strings.

    arguments:
    - input_file: path to the text file
    - input_type: type of input file. May be 'smiles', 'smarts', or 'inchi'

    returns:
    - mols: list of RDKit Mol objects

    input file format:
    name,smiles
    If name is not provided, a name is generated automatically.
    """
    input_type = input_type.lower()

    input_readers = {
        "smiles": Chem.MolFromSmiles,
        "smarts": Chem.MolFromSmarts,
        "inchi": Chem.MolFromInchi,
    }

    if input_type not in input_readers:
        raise ValueError(f"Cannot parse {input_type}")

    mols = []

    with open(input_file) as f:
        for line in f:
            if line.strip() == "":
                continue
            try:
                parts = line.strip().split(",")
                if len(parts) == 1:
                    smiles = parts[0]
                    name = None
                else:
                    name, smiles = parts[0], parts[1]
                mol = input_readers[input_type](smiles, sanitize=True)
                mol = Chem.AddHs(mol)
                if name is not None:
                    mol.SetProp("_Name", name)
                mols.append(mol)
            except Exception as e:
                # to make an logging.error
                logging.error(f"Error reading line: {line} - {e}")

    return mols


def _generate_diastereomers(
    mol: Chem.rdchem.Mol,
    mutable_atoms: Union[List[int], None] = None,
    double_bonds: bool = False,
) -> List[Chem.rdchem.Mol]:
    """
    Enumerates all diastereomers and double bond isomers.

    Given an rdkit Mol object, generates all possible diastereomers and returns them as a list.
    Note that the function also checks if the resulting molecule can be embedded.

    Parameters:
    - mol (rdkit.Chem.rdchem.Mol): the molecule to have its stereocentres enumerated
    - protected_atoms (list[int], None): the list of atoms with a fixed stereochemistry, numbering starts with 1
    - double_bonds (bool): if double bond configuration is also altered. Defaults to False

    Returns:
    list of Mols
    """
    # copy the input to prevent accidental editing
    target_mol = AllChem.AddHs(mol)

    mutable_atoms = mutable_atoms.copy()

    enum_opts = EnumerateStereoisomers.StereoEnumerationOptions(
        tryEmbedding=False, onlyUnassigned=True, maxIsomers=0, unique=True
    )
    # extract chiral information if present in a molecule
    atom_chiral = [at.GetChiralTag() for at in target_mol.GetAtoms()]
    bond_geometry = [b.GetStereo() for b in target_mol.GetBonds()]

    # Creates list of mutable atoms. If none are specified, retains everything but the first stereocentre.
    atoms_with_stereo = [
        i
        for i, tag in enumerate(atom_chiral, start=1)
        if tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
    ]

    if not mutable_atoms:
        if len(atoms_with_stereo) > 0:
            atoms_with_stereo.pop(0)
        mutable_atoms = atoms_with_stereo

    mutable_atoms = [ind - 1 for ind in mutable_atoms]

    # all other atoms will have stereochemistry tag reset
    for atom in target_mol.GetAtoms():
        if atom.GetIdx() in mutable_atoms:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

    # scrambles all double bonds: if particular configuration was assigned, it is reset
    if double_bonds == True:
        [
            b.SetStereo(Chem.rdchem.BondStereo.STEREOANY)
            for b, g in zip(target_mol.GetBonds(), bond_geometry)
            if g != Chem.rdchem.BondStereo.STEREONONE
        ]

    result = []
    for isomer in EnumerateStereoisomers.EnumerateStereoisomers(
        target_mol, options=enum_opts
    ):
        isomer.RemoveAllConformers()
        cid = AllChem.EmbedMolecule(isomer, randomSeed=0, forceTol=0.0135)
        if cid >= 0:
            AllChem.MMFFOptimizeMolecule(isomer)
            Chem.rdmolops.AssignStereochemistryFrom3D(isomer)
            result.append(isomer)

    return result


def prepare_inputs(
    input_files: List[str],
    input_type: str,
    stereocentres: List[int],
    workflow: Dict,
    nn_model: Dict, 
    output_folder: str = None,
    precalculated_sdf: str = None,
) -> List[str]:
    """
    Reads files at the path specified by input config, prepares them as required. Returns paths to the new files.

    Arguments:
    - input_file (list[str]): list of relative paths to structure input files. Contains one text file of several SD Files.
    - input_type (str): format of the input file. May be 'sdf', 'smiles', 'inchi', and 'smarts'.
    - stereocentres (list[int]): specifies mutable stereocentres. Defaults to empty list
    - workflow (dict): dictionary of booleans specifying the workflow.
    - precalculated_sdf (dict or None): dictionary with path to precalculated SDF file

    Returns:
    - mol_paths (list[str]): paths to the transformed files
    """
    # if not sdf, read text
    # generate diastereomers (cleans them) and clean inputs
    # returns paths to inputs
    # in principle, can create list of list of mols, use enumerate
    logger.info(f"Read structures from {input_files}")

    if input_type == "sdf":
        mols = [read_sdf(file) for file in input_files]
        logger.debug("read structures from SD File")
    else:
        mols = read_textfile(input_files[0], input_type)
        logger.debug(f"read structures from {input_type} file")
        input_files = [
            f"{mol.GetProp('_Name')}.sdf" if mol.HasProp('_Name') else f"{input_type}_mol_{i:03}_.sdf"
            for i, mol in enumerate(mols, start=1)
        ]

    if len(mols) < 1:
        raise ValueError("No molecules were provided!")
    
    # check whether each molecule is in the precalculated sdf file
    if precalculated_sdf and precalculated_sdf.get("path"):
        precalculated = check_mols_in_sdf(mols, precalculated_sdf["path"])
    else:
        precalculated = [False] * len(mols)

    logger.info(f"Structures read successfully")

    mols2 = []
    mutable_atoms = stereocentres if len(input_files) == 1 else []

    if workflow["generate"]:
        logger.info("Generating diastereomers")
        mols2 = [_generate_diastereomers(mol, mutable_atoms) for mol in mols]
    elif workflow["cleanup"] or (
        not workflow["conf_search"] and not workflow["dft_opt"] and nn_model["model"] == "cascade"
    ):
        logger.info("Generating MMFF geometries for inputs")
        # ignore sanitise error if just calculating NMR shifts
        ignore_sanitise_error = workflow["save_shifts"]
        mol_iterator = tqdm(zip(mols, precalculated), desc="Generating MMFF geometries", unit="molecule", total=len(mols))
        
        mols2 = []
        for mol, precalc in mol_iterator:
            if precalc:
                # If molecule is in precalculated SDF, use it as is
                mols2.append([mol])
            else:
                # Otherwise, try to clean it up
                cleaned_mol = cleanup_3d(mol, ignore_sanitise_error=ignore_sanitise_error)
                if cleaned_mol is not None:
                    mols2.append([cleaned_mol])
                else:
                    # Skip this molecule if cleanup failed
                    logger.warning("Skipping molecule due to sanitization error")
    else:
        mols2 = [[mol] for mol in mols]

    logger.debug("Preparing to write structure files")
    filenames = []
    for filename, mol, precalc in tqdm(zip(input_files, mols2, precalculated), desc="Writing structure files", total=len(input_files)):
        for i, isomer in enumerate(mol, start=1):
            if precalc:
                # if name is not provided, use InChI key, else use NPA number (name is NPA number)
                if isomer.HasProp("_Name"):
                    name = isomer.GetProp("_Name")
                    if name.startswith("NPA") and any(c.isdigit() for c in name):
                        fname = name
                    else:
                        fname = Chem.MolToInchiKey(isomer)
                else:
                    fname = Chem.MolToInchiKey(isomer)
            else:
                if len(mol) == 1:
                    fname = f"{filename[:-4]}.sdf"
                else:
                    fname = f"{filename[:-4]}isomer{i:03}.sdf"
                write_to_sdf(isomer, fname, output_folder)
            filenames.append(fname)

    return filenames

def check_mols_in_sdf(mols, precalculated_sdf):
    """
    Check whether each molecule is in the precalculated SDF file
    returns a list of booleans
    """
    logger = logging.getLogger(__name__)
    
    # Build indices of InChI keys and NPA numbers in the SDF file
    inchi_key_index, npa_index, _ = build_sdf_index(precalculated_sdf)
    
    if not inchi_key_index and not npa_index:
        # If no indices were built, return all False
        return [False] * len(mols)
    
    # Pre-compute InChI keys for all molecules at once to avoid redundant calculations
    mol_inchi_keys = [get_inchi_key(mol) for mol in mols]
    
    # Check if each molecule is in the index by InChI key or NPA number
    precalculated = []
    for mol, inchi_key in zip(mols, mol_inchi_keys):
        if inchi_key is not None and inchi_key in inchi_key_index:
            precalculated.append(True)
        elif mol.HasProp("_Name"):
            name = mol.GetProp("_Name")
            if name.startswith("NPA") and any(c.isdigit() for c in name) and name in npa_index:
                precalculated.append(True)
            else:
                precalculated.append(False)
        else:
            precalculated.append(False)
    
    # Log how many molecules were found in the precalculated SDF
    found_count = sum(1 for p in precalculated if p)
    logger.info(f"Found {found_count} out of {len(mols)} molecules in precalculated SDF file")
    
    return precalculated
