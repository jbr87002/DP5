import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdForceFieldHelpers

from dp5.conformer_search.run_cs import conf_search
from dp5.dft.run_dft import dft_calculations
from dp5.neural_net.nn_utils import get_nn_shifts
from dp5.analysis.dp5 import DP5
from dp5.analysis.dp4 import DP4

from tqdm import tqdm
import pickle
import json
from pathlib import Path


class Molecule:
    def __init__(self, input_file: str, output_folder: str):
        self.input_file = input_file
        self.output_folder = output_folder
        self.base_name = input_file.rsplit(".", maxsplit=1)[0]
        mol_path = Path(output_folder) / input_file
        mol = Chem.MolFromMolFile(mol_path, removeHs=False)

        self.atoms = [at.GetSymbol() for at in mol.GetAtoms()]
        self.conformers = [mol.GetConformer(0).GetPositions()]
        self.charge = sum([at.GetFormalCharge() for at in mol.GetAtoms()])

        # estimates force field energy
        prop = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, prop)
        if ff is not None:
            self._energies = np.array([float(ff.CalcEnergy()) * 4.184])
        else:
            self._energies = np.array([0.0])
        # creates mol object for further manipulation
        self._rdkit_mols = None
        self._populations = None
        self._mol = mol

    def __repr__(self) -> str:
        return self.base_name

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, new_atoms):
        self._atoms = np.array(new_atoms)

    @property
    def conformers(self):
        return self._conformers

    @conformers.setter
    def conformers(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if values.shape[2] == 3:
            self._conformers = values
        else:
            raise ValueError("Cannot set coordinates!")
        self._rdkit_mols = None

    @property
    def energies(self):
        return self._energies

    @energies.setter
    def energies(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        self._energies = values
        self._populations = None

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, value):
        self._charge = int(value)

    @property
    def rdkit_mols(self):
        if self._rdkit_mols is None:
            mols = []
            for conformer in self.conformers:
                molecule = Chem.Mol(self._mol)
                conf = molecule.GetConformer(0)
                for atom, atom_coord in enumerate(conformer):
                    x, y, z = atom_coord
                    conf.SetAtomPosition(atom, Point3D(x, y, z))
                mol = Chem.Mol(molecule, confId=0)
                mols.append(mol)
            self._rdkit_mols = mols
        return self._rdkit_mols

    @property
    def conformer_C_pred(self):
        return self._conformer_C_pred

    @conformer_C_pred.setter
    def conformer_C_pred(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        self._conformer_C_pred = values

    @property
    def C_labels(self):
        return self._C_labels

    @C_labels.setter
    def C_labels(self, labels):
        self._C_labels = np.array(labels)

    @property
    def H_labels(self):
        return self._H_labels

    @H_labels.setter
    def H_labels(self, labels):
        self._H_labels = np.array(labels)

    @property
    def shielding_labels(self):
        return self._shielding_labels

    @shielding_labels.setter
    def shielding_labels(self, labels):
        self._shielding_labels = np.array(labels)

    def add_conformer_data(self, data):
        self.atoms = data.atoms
        self.conformers = data.conformers
        self.charge = data.charge
        self.energies = data.energies
        self._rdkit_mols = None

    def add_dft_data(self, data):
        attrs = (
            "atoms",
            "conformers",
            "charge",
            "energies",
            "conformer_C_pred",
            "C_labels",
            "conformer_H_pred",
            "H_labels",
            "shielding_labels",
        )
        for attr in attrs:
            if hasattr(data, attr):
                setattr(self, attr, getattr(data, attr))
        self._rdkit_mols = None

    def add_nn_shifts(self, shifts_labels):
        self.conformer_C_pred, self.C_labels, self.conformer_H_pred, self.H_labels = (
            shifts_labels
        )

    def copy(self):
        """Prevents accidental attribute override."""
        new_mol = type(self).__new__(self.__class__)
        new_mol.__dict__.update(self.__dict__)

        return new_mol

    @property
    def populations(self):
        """Assumeds kJ/mol as energy unit"""
        if self._populations is None:
            energies = self.energies
            scaling = 1000 / 8.3415 / 298.15
            energies = energies - np.min(energies)
            exp_energies = np.exp(-energies * scaling)
            self._populations = exp_energies / exp_energies.sum()
        return self._populations

    def boltzmann_weighting(self, attr: str):
        """
        Performs Boltzmann weighting across conformers
        Arguments:
        - attr: attribute name containing predictions
        Returns:
        - weighted predictions maintaining the same shape per atom
        """
        # recomputes populations just in case
        data = getattr(self, attr)
        data = np.array(data, dtype=np.float32)
        return (self.populations[:, np.newaxis] * data).sum(axis=0)

    @property
    def H_shifts(self):
        return self.boltzmann_weighting("conformer_H_pred")

    @property
    def C_shifts(self):
        return self.boltzmann_weighting("conformer_C_pred")

    def assign_nmr(self, C_exp, H_exp):
        self.C_exp, self.H_exp = np.array(C_exp, dtype=np.float32), np.array(
            H_exp, dtype=np.float32
        )

    def add_dp4_data(self, dp4_data):
        self.dp4_data = dp4_data

    def add_dp5_data(self, dp5_data):
        self.dp5_data = dp5_data


class Molecules:
    """Class that handles all the calculations. Should keep the molecular data in itself"""

    def __init__(self, config):
        self.config = config
        mols_list_iterator = tqdm(self.config["structure"], desc="Loading molecules", total=len(self.config["structure"]))
        self.mols = [Molecule(mol, config["output_folder"]) for mol in mols_list_iterator]

    def __iter__(self):
        return (mol.copy() for mol in self.mols)

    def __getitem__(self, idx):
        return self.mols[idx]

    def save(self, filepath=None):
        """Save the Molecules object to a pickle file.
        
        Args:
            filepath: Path to save the pickle file. If None, uses the output folder from config.
        """
        if filepath is None:
            filepath = Path(self.config["output_folder"]) / "molecules.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """Load a Molecules object from a pickle file.
        
        Args:
            filepath: Path to the pickle file.
            
        Returns:
            Molecules: The loaded Molecules object.
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_conformers(self):
        """Runs conformational search."""
        mm_data = conf_search(self.mols, self.config["conformer_search"])
        for mol, data in zip(self.mols, mm_data):
            mol.add_conformer_data(data)
        
        # Save checkpoint after conformer search
        if self.config.get("save_checkpoints", False):
            self.save(Path(self.config["output_folder"]) / "molecules_after_conformers.pkl")

    def get_dft_data(self):
        """Runs DFT calculations"""
        dft_mols = [mol for mol in self.mols]
        dft_data = dft_calculations(
            dft_mols, self.config["workflow"], self.config["dft"]
        )
        for mol, data in zip(self.mols, dft_data):
            mol.add_dft_data(data)
            
        # Save checkpoint after DFT calculations
        if self.config.get("save_checkpoints", False):
            self.save(Path(self.config["output_folder"]) / "molecules_after_dft.pkl")

    def get_nn_nmr_shifts(self):
        """Should get C and H shifts"""
        mols = [mol.rdkit_mols for mol in self.mols]
        cascade_shifts_labels = get_nn_shifts(mols, model=self.config["nn_model"]["model"], n_forward_pass=self.config["nn_model"]["n_forward_pass"])
        for mol, *m_shift_label in zip(self.mols, *cascade_shifts_labels):
            mol.add_nn_shifts(m_shift_label)
            
        # Save checkpoint after NN NMR shifts
        if self.config.get("save_checkpoints", False):
            self.save(Path(self.config["output_folder"]) / "molecules_after_nn_nmr.pkl")
            
    def assign_nmr_spectra(self, nmrdata):
        for mol in self.mols:
            C_exp, H_exp = nmrdata.assign(mol)
            mol.assign_nmr(C_exp, H_exp)

    def dp5_analysis(self):
        dp5 = DP5(self.config["output_folder"], self.config["workflow"]["dft_nmr"], self.config["nn_model"]["model"])
        self.dp5_output = dp5(self.mols)

    def dp4_analysis(self):
        dp4 = DP4(self.config["output_folder"], self.config["dp4"])
        self.dp4_output = dp4(self.mols)

    def print_results(self):
        output = "Workflow summary\n\n"
        output += f"Solvent = {self.config['solvent']}\n"
        if self.config["workflow"]["dft_opt"]:
            output += (
                f"DFT optimisation functional: {self.config['dft']['o_functional']}\n"
                f"DFT optimisation basis set: {self.config['dft']['o_basis_set']}\n"
            )
        if self.config["workflow"]["dft_energies"]:
            output += (
                f"DFT energy functional: {self.config['dft']['e_functional']}\n"
                f"DFT energy basis set: {self.config['dft']['e_basis_set']}\n"
            )
        if self.config["workflow"]["dft_nmr"]:
            output += (
                f"DFT NMR functional: {self.config['dft']['n_functional']}\n"
                f"DFT NMR basis set: {self.config['dft']['n_basis_set']}\n"
            )
        if self.config["dp4"]["param_file"] != "none":
            output += (
                f"\nDP4 statistical model file: {self.config['dp4']['param_file']}\n"
            )

        output += f"\nNumber of candidates: {len(self.mols)}\n"

        for i, mol in enumerate(self.mols):
            output += (
                f"Number of conformers for molecule {mol}: {len(mol.conformers)}\n"
            )

        if self.config["workflow"]["dp4"]:
            dp4_output = output + self.dp4_output
            with open((self.config["output_folder"]) / "output.dp4", "w") as f:
                f.write(dp4_output)

        if self.config["workflow"]["dp5"]:
            dp5_output = output + self.dp5_output
            with open((self.config["output_folder"]) / "output.dp5", "w") as f:
                f.write(dp5_output)

    def save_nmr_shifts_sdf(self, directory=None):
        """Save NMR shifts to SDF files, including molecular structure and properties.
        
        Args:
            directory: Directory to save the SDF files. If None, uses the output folder from config.
            
        Returns:
            str: Path to the saved SDF file.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if directory is None:
            directory = Path(self.config["output_folder"])
        else:
            directory = Path(directory)
            
        directory.mkdir(parents=True, exist_ok=True)
        
        # Create SDF file path
        sdf_file = directory / "nmr_shifts.sdf"
        
        # Create SDF writer
        writer = Chem.SDWriter(str(sdf_file))
        
        # Process each molecule
        logger.info(f"Saving NMR shifts for {len(self.mols)} molecules to SDF")
        
        try:
            mol_iterator = tqdm(self.mols, desc="Saving molecules to SDF", unit="molecule")
        except NameError:
            mol_iterator = self.mols
            
        for mol in mol_iterator:
            # Get a copy of the molecule to add properties to
            rdkit_mol = Chem.Mol(mol._mol)
            
            # Add molecule name as a property
            rdkit_mol.SetProp("_Name", mol.base_name)
            
            # Add InChI key as a property
            mol_without_hs = Chem.RemoveHs(mol._mol)
            inchi_key = Chem.MolToInchiKey(mol_without_hs)
            rdkit_mol.SetProp("INCHIKEY", inchi_key)
            
            # Add SMILES as a property
            smiles = Chem.MolToSmiles(mol_without_hs)
            rdkit_mol.SetProp("SMILES", smiles)
            
            # Add carbon shifts if available
            if hasattr(mol, 'C_shifts') and hasattr(mol, 'C_labels'):
                # Add each carbon shift as a separate property
                for atom_idx, (label, shift) in enumerate(zip(mol.C_labels, mol.C_shifts)):
                    atom_num = int(label[1:]) - 1  # Convert C1 to atom index 0
                    rdkit_mol.SetProp(f"C_SHIFT_{atom_num}", f"{shift:.2f}")
                
                # Also add as a single JSON property for easier parsing
                carbon_shifts_json = json.dumps({
                    int(label[1:])-1: round(float(shift), 2) 
                    for label, shift in zip(mol.C_labels, mol.C_shifts)
                })
                rdkit_mol.SetProp("CARBON_SHIFTS_JSON", carbon_shifts_json)
            
            # Add proton shifts if available
            if hasattr(mol, 'H_shifts') and hasattr(mol, 'H_labels'):
                # Add each proton shift as a separate property
                for atom_idx, (label, shift) in enumerate(zip(mol.H_labels, mol.H_shifts)):
                    atom_num = int(label[1:]) - 1  # Convert H1 to atom index 0
                    rdkit_mol.SetProp(f"H_SHIFT_{atom_num}", f"{shift:.2f}")
                
                # Also add as a single JSON property for easier parsing
                proton_shifts_json = json.dumps({
                    int(label[1:])-1: round(float(shift), 2) 
                    for label, shift in zip(mol.H_labels, mol.H_shifts)
                })
                rdkit_mol.SetProp("PROTON_SHIFTS_JSON", proton_shifts_json)
            
            # Write the molecule to the SDF file
            writer.write(rdkit_mol)
        
        # Close the writer
        writer.close()
        
        logger.info(f"NMR shifts saved to SDF file: {sdf_file}")
        return str(sdf_file)
        
    def load_nmr_shifts_sdf(self, sdf_file=None):
        """Load NMR shifts from an SDF file.
        
        Args:
            sdf_file: Path to the SDF file. If None, uses the default path in the output folder.
            
        Returns:
            tuple: (bool, list) - Success flag and list of molecules with missing shifts.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if sdf_file is None:
            sdf_file = Path(self.config["output_folder"]) / "nmr_shifts.sdf"
        else:
            sdf_file = Path(sdf_file)
        
        # Check if file exists
        if not sdf_file.exists():
            logger.warning(f"NMR shift SDF file not found: {sdf_file}")
            return False, []
        
        # Create a dictionary to map InChI keys to shift data
        shifts_data = {}
        
        # Read the SDF file
        try:
            logger.info(f"Reading NMR shifts from SDF file: {sdf_file}")
            sdf_supplier = Chem.SDMolSupplier(str(sdf_file))
            
            for sdf_mol in sdf_supplier:
                if sdf_mol is None:
                    continue
                
                # Get InChI key
                if sdf_mol.HasProp("INCHIKEY"):
                    inchi_key = sdf_mol.GetProp("INCHIKEY")
                else:
                    # Generate InChI key if not present
                    inchi_key = Chem.MolToInchiKey(sdf_mol)
                
                # Get carbon shifts
                carbon_shifts = {}
                if sdf_mol.HasProp("CARBON_SHIFTS_JSON"):
                    carbon_shifts = json.loads(sdf_mol.GetProp("CARBON_SHIFTS_JSON"))
                else:
                    # Try to get individual carbon shift properties
                    for prop_name in sdf_mol.GetPropNames():
                        if prop_name.startswith("C_SHIFT_"):
                            atom_idx = int(prop_name.split("_")[-1])
                            carbon_shifts[atom_idx] = float(sdf_mol.GetProp(prop_name))
                
                # Get proton shifts
                proton_shifts = {}
                if sdf_mol.HasProp("PROTON_SHIFTS_JSON"):
                    proton_shifts = json.loads(sdf_mol.GetProp("PROTON_SHIFTS_JSON"))
                else:
                    # Try to get individual proton shift properties
                    for prop_name in sdf_mol.GetPropNames():
                        if prop_name.startswith("H_SHIFT_"):
                            atom_idx = int(prop_name.split("_")[-1])
                            proton_shifts[atom_idx] = float(sdf_mol.GetProp(prop_name))
                
                # Store the data
                shifts_data[inchi_key] = {
                    "carbon_shifts": carbon_shifts,
                    "proton_shifts": proton_shifts
                }
            
        except Exception as e:
            logger.error(f"Error reading SDF file: {e}")
            return False, []
        
        # Map molecules to their InChI keys
        mol_inchi_map = {}
        for mol in self.mols:
            mol_without_hs = Chem.RemoveHs(mol._mol)
            inchi_key = Chem.MolToInchiKey(mol_without_hs)
            mol_inchi_map[inchi_key] = mol
        
        # Check if we have shifts for all molecules
        missing_mols = []
        for inchi_key, mol in mol_inchi_map.items():
            if inchi_key not in shifts_data:
                missing_mols.append(mol.base_name)
        
        if missing_mols:
            logger.warning(f"Missing shifts for {len(missing_mols)}/{len(self.mols)} molecules")
            if len(missing_mols) == len(self.mols):
                logger.error("No shifts found for any molecules")
                return False, missing_mols
        
        # Apply shifts to molecules
        shifts_loaded = False
        
        try:
            mol_iterator = tqdm(mol_inchi_map.items(), desc="Loading NMR shifts from SDF", unit="molecule")
        except NameError:
            mol_iterator = mol_inchi_map.items()
            
        for inchi_key, mol in mol_iterator:
            if inchi_key in shifts_data:
                # Process carbon shifts
                carbon_shifts = shifts_data[inchi_key]["carbon_shifts"]
                if carbon_shifts:
                    c_shifts = []
                    c_labels = []
                    
                    # Sort by atom index to ensure correct order
                    for atom_idx in sorted([int(idx) for idx in carbon_shifts.keys()]):
                        c_labels.append(f"C{atom_idx+1}")
                        c_shifts.append(float(carbon_shifts[str(atom_idx)]))
                    
                    # Create conformer predictions (assuming single conformer for loaded shifts)
                    mol.C_labels = np.array(c_labels)
                    mol.conformer_C_pred = np.array([c_shifts])
                    
                    shifts_loaded = True
                
                # Process proton shifts
                proton_shifts = shifts_data[inchi_key]["proton_shifts"]
                if proton_shifts:
                    h_shifts = []
                    h_labels = []
                    
                    # Sort by atom index to ensure correct order
                    for atom_idx in sorted([int(idx) for idx in proton_shifts.keys()]):
                        h_labels.append(f"H{atom_idx+1}")
                        h_shifts.append(float(proton_shifts[str(atom_idx)]))
                    
                    # Create conformer predictions (assuming single conformer for loaded shifts)
                    mol.H_labels = np.array(h_labels)
                    mol.conformer_H_pred = np.array([h_shifts])
                    
                    shifts_loaded = True
        
        if shifts_loaded:
            logger.info("Successfully loaded NMR shifts from SDF file")
            return True, []
        else:
            logger.warning("No shifts were loaded from SDF file")
            return False, missing_mols
