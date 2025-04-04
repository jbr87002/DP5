import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdForceFieldHelpers

from dp5.conformer_search.run_cs import conf_search
from dp5.dft.run_dft import dft_calculations
from dp5.neural_net.nn_utils import get_nn_shifts
from dp5.analysis.dp5 import DP5
from dp5.analysis.dp4 import DP4
from dp5.run.utils import build_sdf_index, get_inchi_key, get_sdf_indices

from tqdm import tqdm
import pickle
import json
import logging
from pathlib import Path
import os

class Molecule:
    def __init__(self, input_file: str, output_folder: str):
        logger = logging.getLogger(__name__)
        self.input_file = input_file
        self.output_folder = output_folder
        self.base_name = input_file.rsplit(".", maxsplit=1)[0]
        mol_path = os.path.join(output_folder, input_file)
        mol = Chem.MolFromMolFile(mol_path, removeHs=False)

        self.atoms = [at.GetSymbol() for at in mol.GetAtoms()]
        self.conformers = [mol.GetConformer(0).GetPositions()]
        self.charge = sum([at.GetFormalCharge() for at in mol.GetAtoms()])

        # estimates force field energy
        prop = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, prop)
        if ff is not None:
            logger.debug(f'FF is not None')
            energies = float(ff.CalcEnergy()) * 4.184
            logger.debug(f'Calculated energies: {energies}')
            if not np.isnan(energies):
                self._energies = np.array([energies])
            else:
                logger.debug(f'Energies is nan, setting to 0.0')
                self._energies = np.array([0.0])
        else:
            logger.debug(f'FF is None')
            self._energies = np.array([0.0])
        logger.debug(f'Energies: {self._energies}')
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

    @property
    def conformer_H_pred(self):
        return self._conformer_H_pred

    @conformer_H_pred.setter
    def conformer_H_pred(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        self._conformer_H_pred = values

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
        logger = logging.getLogger(__name__)
        logger.debug(f'\nC_shifts before boltzmann weighting: {self.conformer_C_pred}')
        logger.debug(f'Populations: {self.populations}')
        logger.debug(f'Energies: {self.energies}')
        logger.debug(f'C_shifts after boltzmann weighting: {self.boltzmann_weighting("conformer_C_pred")}\n')
        return self.boltzmann_weighting("conformer_C_pred")

    def assign_nmr(self, C_exp, H_exp):
        # Convert None values to NaN for proper float array handling
        import numpy as np
        
        # Convert NoneType values to NaN
        if C_exp:
            C_exp_processed = np.array([np.nan if x is None else x for x in C_exp], dtype=np.float32)
        else:
            C_exp_processed = np.array([], dtype=np.float32)
            
        if H_exp:
            H_exp_processed = np.array([np.nan if x is None else x for x in H_exp], dtype=np.float32)
        else:
            H_exp_processed = np.array([], dtype=np.float32)
            
        self.C_exp, self.H_exp = C_exp_processed, H_exp_processed

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
        
    def get_dft_data(self):
        """Runs DFT calculations"""
        dft_mols = [mol for mol in self.mols]
        dft_data = dft_calculations(
            dft_mols, self.config["workflow"], self.config["dft"]
        )
        for mol, data in zip(self.mols, dft_data):
            mol.add_dft_data(data)
            
    def get_nn_nmr_shifts(self):
        """
        Get NMR shifts for molecules that don't already have pre-calculated shifts.
        
        This method overrides the parent class method to avoid regenerating shifts
        for molecules that already have pre-calculated shifts.
        """
        logger = logging.getLogger(__name__)
        
        # Check which molecules need shifts generated
        mols_needing_shifts = []
        indices_needing_shifts = []
        
        for i, mol in enumerate(self.mols):
            if not self.has_precalculated_shifts(mol):
                mols_needing_shifts.append(mol.rdkit_mols)
                indices_needing_shifts.append(i)
        
        if not mols_needing_shifts:
            logger.info("All molecules have pre-calculated shifts, skipping shift generation")
            return
        
        logger.info(f"Generating shifts for {len(mols_needing_shifts)} molecules without pre-calculated data")
        
        # Generate shifts only for molecules that need them
        cascade_shifts_labels = get_nn_shifts(
            mols_needing_shifts, 
            model=self.config["nn_model"]["model"], 
            n_forward_pass=self.config["nn_model"]["n_forward_pass"]
        )
        logger.debug(f"Cascade shifts labels: {cascade_shifts_labels}\n")
        
        # Assign the generated shifts to the appropriate molecules
        for idx, *m_shift_label in zip(indices_needing_shifts, *cascade_shifts_labels):
            self.mols[idx].add_nn_shifts(m_shift_label)
        
    def assign_nmr_spectra(self, nmrdata):
        logger = logging.getLogger(__name__)
        for mol in self.mols:
            C_exp, H_exp = nmrdata.assign(mol)
            logger.debug(f"C_exp: {C_exp}, H_exp: {H_exp}\n")
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
        logger = logging.getLogger(__name__)
        
        if directory is None:
            directory = Path(self.config["output_folder"])
        else:
            directory = Path(directory)
            
        directory.mkdir(parents=True, exist_ok=True)
        
        sdf_file = directory / "nmr_shifts.sdf"
        
        writer = Chem.SDWriter(str(sdf_file))
        
        logger.info(f"Saving NMR shifts for {len(self.mols)} molecules to SDF")
        
        for mol in self.mols:
            rdkit_mol = Chem.Mol(mol._mol)
            
            rdkit_mol.SetProp("_Name", mol.base_name)
            
            mol_without_hs = Chem.RemoveHs(mol._mol)
            inchi_key = Chem.MolToInchiKey(mol_without_hs)
            rdkit_mol.SetProp("INCHIKEY", inchi_key)
            
            smiles = Chem.MolToSmiles(mol_without_hs)
            rdkit_mol.SetProp("SMILES", smiles)

            if hasattr(mol, 'C_shifts') and hasattr(mol, 'C_labels'):
                carbon_shifts_json = json.dumps({
                    int(label[1:])-1: round(float(shift), 2) 
                    for label, shift in zip(mol.C_labels, mol.C_shifts)
                })
                rdkit_mol.SetProp("CARBON_SHIFTS_JSON", carbon_shifts_json)
            
            if hasattr(mol, 'H_shifts') and hasattr(mol, 'H_labels'):
                proton_shifts_json = json.dumps({
                    int(label[1:])-1: round(float(shift), 2) 
                    for label, shift in zip(mol.H_labels, mol.H_shifts)
                })
                rdkit_mol.SetProp("PROTON_SHIFTS_JSON", proton_shifts_json)
            
            writer.write(rdkit_mol)
        
        writer.close()
        
        logger.info(f"NMR shifts saved to SDF file: {sdf_file}")
        inchi_key_index, npa_index = build_sdf_index(sdf_file)
        return str(sdf_file)

    def has_precalculated_shifts(self, mol):
        """
        Check if a molecule has pre-calculated NMR shifts.
        
        Args:
            mol: Molecule object to check
            
        Returns:
            bool: True if the molecule has pre-calculated shifts, False otherwise
        """
        return (hasattr(mol, 'conformer_C_pred') and hasattr(mol, 'C_labels') and 
                len(getattr(mol, 'conformer_C_pred', [])) > 0)


class Molecules_precalculated(Molecules):
    """
    Class that handles molecules with pre-calculated NMR shifts.
    
    This class can load molecules either from SDF files or directly from a pre-calculated SDF file
    using InChI keys or NPA identifiers as identifiers.
    """
    def __init__(self, config):
        """
        Initialize the Molecules_precalculated object.
        
        Args:
            config: Configuration dictionary containing structure files and other settings
        """
        self.config = config
        
        precalculated_sdf = config.get("precalculated_sdf")
        
        logger = logging.getLogger(__name__)
        
        self.inchi_key_index = {}
        self.npa_index = {}
        self.sdf_supplier = None
        
        if precalculated_sdf and precalculated_sdf.get("path"):
            sdf_path = precalculated_sdf["path"]
            self.inchi_key_index, self.npa_index, self.sdf_supplier = get_sdf_indices(sdf_path)
        
        mols_list_iterator = tqdm(self.config["structure"], desc="Loading molecules", total=len(self.config["structure"]))
        
        # Initialize molecules
        self.mols = []
        for mol in mols_list_iterator:
            try:
                self.mols.append(self._create_molecule(mol, config["output_folder"], 
                                                      precalculated_sdf.get("path") if precalculated_sdf else None))
            except ValueError as e:
                logger.warning(f"Error loading molecule {mol}: {str(e)}")
                logger.warning(f"Initializing as regular Molecule instead")
                self.mols.append(Molecule(mol, config["output_folder"]))
        
        logger.info(f"Loaded {len(self.mols)} molecules")
        
        # Count molecules with pre-calculated shifts
        precalc_count = sum(1 for mol in self.mols if self.has_precalculated_shifts(mol))
        if precalc_count > 0:
            logger.info(f"{precalc_count} molecules have pre-calculated NMR shifts")
        else:
            logger.info("No molecules have pre-calculated NMR shifts")
    
    def _create_molecule(self, input_file, output_folder, precalculated_sdf):
        """
        Create a Molecule object with optimized SDF access.
        
        Args:
            input_file: Path to the molecule file or InChI key or NPA identifier
            output_folder: Output folder for the molecule
            precalculated_sdf: Path to the precalculated SDF file
            
        Returns:
            Molecule: The created molecule object
        """
        # Check if input_file is an SDF file or an identifier
        if input_file.endswith('.sdf'):
            # Initialize normally from SDF file
            return Molecule(input_file, output_folder)
        else:
            # Check if input_file is an npaid
            is_npa = input_file.startswith('NPA') and any(c.isdigit() for c in input_file)
            
            if precalculated_sdf is None or self.sdf_supplier is None:
                raise ValueError("precalculated_sdf must be provided when initializing from identifier")
            
            # Find the molecule with matching identifier in the precalculated SDF file
            mol_idx = None
            if is_npa and input_file in self.npa_index:
                mol_idx = self.npa_index[input_file]
            elif input_file in self.inchi_key_index:
                mol_idx = self.inchi_key_index[input_file]
            else:
                raise ValueError(f"Molecule with identifier {input_file} not found in precalculated SDF file")
            
            # Get the molecule at the specified index
            mol = self.sdf_supplier[mol_idx]
            
            if mol is None:
                raise ValueError(f"Failed to load molecule with identifier {input_file} from precalculated SDF file")
            
            molecule = Molecule.__new__(Molecule)
            molecule.input_file = input_file
            molecule.output_folder = output_folder
            if mol.HasProp("_Name"):
                molecule.base_name = mol.GetProp("_Name")
            else:
                molecule.base_name = input_file
            
            molecule._mol = mol
            molecule.atoms = [at.GetSymbol() for at in mol.GetAtoms()]
            molecule.conformers = [mol.GetConformer(0).GetPositions()]
            molecule.charge = sum([at.GetFormalCharge() for at in mol.GetAtoms()])
            
            if mol.HasProp("CARBON_SHIFTS_JSON") and mol.HasProp("PROTON_SHIFTS_JSON"):
                carbon_shifts = json.loads(mol.GetProp("CARBON_SHIFTS_JSON"))
                proton_shifts = json.loads(mol.GetProp("PROTON_SHIFTS_JSON"))
                
                if carbon_shifts:
                    c_shifts = []
                    c_labels = []
                    
                    for atom_idx in sorted([int(idx) for idx in carbon_shifts.keys()]):
                        c_labels.append(f"C{atom_idx+1}")
                        c_shifts.append(float(carbon_shifts[str(atom_idx)]))
                    
                    # Create conformer predictions (assume single conformer for loaded shifts)
                    molecule._C_labels = np.array(c_labels)
                    molecule._conformer_C_pred = np.array([c_shifts])
                else:
                    molecule._C_labels = np.array([])
                    molecule._conformer_C_pred = np.array([])
                
                if proton_shifts:
                    h_shifts = []
                    h_labels = []
                    
                    for atom_idx in sorted([int(idx) for idx in proton_shifts.keys()]):
                        h_labels.append(f"H{atom_idx+1}")
                        h_shifts.append(float(proton_shifts[str(atom_idx)]))
                    
                    # Create conformer predictions (assume single conformer for loaded shifts)
                    molecule._H_labels = np.array(h_labels)
                    molecule._conformer_H_pred = np.array([h_shifts])
                else:
                    molecule._H_labels = np.array([])
                    molecule._conformer_H_pred = np.array([])
            
            try:
                prop = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
                ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, prop)
                if ff is not None:
                    energies = float(ff.CalcEnergy()) * 4.184
                    if not np.isnan(energies):
                        molecule._energies = np.array([energies])
                    else:
                        molecule._energies = np.array([0.0])
                else:
                    molecule._energies = np.array([0.0])
            except:
                molecule._energies = np.array([0.0])
            
            molecule._rdkit_mols = None
            molecule._populations = None
            
            return molecule