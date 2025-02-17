import logging

from .CNN_model import get_shifts_and_labels_cascade
from .sgnn_model import get_shifts_and_labels_sgnn


logger = logging.getLogger(__name__)


def get_nn_shifts(mols, batch_size=16, model='cascade'):
    """
    Predicts shifts from rdkit Mol objects.
    Arguments:
    - list of lists of RDKit mol objects
    Returns:
    - list of lists of 13C chemical shifts for each atom in a molecule
    - list of lists of C atomic labels
    - list of lists of 1H chemical shifts for each atom in a molecule
    - list of lists of H atomic labels
    """

    C_shifts, C_labels = predict_C_shifts(mols, batch_size, model)

    H_shifts, H_labels = predict_H_shifts(mols, batch_size, model)

    # will add H_shifts and H_labels later!

    return C_shifts, C_labels, H_shifts, H_labels


def predict_C_shifts(mols, batch_size, model):
    model_paths = {
        "cascade": "NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
        "sgnn": "sgnn_13c.npz.pt"
    }
    if model == 'cascade':
        return get_shifts_and_labels_cascade(
            mols,
            atomic_symbol="C",
            model_path=model_paths[model],
            batch_size=batch_size,
        )
    elif model == 'sgnn':
        return get_shifts_and_labels_sgnn(
            mols,
            atomic_symbol="C",
            model_path=model_paths[model],
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Model {model} not supported")


def predict_H_shifts(mols, batch_size, model):
    return [[[]]] * len(mols), [[]] * len(mols)
