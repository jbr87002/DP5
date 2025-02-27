import logging

CASCADE_AVAILABLE = False

try:
    from .CNN_model import get_shifts_and_labels_cascade
    CASCADE_AVAILABLE = True
except ImportError:
    def get_shifts_and_labels_cascade(*args, **kwargs):
        raise ImportError("Cascade model dependencies are not installed. "
                         "Please use the 'sgnn' model instead.")

SGNN_AVAILABLE = False

# Try to import sgnn_model, but don't fail if it's not available
try:
    from .sgnn_model import get_shifts_and_labels_sgnn
    SGNN_AVAILABLE = True
except ImportError:
    def get_shifts_and_labels_sgnn(*args, **kwargs):
        raise ImportError("SGNN model dependencies (dgl, torch, etc.) are not installed. "
                         "Please use the 'cascade' model instead.")


logger = logging.getLogger(__name__)


def get_nn_shifts(mols, batch_size=16, model='cascade', n_forward_pass=50):
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

    C_shifts, C_labels = predict_C_shifts(mols, batch_size, model, n_forward_pass)

    H_shifts, H_labels = predict_H_shifts(mols, batch_size, model, n_forward_pass)

    # will add H_shifts and H_labels later!

    return C_shifts, C_labels, H_shifts, H_labels


def predict_C_shifts(mols, batch_size, model, n_forward_pass):
    model_paths = {
        "cascade": "NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
        "sgnn": "sgnn_13c.pt"
    }
    if model == 'cascade':
        return get_shifts_and_labels_cascade(
            mols,
            atomic_symbol="C",
            model_path=model_paths[model],
            batch_size=batch_size,
        )
    elif model == 'sgnn':
        if not SGNN_AVAILABLE:
            logger.warning("SGNN model dependencies not available. Falling back to cascade model.")
            model = 'cascade'
            return get_shifts_and_labels_cascade(
                mols,
                atomic_symbol="C",
                model_path=model_paths[model],
                batch_size=batch_size,
            )
        else:
            return get_shifts_and_labels_sgnn(
                mols,
                atomic_symbol="C",
                model_path=model_paths[model],
                batch_size=batch_size,
                n_forward_pass=n_forward_pass
            )
    else:
        raise ValueError(f"Model {model} not supported")


def predict_H_shifts(mols, batch_size, model, n_forward_pass):
    return [[[]]] * len(mols), [[]] * len(mols)
