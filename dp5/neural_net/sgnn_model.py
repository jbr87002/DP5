import pandas as pd
import logging
import numpy as np
import torch
from pathlib import Path
from rdkit import Chem
from .CNN_model import mols_to_df
from .sgnn.mpnn_proposed import nmr_mpnn_PROPOSED
from dgl.data.utils import split_dataset
from dgllife.utils import RandomSplitter
from .sgnn.util import collate_reaction_graphs
from .sgnn.nmrshiftdb2_get_data import add_mol_sparsified_graph, add_mol_fully_connected_graph
from .sgnn.dataset import GraphDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def load_NMR_prediction_model(model_path):
    """
    Loads SGNN model from the given path
    Arguments:
    - model_path: path to model file
    Returns:
    - loaded model
    """
    # Default parameters - these should match your training configuration
    node_dim = 256
    edge_dim = 256
    readout_mode = 'proposed'
    node_embedding_dim = 256
    readout_n_hidden_dim = 256
    quantiles = np.linspace(0.005, 0.995, 100)
    
    model = nmr_mpnn_PROPOSED(
        node_dim, 
        edge_dim, 
        readout_mode,
        node_embedding_dim,
        readout_n_hidden_dim,
        quantiles=quantiles
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path(__file__).parent / model_path
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_shifts(model, test_df, batch_size=16):
    """
    Predicts shifts for molecules in test_df using SGNN model
    Arguments:
    - model: loaded SGNN model
    - test_df: DataFrame with mol_id, conf_id, Mol, atom_index columns
    - batch_size: batch size for predictions
    Returns:
    - list of lists of predicted shifts for each molecule
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Process molecules into graph format
    mol_dict = {
        'n_node': [], 'n_edge': [], 'node_attr': [], 'edge_attr': [],
        'src': [], 'dst': [], 'shift': [], 'mask': [], 'smi': []
    }
    
    for _, row in test_df.iterrows():
        mol = row['Mol']
        atom_indices = row['atom_index']
        
        # Add dummy shifts and masks for prediction
        for atom in mol.GetAtoms():
            atom.SetProp('shift', '0.0')
            atom.SetBoolProp('mask', True)
        
        mol = Chem.RemoveHs(mol)
        mol_dict = add_mol_sparsified_graph(mol_dict, mol, 'C')
    
    # Convert lists to numpy arrays
    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    mol_dict['shift'] = np.hstack(mol_dict['shift'])
    mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
    mol_dict['smi'] = np.array(mol_dict['smi'])
    
    # Create dataset and dataloader
    dataset = GraphDataset('13C', 'sparsified', mol_dict=mol_dict)
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        collate_fn=collate_reaction_graphs
    )
    
    # Get predictions
    with torch.no_grad():
        all_predictions = []
        for batch in loader:
            predictions = model(batch)
            # Get median predictions (index 50 for 100 quantiles)
            median_preds = predictions[:, 50].cpu().numpy()
            all_predictions.extend(median_preds)
    
    # Reshape predictions to match input format
    predictions_by_mol = []
    start_idx = 0
    for mol_id, group in test_df.groupby('mol_id'):
        n_atoms = len(group['atom_index'].iloc[0])
        mol_preds = all_predictions[start_idx:start_idx + n_atoms]
        predictions_by_mol.append(mol_preds)
        start_idx += n_atoms
    
    return predictions_by_mol

def get_shifts_and_labels_sgnn(mols, atomic_symbol, model_path, batch_size=16):
    """
    Predicts shifts from rdkit Mol objects using SGNN model
    Arguments:
    - list of lists of RDKit mol objects
    Returns:
    - list of lists of 13C chemical shifts for each atom in a molecule
    - list of lists of C atomic labels
    """
    model = load_NMR_prediction_model(model_path)
    logger.info("Loaded NMR prediction model")

    all_df, all_labels = mols_to_df(mols, atomic_symbol)
    logger.info(f"Ready to predict shifts for {atomic_symbol}")
    all_shifts = predict_shifts(model, all_df, batch_size=batch_size)

    return all_shifts, all_labels

