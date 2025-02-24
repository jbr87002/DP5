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
import time
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

def load_model_metadata(model_path):
    """
    Loads model metadata (dimensions and training statistics) from a JSON file
    Arguments:
    - model_path: path to model file (metadata will be in same directory)
    Returns:
    - dict containing model metadata
    """
    metadata_path = Path(__file__).parent / (Path(model_path).stem + '_metadata.json')
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_NMR_prediction_model(model_path):
    """
    Loads SGNN model from the given path
    Arguments:
    - model_path: path to model file
    Returns:
    - loaded model, training statistics
    """
    # Load model metadata
    metadata = load_model_metadata(model_path)
    
    node_dim = metadata['node_dim']
    edge_dim = metadata['edge_dim']
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, metadata['train_y_mean'], metadata['train_y_std']

def MC_dropout(model):
    """Enable MC dropout during inference"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def inference(model, loader, train_y_mean, train_y_std, n_forward_pass=5, device=None):
    """
    Run inference with MC dropout
    Arguments:
    - model: trained model
    - loader: DataLoader with test data
    - train_y_mean: mean of training targets
    - train_y_std: std of training targets
    - n_forward_pass: number of forward passes for MC dropout
    - device: torch device
    Returns:
    - predictions: numpy array of predictions
    - time_per_mol: average time per molecule
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    MC_dropout(model)
    predictions = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            inputs = batch[0].to(device)
            n_nodes = batch[1].to(device)
            masks = batch[-1].to(device)

            mean_list = []
            for _ in range(n_forward_pass):
                mean = model(inputs, n_nodes, masks)
                mean_list.append(mean.cpu().numpy())
            
            predictions.append(np.array(mean_list).mean(axis=0))

    predictions = np.vstack(predictions) * train_y_std + train_y_mean
    time_per_mol = (time.time() - start_time) / len(loader.dataset)

    return predictions, time_per_mol

def _prepare_molecule_graph(test_df):
    """Helper function to prepare molecule graph dictionary"""
    mol_dict = {
        'n_node': [], 'n_edge': [], 'node_attr': [], 'edge_attr': [],
        'src': [], 'dst': [], 'shift': [], 'mask': [], 'smi': []
    }
    
    total_atoms = 0
    for _, row in test_df.iterrows():
        mol = row['Mol']
        
        # Add dummy shifts and masks for prediction
        for atom in mol.GetAtoms():
            atom.SetProp('shift', '0.0')
            atom.SetBoolProp('mask', True)
        
        mol = Chem.RemoveHs(mol)
        total_atoms += len(mol.GetAtoms())
        mol_dict = add_mol_sparsified_graph(mol_dict, mol, '13C')
    
    return mol_dict, total_atoms

def _convert_to_numpy(mol_dict):
    """Helper function to convert dictionary lists to numpy arrays"""
    return {
        'n_node': np.array(mol_dict['n_node']).astype(int),
        'n_edge': np.array(mol_dict['n_edge']).astype(int),
        'node_attr': np.vstack(mol_dict['node_attr']).astype(bool),
        'edge_attr': np.vstack(mol_dict['edge_attr']).astype(bool),
        'src': np.hstack(mol_dict['src']).astype(int),
        'dst': np.hstack(mol_dict['dst']).astype(int),
        'shift': np.hstack(mol_dict['shift']),
        'mask': np.hstack(mol_dict['mask']).astype(bool),
        'smi': np.array(mol_dict['smi'])
    }

def predict_shifts(model, test_df, train_y_mean, train_y_std, batch_size=16):
    """
    Predicts shifts for molecules in test_df using SGNN model
    Arguments:
    - model: trained model
    - test_df: DataFrame containing molecules to predict
    - train_y_mean: mean of training targets
    - train_y_std: std of training targets
    - batch_size: batch size for predictions
    Returns:
    - predictions_by_mol: list of numpy arrays containing predictions for each molecule
    """
    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Prepare molecule graphs
    mol_dict, total_atoms = _prepare_molecule_graph(test_df)
    mol_dict = _convert_to_numpy(mol_dict)
    
    # Create dataset and dataloader
    dataset = GraphDataset('13C', 'sparsified', mol_dict=mol_dict)
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        collate_fn=collate_reaction_graphs
    )
    
    # Get predictions
    predictions, _ = inference(model, loader, train_y_mean, train_y_std, n_forward_pass=5, device=device)

    # Validate number of predictions
    if len(predictions) != total_atoms:
        raise ValueError(f'Number of predictions ({len(predictions)}) does not match number of atoms ({total_atoms})')
    
    # Reshape predictions to match input format
    predictions_by_mol = []
    shift_count = 0

    for mol_id, group in test_df.groupby('mol_id'):
        mol = Chem.RemoveHs(group['Mol'].iloc[0])
        atom_count = len(mol.GetAtoms())
        
        # Extract predictions for current molecule
        mol_predictions = predictions[shift_count:shift_count + atom_count]
        predictions_by_mol.append(mol_predictions)
        
        # Update shift count for next molecule
        shift_count += atom_count
    
    return predictions_by_mol

def get_shifts_and_labels_sgnn(mols, atomic_symbol, model_path, batch_size=16):
    """
    Predicts shifts from rdkit Mol objects using SGNN model
    Arguments:
    - list of lists of RDKit mol objects
    Returns:
    - list of list of lists of 13C chemical shifts for each atom in a molecule
    - list of lists of C atomic labels
    """
    model, train_y_mean, train_y_std = load_NMR_prediction_model(model_path)
    logger.info("Loaded NMR prediction model")

    all_df, all_labels = mols_to_df(mols, atomic_symbol)
    logger.info(f"Ready to predict shifts for {atomic_symbol}")
    all_shifts = predict_shifts(model, all_df, train_y_mean, train_y_std, batch_size=batch_size)
    # just take the median predictions
    median_idx = len(all_shifts[0]) // 2

    # Filter shifts based on labels
    filtered_shifts = []
    for mol_shifts, mol_labels in zip(all_shifts, all_labels):
        # mol_shifts is a list of lists
        # keep just the middle value from each sub-list
        mol_shifts = [shift[median_idx] for shift in mol_shifts]
        # Convert labels like 'C2' to indices (subtract 1 to get 0-based index)
        indices = [int(label[1:]) - 1 for label in mol_labels]
        
        # Create boolean mask for the atoms we want to keep
        mol_shifts = np.array(mol_shifts)
        filtered_mol_shifts = mol_shifts[indices]
        filtered_shifts.append([filtered_mol_shifts])

    return filtered_shifts, all_labels

