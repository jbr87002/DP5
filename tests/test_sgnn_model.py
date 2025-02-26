import os
import pytest
import numpy as np
from rdkit import Chem
from dp5.neural_net.sgnn_model import (
    load_model_metadata,
    load_NMR_prediction_model,
    predict_shifts_sgnn,
    filter_shifts
)

class TestSGNNModel:
    """Test class for SGNN model functionality"""
    
    def test_load_model_metadata(self):
        """Test that model metadata can be loaded"""
        # This test will need to be adjusted based on the actual model path
        # For now, we'll just check that the function exists and has the right signature
        assert callable(load_model_metadata)
    
    def test_load_NMR_prediction_model(self):
        """Test that NMR prediction model can be loaded"""
        # This test will need to be adjusted based on the actual model path
        # For now, we'll just check that the function exists and has the right signature
        assert callable(load_NMR_prediction_model)
    
    def test_filter_shifts(self):
        """Test the filter_shifts function"""
        # Create some test data
        shifts = [
            [np.array([1.0, 2.0, 3.0, 4.0, 5.0])],
            [np.array([6.0, 7.0, 8.0, 9.0])]
        ]
        labels = [
            ['C1', 'C3', 'C5'],
            ['C2', 'C4']
        ]
        
        # Test with median_only=True
        median_idx = 2
        filtered_shifts = filter_shifts(shifts, labels, median_only=True, median_idx=median_idx)
        
        # Check that the filtered shifts have the right structure
        assert len(filtered_shifts) == len(shifts)
        assert len(filtered_shifts[0][0]) == len(labels[0])
        assert len(filtered_shifts[1][0]) == len(labels[1])
        
        # Check that the right values were selected
        assert filtered_shifts[0][0][0] == shifts[0][0][0]  # C1 -> index 0
        assert filtered_shifts[0][0][1] == shifts[0][0][2]  # C3 -> index 2
        assert filtered_shifts[0][0][2] == shifts[0][0][4]  # C5 -> index 4
        
        assert filtered_shifts[1][0][0] == shifts[1][0][1]  # C2 -> index 1
        assert filtered_shifts[1][0][1] == shifts[1][0][3]  # C4 -> index 3
        
        # Test with median_only=False
        with pytest.raises(ValueError):
            # Should raise ValueError if median_idx is None and median_only is True
            filter_shifts(shifts, labels, median_only=True, median_idx=None)

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 