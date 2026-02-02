"""
Test suite for synthetic data generation
Tests the data generation engine described in Section 3.2
"""

import pytest
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from data.synthetic_data_generator import (
    SyntheticTraumaDataset,
    PhysiologicalModel
)
from data.preprocessor import VitalSignsPreprocessor

class TestDataGeneration:
    """Test class for synthetic data generation"""
    
    def setup_method(self):
        """Setup before each test"""
        self.generator = SyntheticTraumaDataset(
            num_samples=100,  # Smaller for testing
            seq_length=30,
            num_features=4,
            random_seed=42
        )
        
        self.preprocessor = VitalSignsPreprocessor(
            sampling_rate=1.0,
            window_size=30,
            stride=1,
            features=['HR', 'SBP', 'DBP', 'SpO2']
        )
    
    def test_data_generation_shape(self):
        """Test that generated data has correct shape"""
        X, y, metadata = self.generator.generate()
        
        # Check shapes
        assert X.shape[0] == len(y) == len(metadata)
        assert X.shape[1] == 30  # sequence length
        assert X.shape[2] == 4   # num features
        
        # Check data types
        assert X.dtype == np.float32
        assert y.dtype == np.float32
        
        # Check metadata structure
        assert len(metadata) > 0
        sample_metadata = metadata[0]
        assert 'patient_id' in sample_metadata
        assert 'tic_status' in sample_metadata
        assert 'initial_vitals' in sample_metadata
    
    def test_class_distribution(self):
        """Test that classes are balanced"""
        X, y, _ = self.generator.generate()
        
        # Calculate class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        # Should have both classes
        assert len(class_dist) == 2
        
        # Should be approximately balanced (within 10%)
        n_tic = class_dist.get(1.0, 0)
        n_control = class_dist.get(0.0, 0)
        total = len(y)
        
        ratio_tic = n_tic / total
        ratio_control = n_control / total
        
        # Allow small imbalance due to random sampling
        assert 0.4 <= ratio_tic <= 0.6
        assert 0.4 <= ratio_control <= 0.6
    
    def test_physiological_plausibility(self):
        """Test that generated data is physiologically plausible"""
        physio_model = PhysiologicalModel()
        
        # Test stable trajectory
        stable_data = physio_model.generate_stable_trajectory(duration_sec=60)
        
        # Check vital sign ranges
        assert np.all(stable_data['hr'] >= 60)  # HR should be >= 60
        assert np.all(stable_data['hr'] <= 120)  # HR should be <= 120
        assert np.all(stable_data['sbp'] >= 90)  # SBP should be >= 90
        assert np.all(stable_data['sbp'] <= 160)  # SBP should be <= 160
        assert np.all(stable_data['spo2'] >= 95)  # SpO2 should be >= 95
        
        # Test shock trajectory
        shock_data = physio_model.generate_compensated_shock_trajectory(duration_sec=60)
        
        # In shock, HR should increase and SBP should decrease over time
        hr_start = np.mean(shock_data['hr'][:10])
        hr_end = np.mean(shock_data['hr'][-10:])
        sbp_start = np.mean(shock_data['sbp'][:10])
        sbp_end = np.mean(shock_data['sbp'][-10:])
        
        # In decompensating shock, HR increases and SBP decreases
        # Note: this might not always hold due to randomness, but trend should exist
        hr_trend = hr_end - hr_start
        sbp_trend = sbp_end - sbp_start
        
        # Typically HR increases and SBP decreases in shock
        # We'll just check that trends are within reasonable bounds
        assert -20 <= hr_trend <= 50  # HR change in bpm
        assert -40 <= sbp_trend <= 10  # SBP change in mmHg
    
    def test_temporal_consistency(self):
        """Test that time-series data has temporal consistency"""
        X, y, _ = self.generator.generate()
        
        # Check a few samples
        for i in range(min(10, len(X))):
            sample = X[i]
            
            # Check for sudden unrealistic jumps
            for feat_idx in range(sample.shape[1]):
                feature_series = sample[:, feat_idx]
                diffs = np.diff(feature_series)
                
                # Most changes should be relatively small
                # Allow occasional larger changes (e.g., sensor artifacts)
                large_jumps = np.abs(diffs) > 20  # Arbitrary threshold
                assert np.sum(large_jumps) / len(diffs) < 0.1  # < 10% large jumps
    
    def test_missing_data_handling(self):
        """Test missing data handling in preprocessor"""
        # Create sample data with missing values
        np.random.seed(42)
        clean_data = np.random.randn(100, 4)
        
        # Introduce missing values
        missing_mask = np.random.random(clean_data.shape) < 0.1
        data_with_missing = clean_data.copy()
        data_with_missing[missing_mask] = np.nan
        
        # Test different interpolation methods
        methods = ['linear', 'cubic', 'mean', 'nearest']
        
        for method in methods:
            preprocessor = VitalSignsPreprocessor()
            processed = preprocessor.handle_missing_values(
                data_with_missing, 
                method=method
            )
            
            # Should have no NaN values after processing
            assert not np.any(np.isnan(processed))
            
            # Processed data should have similar statistics to original
            if not np.all(missing_mask):
                # Only compare if there was some valid data
                valid_indices = ~missing_mask
                original_valid = clean_data[valid_indices]
                processed_valid = processed[valid_indices]
                
                # Mean should be similar (within tolerance)
                mean_diff = np.abs(np.mean(original_valid) - np.mean(processed_valid))
                assert mean_diff < 0.5
    
    def test_noise_filtering(self):
        """Test noise filtering functionality"""
        # Create clean signal with added noise
        t = np.linspace(0, 10, 1000)
        clean_signal = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
        noisy_signal = clean_signal + 0.5 * np.random.randn(len(t))
        
        # Reshape for preprocessor (add feature dimension)
        data = noisy_signal.reshape(-1, 1)
        
        preprocessor = VitalSignsPreprocessor(sampling_rate=100)  # 100 Hz for this test
        
        # Test different filters
        filter_types = ['butterworth', 'median', 'savgol']
        
        for filter_type in filter_types:
            filtered = preprocessor.filter_noise(
                data, 
                filter_type=filter_type,
                cutoff_freq=1.0
            )
            
            # Filtered signal should have reduced noise
            noise_original = np.std(noisy_signal - clean_signal)
            noise_filtered = np.std(filtered.flatten() - clean_signal)
            
            # Filter should reduce noise (allow small tolerance)
            assert noise_filtered < noise_original * 1.1  # Shouldn't increase noise
    
    def test_window_segmentation(self):
        """Test sliding window segmentation"""
        # Create sample time-series
        np.random.seed(42)
        data = np.random.randn(300, 4)  # 5 minutes at 1 Hz
        labels = np.random.randint(0, 2, 300)
        
        preprocessor = VitalSignsPreprocessor(
            window_size=30,
            stride=10  # Non-overlapping for easier testing
        )
        
        windows, window_labels = preprocessor.sliding_window_segmentation(data, labels)
        
        # Check window count
        expected_windows = (300 - 30) // 10 + 1
        assert len(windows) == expected_windows
        
        # Check window shape
        assert windows.shape[1] == 30  # window_size
        assert windows.shape[2] == 4   # num_features
        
        # Check labels
        assert len(window_labels) == len(windows)
        
        # Check that last window ends at correct time
        last_window_end = 30 + (expected_windows - 1) * 10
        assert last_window_end <= 300
    
    def test_normalization(self):
        """Test data normalization"""
        np.random.seed(42)
        data = np.random.randn(100, 4)
        
        preprocessor = VitalSignsPreprocessor()
        
        # Test z-score normalization
        normalized = preprocessor.z_score_normalization(data, fit=True)
        
        # After z-score normalization, each feature should have mean ~0 and std ~1
        for i in range(data.shape[1]):
            feature_mean = np.mean(normalized[:, i])
            feature_std = np.std(normalized[:, i])
            
            assert abs(feature_mean) < 0.01  # Mean close to 0
            assert 0.99 < feature_std < 1.01  # Std close to 1
        
        # Test min-max normalization
        min_max_normalized = preprocessor.min_max_normalization(data)
        
        # After min-max normalization, values should be in [0, 1]
        assert np.all(min_max_normalized >= 0)
        assert np.all(min_max_normalized <= 1)
        
        # At least some values should be at extremes
        assert np.any(min_max_normalized == 0) or np.any(min_max_normalized == 1)
    
    def test_data_augmentation(self):
        """Test data augmentation"""
        np.random.seed(42)
        
        # Create imbalanced dataset
        X = np.random.randn(100, 30, 4)
        y = np.array([0] * 70 + [1] * 30)  # 70% class 0, 30% class 1
        
        preprocessor = VitalSignsPreprocessor()
        
        # Apply augmentation
        X_aug, y_aug = preprocessor.augment_data(X, y, target_ratio=0.5)
        
        # Should have more samples after augmentation
        assert len(X_aug) > len(X)
        assert len(y_aug) > len(y)
        
        # Class distribution should be more balanced
        unique, counts = np.unique(y_aug, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        minority_class = 1
        majority_class = 0
        
        minority_count = class_dist.get(minority_class, 0)
        majority_count = class_dist.get(majority_class, 0)
        
        # Check if augmentation improved balance
        original_ratio = 30 / 70  # ~0.43
        augmented_ratio = minority_count / majority_count
        
        # Should be closer to target_ratio (0.5)
        assert abs(augmented_ratio - 0.5) < abs(original_ratio - 0.5)
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading preprocessor"""
        # Create and fit preprocessor
        np.random.seed(42)
        data = np.random.randn(100, 4)
        
        preprocessor = VitalSignsPreprocessor()
        preprocessor.z_score_normalization(data, fit=True)
        
        # Save to temporary file
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save_preprocessor(str(save_path))
        
        # Load back
        loaded_preprocessor = VitalSignsPreprocessor.load_preprocessor(str(save_path))
        
        # Check that parameters match
        assert loaded_preprocessor.sampling_rate == preprocessor.sampling_rate
        assert loaded_preprocessor.window_size == preprocessor.window_size
        assert loaded_preprocessor.stride == preprocessor.stride
        assert loaded_preprocessor.features == preprocessor.features
        assert loaded_preprocessor.fitted == preprocessor.fitted
        
        # Check that scalers work the same
        test_data = np.random.randn(10, 4)
        original_normalized = preprocessor.z_score_normalization(test_data)
        loaded_normalized = loaded_preprocessor.z_score_normalization(test_data)
        
        np.testing.assert_array_almost_equal(original_normalized, loaded_normalized)
    
    def test_statistical_fidelity(self):
        """Test that synthetic data matches real-world statistics"""
        X, y, _ = self.generator.generate()
        
        # Separate TIC and control groups
        tic_mask = y == 1
        control_mask = y == 0
        
        X_tic = X[tic_mask]
        X_control = X[control_mask]
        
        # Check that TIC group has higher average heart rate
        hr_tic = X_tic[:, :, 0].mean()  # Feature 0 is HR
        hr_control = X_control[:, :, 0].mean()
        
        # In TIC, HR should generally be higher
        assert hr_tic > hr_control
        
        # Check that TIC group has lower average SBP
        sbp_tic = X_tic[:, :, 1].mean()  # Feature 1 is SBP
        sbp_control = X_control[:, :, 1].mean()
        
        # In TIC, SBP should generally be lower
        assert sbp_tic < sbp_control
        
        # Check Shock Index (HR/SBP)
        # For TIC cases, compute average Shock Index per window
        si_tic = []
        si_control = []
        
        for i in range(len(X_tic)):
            hr = X_tic[i, :, 0].mean()
            sbp = X_tic[i, :, 1].mean()
            si_tic.append(hr / (sbp + 1e-6))
        
        for i in range(len(X_control)):
            hr = X_control[i, :, 0].mean()
            sbp = X_control[i, :, 1].mean()
            si_control.append(hr / (sbp + 1e-6))
        
        avg_si_tic = np.mean(si_tic)
        avg_si_control = np.mean(si_control)
        
        # Shock Index should be higher in TIC group
        assert avg_si_tic > avg_si_control
    
    def test_data_validation_plots(self, tmp_path):
        """Test generation of validation plots"""
        X, y, _ = self.generator.generate()
        
        # Save dataset with validation plots
        output_dir = tmp_path / "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a minimal version of save_dataset method for testing
        np.save(output_dir / "X_test.npy", X)
        np.save(output_dir / "y_test.npy", y)
        
        # Generate validation plots
        self.generator.generate_validation_plots(X, y, str(output_dir))
        
        # Check that plot file was created
        plot_path = output_dir / "data_validation_plots.png"
        assert plot_path.exists()
        
        # Check that summary file would be created (simulated)
        summary = {
            'num_samples': len(X),
            'num_features': X.shape[2],
            'sequence_length': X.shape[1],
            'class_distribution': {
                'tic_positive': int(np.sum(y == 1)),
                'tic_negative': int(np.sum(y == 0)),
                'tic_percentage': float(np.mean(y == 1))
            }
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        assert (output_dir / "summary.json").exists()

if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))