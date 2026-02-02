"""
Test suite for model inference
Tests the Trauma-Former model and inference pipeline
"""

import pytest
import torch
import numpy as np
import json
import time
import tempfile
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from models.trauma_former import TraumaFormer, TraumaFormerWithUncertainty
from models.lstm_baseline import LSTMBaseline, BidirectionalLSTMWithAttention
from models.shock_index import ShockIndexCalculator, DynamicShockIndex
from data.synthetic_data_generator import SyntheticTraumaDataset
from data.preprocessor import VitalSignsPreprocessor

class TestModelInference:
    """Test class for model inference"""
    
    def setup_method(self):
        """Setup before each test"""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test data
        self.batch_size = 8
        self.seq_length = 30
        self.num_features = 4
        
        # Generate random test data
        self.test_data = torch.randn(
            self.batch_size, 
            self.seq_length, 
            self.num_features
        )
        
        # Create models
        self.trauma_former = TraumaFormer(
            input_dim=self.num_features,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.1
        )
        
        self.lstm_baseline = LSTMBaseline(
            input_dim=self.num_features,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        self.shock_index_calc = ShockIndexCalculator(threshold=1.0)
        
        # Create sample vital signs for rule-based tests
        self.sample_hr = np.array([80, 85, 90, 95, 100, 105, 110, 115])
        self.sample_sbp = np.array([120, 115, 110, 105, 100, 95, 90, 85])
    
    def test_model_initialization(self):
        """Test that models initialize correctly"""
        # Check Trauma-Former
        assert isinstance(self.trauma_former, TraumaFormer)
        assert hasattr(self.trauma_former, 'transformer_encoder')
        assert hasattr(self.trauma_former, 'classifier')
        
        # Check LSTM
        assert isinstance(self.lstm_baseline, LSTMBaseline)
        assert hasattr(self.lstm_baseline, 'lstm')
        assert hasattr(self.lstm_baseline, 'classifier')
        
        # Check parameter counts
        trauma_former_params = sum(p.numel() for p in self.trauma_former.parameters())
        lstm_params = sum(p.numel() for p in self.lstm_baseline.parameters())
        
        print(f"Trauma-Former parameters: {trauma_former_params:,}")
        print(f"LSTM parameters: {lstm_params:,}")
        
        # Trauma-Former should have more parameters than LSTM
        assert trauma_former_params > lstm_params
    
    def test_forward_pass(self):
        """Test forward pass through models"""
        # Test Trauma-Former
        trauma_former_output = self.trauma_former(self.test_data)
        assert trauma_former_output.shape == (self.batch_size,)
        assert torch.all(trauma_former_output >= 0) and torch.all(trauma_former_output <= 1)
        
        # Test LSTM
        lstm_output = self.lstm_baseline(self.test_data)
        assert lstm_output.shape == (self.batch_size,)
        assert torch.all(lstm_output >= 0) and torch.all(lstm_output <= 1)
        
        # Test with attention (Trauma-Former)
        trauma_former_output_with_attn, attention_weights = self.trauma_former(
            self.test_data, 
            return_attention=True
        )
        
        assert trauma_former_output_with_attn.shape == (self.batch_size,)
        assert len(attention_weights) == 2  # 2 transformer layers
        assert attention_weights[0].shape == (self.batch_size, 4, 30, 30)  # [batch, heads, seq, seq]
    
    def test_batch_size_independence(self):
        """Test that models work with different batch sizes"""
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            test_data = torch.randn(batch_size, self.seq_length, self.num_features)
            
            # Trauma-Former
            output_tf = self.trauma_former(test_data)
            assert output_tf.shape == (batch_size,)
            
            # LSTM
            output_lstm = self.lstm_baseline(test_data)
            assert output_lstm.shape == (batch_size,)
    
    def test_sequence_length_independence(self):
        """Test that models work with different sequence lengths"""
        seq_lengths = [15, 30, 45, 60]
        
        for seq_len in seq_lengths:
            test_data = torch.randn(self.batch_size, seq_len, self.num_features)
            
            # Trauma-Former should handle different lengths
            output_tf = self.trauma_former(test_data)
            assert output_tf.shape == (self.batch_size,)
            
            # LSTM should also handle different lengths
            output_lstm = self.lstm_baseline(test_data)
            assert output_lstm.shape == (self.batch_size,)
    
    def test_shock_index_calculation(self):
        """Test Shock Index calculations"""
        # Test basic calculation
        si = self.shock_index_calc.calculate_si(self.sample_hr, self.sample_sbp)
        assert si.shape == self.sample_hr.shape
        
        # Check that calculation is correct
        expected_si = self.sample_hr / self.sample_sbp
        np.testing.assert_array_almost_equal(si, expected_si)
        
        # Test predictions
        predictions = self.shock_index_calc.predict_tic(self.sample_hr, self.sample_sbp)
        assert predictions.shape == self.sample_hr.shape
        assert predictions.dtype == bool
        
        # Test probability calculation
        probabilities = self.shock_index_calc.predict_tic_probability(
            self.sample_hr, 
            self.sample_sbp
        )
        assert probabilities.shape == self.sample_hr.shape
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    def test_dynamic_shock_index(self):
        """Test dynamic Shock Index with rate of change"""
        dynamic_si = DynamicShockIndex(threshold=1.0, alpha=0.5)
        
        analysis = dynamic_si.analyze_dynamic_trajectory(
            self.sample_hr,
            self.sample_sbp
        )
        
        assert 'standard_shock_index' in analysis
        assert 'dynamic_shock_index' in analysis
        assert 'standard_predictions' in analysis
        assert 'dynamic_predictions' in analysis
        
        # Dynamic SI should have same length as input
        assert len(analysis['dynamic_shock_index']) == len(self.sample_hr)
        
        # With positive alpha, dynamic SI should emphasize trends
        # For increasing HR and decreasing SBP, dynamic SI should be higher than standard
        hr_trend = self.sample_hr[-1] - self.sample_hr[0]
        sbp_trend = self.sample_sbp[-1] - self.sample_sbp[0]
        
        if hr_trend > 0 and sbp_trend < 0:  # Typical shock pattern
            # Dynamic SI at the end should be higher than standard SI
            assert analysis['dynamic_shock_index'][-1] > analysis['standard_shock_index'][-1]
    
    def test_model_computation_metrics(self):
        """Test computation of FLOPs and model size"""
        # Test Trauma-Former FLOPs calculation
        flops_info = self.trauma_former.compute_flops(seq_len=30)
        
        assert 'total_flops' in flops_info
        assert 'total_mflops' in flops_info
        assert 'expected_inference_latency_ms' in flops_info
        
        # FLOPs should be positive
        assert flops_info['total_flops'] > 0
        assert flops_info['total_mflops'] > 0
        
        # Test model size calculation
        size_info = self.trauma_former.get_model_size()
        
        assert 'total_parameters' in size_info
        assert 'parameter_size_mb' in size_info
        
        # Parameters should be positive
        assert size_info['total_parameters'] > 0
        assert size_info['parameter_size_mb'] > 0
        
        # Check against paper specifications (~1.2M parameters)
        # Allow some variance
        assert 1_000_000 <= size_info['total_parameters'] <= 1_500_000
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation with Monte Carlo dropout"""
        # Create model with uncertainty estimation
        tf_uncertainty = TraumaFormerWithUncertainty(
            input_dim=self.num_features,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.1
        )
        
        # Test uncertainty prediction
        uncertainty_results = tf_uncertainty.predict_with_uncertainty(
            self.test_data,
            n_samples=50
        )
        
        assert 'mean' in uncertainty_results
        assert 'std' in uncertainty_results
        assert 'ci_lower' in uncertainty_results
        assert 'ci_upper' in uncertainty_results
        assert 'all_samples' in uncertainty_results
        
        # Check shapes
        assert uncertainty_results['mean'].shape == (self.batch_size,)
        assert uncertainty_results['std'].shape == (self.batch_size,)
        assert uncertainty_results['ci_lower'].shape == (self.batch_size,)
        assert uncertainty_results['ci_upper'].shape == (self.batch_size,)
        assert uncertainty_results['all_samples'].shape == (50, self.batch_size)
        
        # Confidence intervals should contain the mean
        for i in range(self.batch_size):
            assert uncertainty_results['ci_lower'][i] <= uncertainty_results['mean'][i]
            assert uncertainty_results['ci_upper'][i] >= uncertainty_results['mean'][i]
        
        # Standard deviation should be positive
        assert np.all(uncertainty_results['std'] >= 0)
    
    def test_attention_mechanism(self):
        """Test attention mechanism functionality"""
        # Get attention weights
        _, attention_weights = self.trauma_former(
            self.test_data, 
            return_attention=True
        )
        
        # Should have attention weights for each layer
        assert len(attention_weights) == 2  # 2 transformer layers
        
        # Check attention weight properties
        for layer_idx, attn_weights in enumerate(attention_weights):
            # Shape: [batch, heads, seq_len, seq_len]
            assert attn_weights.shape == (self.batch_size, 4, 30, 30)
            
            # Attention weights should sum to 1 along the last dimension
            for b in range(self.batch_size):
                for h in range(4):
                    row_sums = attn_weights[b, h].sum(dim=-1)
                    # Allow small numerical errors
                    assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5)
            
            # Self-attention diagonal may have higher values
            # but not required to be highest
        
        # Test attention map retrieval
        attention_maps = self.trauma_former.get_attention_maps(layer_idx=-1)
        assert attention_maps is not None
        assert attention_maps.shape == (self.batch_size, 4, 30, 30)
    
    def test_model_serialization(self, tmp_path):
        """Test model saving and loading"""
        # Create a temporary file
        model_path = tmp_path / "test_model.pth"
        
        # Save model
        torch.save({
            'model_state_dict': self.trauma_former.state_dict(),
            'config': self.trauma_former.config
        }, model_path)
        
        # Load model
        checkpoint = torch.load(model_path)
        loaded_model = TraumaFormer.from_config(checkpoint['config'])
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test that loaded model produces same output
        with torch.no_grad():
            original_output = self.trauma_former(self.test_data)
            loaded_output = loaded_model(self.test_data)
            
            # Should be very close (allowing for small numerical differences)
            assert torch.allclose(original_output, loaded_output, rtol=1e-5)
    
    def test_inference_latency(self):
        """Test inference latency"""
        # Warm-up
        with torch.no_grad():
            _ = self.trauma_former(self.test_data)
        
        # Measure latency
        n_iterations = 100
        latencies = []
        
        for _ in range(n_iterations):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.trauma_former(self.test_data)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"Mean inference latency: {mean_latency:.2f} ms")
        print(f"P95 inference latency: {p95_latency:.2f} ms")
        
        # Check against paper specifications (15.2 ms on A100)
        # This test runs on CPU, so we just check that it's reasonable
        assert mean_latency < 100  # Should be under 100ms even on CPU
        
        # Test batch size effect on latency
        small_batch = torch.randn(1, self.seq_length, self.num_features)
        
        small_start = time.perf_counter()
        with torch.no_grad():
            _ = self.trauma_former(small_batch)
        small_latency = (time.perf_counter() - small_start) * 1000
        
        print(f"Single sample latency: {small_latency:.2f} ms")
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        # Create a simple loss function
        criterion = torch.nn.BCELoss()
        
        # Forward pass with gradient computation
        predictions = self.trauma_former(self.test_data)
        
        # Create dummy targets
        targets = torch.rand_like(predictions)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are not all zero
        has_non_zero_grad = False
        for name, param in self.trauma_former.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.any(param.grad != 0):
                    has_non_zero_grad = True
                    break
        
        assert has_non_zero_grad, "No non-zero gradients found"
    
    def test_model_on_synthetic_data(self):
        """Test model on synthetic dataset"""
        # Generate small synthetic dataset
        generator = SyntheticTraumaDataset(
            num_samples=50,
            seq_length=30,
            num_features=4,
            random_seed=42
        )
        
        X, y, _ = generator.generate()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Test Trauma-Former
        with torch.no_grad():
            predictions_tf = self.trauma_former(X_tensor)
        
        assert predictions_tf.shape == (50,)
        assert torch.all(predictions_tf >= 0) and torch.all(predictions_tf <= 1)
        
        # Test LSTM
        with torch.no_grad():
            predictions_lstm = self.lstm_baseline(X_tensor)
        
        assert predictions_lstm.shape == (50,)
        assert torch.all(predictions_lstm >= 0) and torch.all(predictions_lstm <= 1)
        
        # Basic performance check (should do better than random)
        # Convert predictions to binary
        binary_predictions_tf = (predictions_tf > 0.5).float()
        accuracy_tf = (binary_predictions_tf == y_tensor).float().mean()
        
        # Should be better than 50% (random guessing)
        assert accuracy_tf > 0.5
    
    def test_attention_bilstm(self):
        """Test bidirectional LSTM with attention"""
        attention_lstm = BidirectionalLSTMWithAttention(
            input_dim=self.num_features,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        # Test with attention
        predictions, attention_weights = attention_lstm(
            self.test_data, 
            return_attention=True
        )
        
        assert predictions.shape == (self.batch_size,)
        assert attention_weights.shape == (self.batch_size, self.seq_length)
        
        # Attention weights should sum to 1 for each sample
        for i in range(self.batch_size):
            assert torch.allclose(
                attention_weights[i].sum(), 
                torch.tensor(1.0), 
                rtol=1e-5
            )
        
        # Test without attention
        predictions_no_attn = attention_lstm(self.test_data, return_attention=False)
        assert predictions_no_attn.shape == (self.batch_size,)
    
    def test_edge_cases(self):
        """Test model with edge cases"""
        # Test with all zeros
        zeros = torch.zeros(self.batch_size, self.seq_length, self.num_features)
        with torch.no_grad():
            output_zeros = self.trauma_former(zeros)
        
        assert output_zeros.shape == (self.batch_size,)
        
        # Test with very large values
        large_values = torch.randn(self.batch_size, self.seq_length, self.num_features) * 100
        with torch.no_grad():
            output_large = self.trauma_former(large_values)
        
        assert output_large.shape == (self.batch_size,)
        assert torch.all(output_large >= 0) and torch.all(output_large <= 1)
        
        # Test with NaN values (should handle gracefully or fail predictably)
        nan_tensor = torch.randn(self.batch_size, self.seq_length, self.num_features)
        nan_tensor[0, 0, 0] = float('nan')
        
        # Model should either handle NaN or raise an error
        try:
            with torch.no_grad():
                output_nan = self.trauma_former(nan_tensor)
            # If it doesn't crash, output should not contain NaN
            assert not torch.any(torch.isnan(output_nan))
        except Exception as e:
            # It's acceptable for model to fail with NaN input
            print(f"Model raised expected exception with NaN input: {e}")
    
    def test_model_config_consistency(self):
        """Test that model configuration is consistent"""
        # Create model from config dictionary
        config = {
            'input_dim': 4,
            'd_model': 128,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 512,
            'dropout': 0.1
        }
        
        model_from_config = TraumaFormer.from_config(config)
        
        # Should have same architecture
        assert model_from_config.config == config
        
        # Should produce output of correct shape
        with torch.no_grad():
            output = model_from_config(self.test_data)
        assert output.shape == (self.batch_size,)

if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))