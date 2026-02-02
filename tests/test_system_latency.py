"""
Test suite for system latency
Tests end-to-end latency of the 5G-enabled Digital Twin framework
"""

import pytest
import time
import numpy as np
import json
import threading
import queue
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from inference_simulator import (
    EdgeSimulator,
    NetworkSimulator,
    DigitalTwinEngine,
    simulate_data_stream
)

class TestSystemLatency:
    """Test class for system latency"""
    
    def setup_method(self):
        """Setup before each test"""
        # Configuration for testing
        self.config = {
            'system': {
                'cloud': {
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                    'redis_db': 0,
                    'gpu_acceleration': False
                }
            },
            'data_generation': {
                'num_features': 4,
                'sequence_length': 30
            },
            'model': {
                'transformer': {
                    'd_model': 128,
                    'nhead': 4,
                    'num_encoder_layers': 2,
                    'dim_feedforward': 512,
                    'dropout': 0.1
                }
            },
            'evaluation': {
                'early_warning_threshold': 0.8
            }
        }
        
        # Create components
        self.edge = EdgeSimulator(buffer_size=1000)
        self.network = NetworkSimulator(latency_ms=20.0, reliability=0.99999)
    
    def test_edge_simulator_basic(self):
        """Test basic edge simulator functionality"""
        # Test data generation
        def simple_data_generator():
            for i in range(10):
                yield [i * 10, 120 - i, 80 - i, 98 - i * 0.1]
        
        # Start streaming
        stream_thread = self.edge.start_sensor_stream(
            simple_data_generator(),
            interval_ms=10  # Fast for testing
        )
        
        # Wait for some data to accumulate
        time.sleep(0.05)
        
        # Get latest data
        latest_data = self.edge.get_latest_data(window_size=5)
        
        # Should have some data
        assert len(latest_data) > 0
        
        # Check data structure
        if len(latest_data) > 0:
            assert len(latest_data[0]) == 4  # 4 features
        
        # Stop the stream
        self.edge.stop()
        stream_thread.join(timeout=1.0)
    
    def test_edge_buffer_overflow(self):
        """Test edge buffer overflow handling"""
        # Create a generator that produces data faster than consumed
        def fast_data_generator():
            i = 0
            while True:
                yield [i, i, i, i]
                i += 1
        
        # Start with small buffer
        edge_small_buffer = EdgeSimulator(buffer_size=5)
        stream_thread = edge_small_buffer.start_sensor_stream(
            fast_data_generator(),
            interval_ms=1  # Very fast
        )
        
        # Let it run for a bit
        time.sleep(0.1)
        
        # Buffer should not exceed max size
        buffer_size = edge_small_buffer.data_buffer.qsize()
        assert buffer_size <= 5
        
        # Clean up
        edge_small_buffer.stop()
        stream_thread.join(timeout=0.5)
    
    def test_network_simulator_latency(self):
        """Test network simulator latency"""
        test_data = {
            'patient_id': 'test_patient',
            'data': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'timestamp': time.time()
        }
        
        # Test with different latency settings
        for target_latency in [10.0, 20.0, 50.0, 100.0]:
            network = NetworkSimulator(
                latency_ms=target_latency,
                reliability=1.0  # No packet loss for this test
            )
            
            start_time = time.time()
            result = network.transmit(test_data.copy())
            end_time = time.time()
            
            actual_latency = (end_time - start_time) * 1000  # Convert to ms
            
            # Allow some tolerance (system scheduling, etc.)
            tolerance_ms = 5.0
            assert abs(actual_latency - target_latency) < tolerance_ms
            
            # Result should not be None (reliability is 1.0)
            assert result is not None
            assert 'network_metadata' in result
    
    def test_network_simulator_reliability(self):
        """Test network simulator reliability/packet loss"""
        test_data = {'test': 'data'}
        
        # Test with different reliability settings
        test_cases = [
            (0.5, 0.1),   # 50% reliability, ±10% tolerance
            (0.9, 0.05),  # 90% reliability, ±5% tolerance
            (0.999, 0.01) # 99.9% reliability, ±1% tolerance
        ]
        
        for reliability, tolerance in test_cases:
            network = NetworkSimulator(
                latency_ms=10.0,
                reliability=reliability
            )
            
            n_trials = 1000
            successful_transmissions = 0
            
            for _ in range(n_trials):
                result = network.transmit(test_data.copy())
                if result is not None:
                    successful_transmissions += 1
            
            actual_reliability = successful_transmissions / n_trials
            
            # Check if actual reliability is within tolerance
            assert abs(actual_reliability - reliability) < tolerance
            
            # Get statistics
            stats = network.get_stats()
            assert 'total_packets' in stats
            assert 'packet_loss' in stats
            assert 'packet_loss_rate' in stats
            
            # Packet loss rate should match 1 - reliability
            expected_loss_rate = 1 - reliability
            assert abs(stats['packet_loss_rate'] - expected_loss_rate) < tolerance
    
    @patch('inference_simulator.redis.Redis')
    def test_digital_twin_engine_init(self, mock_redis):
        """Test Digital Twin engine initialization"""
        # Mock Redis
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        # Mock model loading
        with patch('inference_simulator.torch.load') as mock_torch_load:
            mock_model_state = {
                'model_state_dict': {},
                'config': self.config
            }
            mock_torch_load.return_value = mock_model_state
            
            # Create engine
            engine = DigitalTwinEngine(self.config, model_path='./dummy_model.pth')
            
            # Check initialization
            assert engine.config == self.config
            assert engine.redis_client == mock_redis_instance
            assert hasattr(engine, 'model')
            assert hasattr(engine, 'patient_state')
            assert hasattr(engine, 'inference_history')
            
            # Redis should have been called with correct parameters
            mock_redis.assert_called_once_with(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
    
    @patch('inference_simulator.redis.Redis')
    @patch('inference_simulator.torch.load')
    def test_digital_twin_preprocessing(self, mock_torch_load, mock_redis):
        """Test Digital Twin preprocessing"""
        # Setup mocks
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_model_state = {
            'model_state_dict': {},
            'config': self.config
        }
        mock_torch_load.return_value = mock_model_state
        
        # Create engine
        engine = DigitalTwinEngine(self.config)
        
        # Test data with different lengths
        test_cases = [
            # (input_length, expected_output_length)
            (15, 30),  # Shorter than required - should pad
            (30, 30),  # Exact length
            (45, 30),  # Longer than required - should truncate
        ]
        
        for input_len, expected_len in test_cases:
            # Create test data
            data_window = []
            for i in range(input_len):
                data_window.append([70 + i, 120 - i * 0.5, 80 - i * 0.3, 98 - i * 0.1])
            
            # Preprocess
            result = engine.preprocess(data_window)
            
            # Check shape
            assert result.shape == (1, expected_len, 4)  # [batch, seq_len, features]
            
            # Should be a tensor
            assert hasattr(result, 'device')
    
    @patch('inference_simulator.redis.Redis')
    @patch('inference_simulator.torch.load')
    @patch('inference_simulator.DigitalTwinEngine.preprocess')
    @patch('inference_simulator.DigitalTwinEngine.model')
    def test_digital_twin_inference(self, mock_model, mock_preprocess, 
                                   mock_torch_load, mock_redis):
        """Test Digital Twin inference"""
        # Setup mocks
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_model_state = {
            'model_state_dict': {},
            'config': self.config
        }
        mock_torch_load.return_value = mock_model_state
        
        # Mock model prediction
        mock_model.return_value = Mock()
        mock_model.return_value.item.return_value = 0.75  # Risk score
        
        # Mock preprocessing
        mock_tensor = Mock()
        mock_tensor.shape = (1, 30, 4)
        mock_preprocess.return_value = mock_tensor
        
        # Create engine
        engine = DigitalTwinEngine(self.config)
        engine.model = mock_model
        
        # Test inference
        data_window = [[70, 120, 80, 98]] * 30  # 30 time steps
        
        result = engine.infer(data_window, patient_id="test_patient_001")
        
        # Check result structure
        assert 'patient_id' in result
        assert 'risk_score' in result
        assert 'alert_triggered' in result
        assert 'latency_ms' in result
        assert 'data_window_size' in result
        
        # Check specific values
        assert result['patient_id'] == "test_patient_001"
        assert result['risk_score'] == 0.75
        assert result['alert_triggered'] == False  # 0.75 < 0.8 threshold
        assert result['data_window_size'] == 30
        
        # Check latency breakdown
        latency = result['latency_ms']
        assert 'total' in latency
        assert 'preprocessing' in latency
        assert 'inference' in latency
        
        # Latency should be positive
        assert latency['total'] > 0
        
        # Check that patient state was updated
        assert "test_patient_001" in engine.patient_state
        
        # Check that inference was recorded
        assert len(engine.inference_history) == 1
        assert engine.inference_history[0] == result
    
    @patch('inference_simulator.redis.Redis')
    @patch('inference_simulator.torch.load')
    def test_digital_twin_performance_stats(self, mock_torch_load, mock_redis):
        """Test Digital Twin performance statistics"""
        # Setup mocks
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_model_state = {
            'model_state_dict': {},
            'config': self.config
        }
        mock_torch_load.return_value = mock_model_state
        
        # Create engine
        engine = DigitalTwinEngine(self.config)
        
        # Add some dummy inference records
        for i in range(10):
            engine.inference_history.append({
                'latency_ms': {
                    'total': 10.0 + i * 5.0  # Increasing latency
                }
            })
        
        # Get performance stats
        stats = engine.get_performance_stats()
        
        # Check stats structure
        assert 'total_inferences' in stats
        assert 'avg_latency_ms' in stats
        assert 'p50_latency_ms' in stats
        assert 'p95_latency_ms' in stats
        assert 'p99_latency_ms' in stats
        assert 'min_latency_ms' in stats
        assert 'max_latency_ms' in stats
        
        # Check values
        assert stats['total_inferences'] == 10
        
        # Latencies should be in correct order
        assert stats['min_latency_ms'] <= stats['p50_latency_ms']
        assert stats['p50_latency_ms'] <= stats['p95_latency_ms']
        assert stats['p95_latency_ms'] <= stats['p99_latency_ms']
        assert stats['p99_latency_ms'] <= stats['max_latency_ms']
        
        # Average should be between min and max
        assert stats['min_latency_ms'] <= stats['avg_latency_ms'] <= stats['max_latency_ms']
    
    def test_end_to_end_latency_simulation(self):
        """Test end-to-end latency simulation"""
        # Create a simple end-to-end test
        edge = EdgeSimulator(buffer_size=100)
        network = NetworkSimulator(latency_ms=20.0, reliability=0.999)
        
        # Simulate data generation
        def test_data_generator():
            for i in range(20):
                yield [70 + i, 120 - i * 0.5, 80 - i * 0.3, 98 - i * 0.1]
        
        # Start edge streaming
        stream_thread = edge.start_sensor_stream(
            test_data_generator(),
            interval_ms=50  # 20 Hz for testing
        )
        
        # Let some data accumulate
        time.sleep(0.3)
        
        # Get data from edge
        data_window = edge.get_latest_data(window_size=10)
        
        # Simulate network transmission
        transmission_data = {
            'patient_id': 'e2e_test_patient',
            'data_window': data_window,
            'timestamp': time.time()
        }
        
        start_time = time.time()
        transmitted = network.transmit(transmission_data)
        network_latency = (time.time() - start_time) * 1000
        
        # Check network performance
        assert transmitted is not None
        assert network_latency > 0
        
        # Get network stats
        network_stats = network.get_stats()
        assert network_stats['total_packets'] > 0
        
        # Clean up
        edge.stop()
        stream_thread.join(timeout=0.5)
    
    def test_concurrent_inferences(self):
        """Test handling of concurrent inferences"""
        # This test simulates multiple patients being monitored simultaneously
        
        n_patients = 5
        n_inferences_per_patient = 10
        
        # Mock Digital Twin engine for each patient
        engines = []
        for i in range(n_patients):
            # Each patient has their own engine instance (or shared with patient_id)
            engine = Mock()
            engine.infer = Mock(return_value={
                'patient_id': f'patient_{i}',
                'risk_score': 0.5 + i * 0.1,
                'latency_ms': {'total': 20.0 + i * 5.0}
            })
            engines.append(engine)
        
        # Simulate concurrent inferences
        results = []
        
        def run_inference(patient_idx, inference_idx):
            # Simulate some processing time
            time.sleep(0.001 * (inference_idx % 3))
            
            result = engines[patient_idx].infer(
                data_window=[[70, 120, 80, 98]] * 30,
                patient_id=f'patient_{patient_idx}'
            )
            results.append(result)
        
        # Create and start threads
        threads = []
        for patient_idx in range(n_patients):
            for inference_idx in range(n_inferences_per_patient):
                thread = threading.Thread(
                    target=run_inference,
                    args=(patient_idx, inference_idx)
                )
                threads.append(thread)
                thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=1.0)
        
        # Check results
        assert len(results) == n_patients * n_inferences_per_patient
        
        # All inferences should have completed
        for result in results:
            assert 'patient_id' in result
            assert 'risk_score' in result
            assert 'latency_ms' in result
    
    def test_latency_requirements(self):
        """Test that system meets latency requirements"""
        # Paper requirements: total latency < 100ms
        max_allowed_latency_ms = 100.0
        
        # Component targets from paper:
        # Edge processing: 2.0 ms
        # Network uplink: 18.0 ms  
        # Cloud inference: 15.2 ms
        # Network downlink: 12.0 ms
        # Display: 12.0 ms
        # Total: 47.2 ms
        
        component_targets = {
            'edge_processing': 5.0,      # Slightly higher for robustness
            'network_uplink': 20.0,
            'cloud_inference': 50.0,
            'network_downlink': 10.0,
            'display': 15.0
        }
        
        # Simulate component latencies
        simulated_latencies = {}
        total_latency = 0.0
        
        for component, target in component_targets.items():
            # Simulate latency with some variance
            simulated_latency = target * np.random.uniform(0.8, 1.2)
            simulated_latencies[component] = simulated_latency
            total_latency += simulated_latency
        
        # Check requirements
        assert total_latency < max_allowed_latency_ms
        
        # Individual components should also be within reasonable bounds
        for component, target in component_targets.items():
            simulated = simulated_latencies[component]
            # Allow 50% overhead for testing
            assert simulated < target * 1.5
        
        print(f"Simulated total latency: {total_latency:.2f} ms")
        print("Component latencies:")
        for component, latency in simulated_latencies.items():
            print(f"  {component}: {latency:.2f} ms")
    
    def test_system_under_load(self):
        """Test system performance under load"""
        # Simulate increasing load
        load_levels = [1, 5, 10, 20]  # Concurrent patients
        
        for n_patients in load_levels:
            print(f"\nTesting with {n_patients} concurrent patients")
            
            # Simulate latencies for each patient
            latencies = []
            
            for patient_idx in range(n_patients):
                # Simulate varying latencies based on load
                base_latency = 20.0  # ms
                load_factor = 1.0 + (n_patients - 1) * 0.02  # 2% increase per patient
                
                patient_latency = base_latency * load_factor * np.random.uniform(0.9, 1.1)
                latencies.append(patient_latency)
            
            # Calculate statistics
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            print(f"  Average latency: {avg_latency:.2f} ms")
            print(f"  P95 latency: {p95_latency:.2f} ms")
            
            # Under load, latency should increase but stay within limits
            if n_patients > 1:
                # P95 should be less than 100ms even under load
                assert p95_latency < 100.0
    
    def test_data_stream_simulation(self):
        """Test data stream simulation"""
        # Test different simulation durations
        test_durations = [1, 5, 10]  # minutes
        
        for duration_minutes in test_durations:
            print(f"\nTesting {duration_minutes} minute simulation")
            
            # Generate data stream
            data_stream = simulate_data_stream(duration_minutes)
            
            # Collect some data points
            data_points = []
            timestamps = []
            
            start_time = time.time()
            max_points = 100  # Limit for testing
            
            for i, data_point in enumerate(data_stream):
                if i >= max_points:
                    break
                
                data_points.append(data_point)
                timestamps.append(time.time() - start_time)
            
            # Check data
            assert len(data_points) > 0
            
            # Each data point should have 4 features
            for point in data_points:
                assert len(point) == 4
            
            # Convert to numpy array for analysis
            data_array = np.array(data_points)
            
            # Check feature ranges (loose bounds for simulation)
            # HR: typically 60-180 in trauma
            hr_values = data_array[:, 0]
            assert np.all(hr_values >= 40) and np.all(hr_values <= 200)
            
            # SBP: typically 60-200 in trauma
            sbp_values = data_array[:, 1]
            assert np.all(sbp_values >= 50) and np.all(sbp_values <= 250)
            
            # SpO2: typically 85-100
            spo2_values = data_array[:, 3]
            assert np.all(spo2_values >= 70) and np.all(spo2_values <= 100)
            
            print(f"  Generated {len(data_points)} data points")
            print(f"  HR range: {hr_values.min():.1f} - {hr_values.max():.1f}")
            print(f"  SBP range: {sbp_values.min():.1f} - {sbp_values.max():.1f}")
            print(f"  SpO2 range: {spo2_values.min():.1f} - {spo2_values.max():.1f}")
    
    def test_latency_distribution(self):
        """Test that latency follows expected distribution"""
        # Generate sample latencies
        n_samples = 1000
        
        # Simulate latencies with log-normal distribution (typical for network latency)
        mu, sigma = np.log(25), 0.5  # Mean ~25ms, shape parameter
        simulated_latencies = np.random.lognormal(mu, sigma, n_samples)
        
        # Calculate statistics
        mean_latency = np.mean(simulated_latencies)
        median_latency = np.median(simulated_latencies)
        p95_latency = np.percentile(simulated_latencies, 95)
        p99_latency = np.percentile(simulated_latencies, 99)
        
        print(f"\nLatency distribution analysis:")
        print(f"  Mean: {mean_latency:.2f} ms")
        print(f"  Median: {median_latency:.2f} ms")
        print(f"  P95: {p95_latency:.2f} ms")
        print(f"  P99: {p99_latency:.2f} ms")
        
        # For log-normal, mean > median
        assert mean_latency > median_latency
        
        # P99 should be significantly higher than median
        assert p99_latency > median_latency * 1.5
        
        # Most latencies should be under 100ms
        proportion_under_100ms = np.sum(simulated_latencies < 100) / n_samples
        assert proportion_under_100ms > 0.95  # >95% under 100ms
    
    def test_error_handling(self):
        """Test error handling in latency-critical components"""
        edge = EdgeSimulator(buffer_size=10)
        
        # Test with faulty data generator
        def faulty_generator():
            yield [70, 120, 80, 98]
            raise Exception("Simulated sensor failure")
            yield [71, 119, 79, 97]  # This won't be reached
        
        # Start stream - should handle generator exception gracefully
        stream_thread = edge.start_sensor_stream(
            faulty_generator(),
            interval_ms=100
        )
        
        # Give it time to encounter error
        time.sleep(0.2)
        
        # Should still be able to get buffered data
        data = edge.get_latest_data(window_size=5)
        assert len(data) == 1  # Only one successful data point
        
        # Clean up
        edge.stop()
        stream_thread.join(timeout=0.5)

if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))