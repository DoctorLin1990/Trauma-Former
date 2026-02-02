#!/usr/bin/env python3
"""
Real-time Inference Simulator for Trauma-Former
Simulates the 5G-enabled Digital Twin framework described in Section 2
"""

import argparse
import yaml
import torch
import numpy as np
import time
import json
import redis
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import threading
import queue

from models.trauma_former import TraumaFormer
from utils.system_monitor import SystemMonitor

class EdgeSimulator:
    """Simulates edge device in ambulance"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.data_buffer = queue.Queue(maxsize=buffer_size)
        self.running = False
        
    def start_sensor_stream(self, data_generator, interval_ms: int = 1000):
        """Simulate sensor data streaming at 1 Hz"""
        self.running = True
        def stream():
            while self.running:
                data = next(data_generator)
                if not self.data_buffer.full():
                    self.data_buffer.put({
                        'timestamp': datetime.now().isoformat(),
                        'data': data,
                        'sensor_id': 'vital_signs_monitor'
                    })
                time.sleep(interval_ms / 1000.0)
        
        thread = threading.Thread(target=stream)
        thread.daemon = True
        thread.start()
        return thread
    
    def get_latest_data(self, window_size: int = 30):
        """Get latest window of data for inference"""
        window_data = []
        buffer_list = list(self.data_buffer.queue)
        
        if len(buffer_list) >= window_size:
            window_data = buffer_list[-window_size:]
        else:
            window_data = buffer_list
        
        return [item['data'] for item in window_data]
    
    def stop(self):
        self.running = False

class NetworkSimulator:
    """Simulates 5G URLLC network with configurable latency"""
    
    def __init__(self, latency_ms: float = 20.0, reliability: float = 0.99999):
        self.latency_ms = latency_ms
        self.reliability = reliability
        self.packet_loss_counter = 0
        self.total_packets = 0
        
    def transmit(self, data: Dict) -> Optional[Dict]:
        """Simulate network transmission with latency and reliability"""
        self.total_packets += 1
        
        # Simulate packet loss
        if np.random.random() > self.reliability:
            self.packet_loss_counter += 1
            return None
        
        # Simulate latency
        time.sleep(self.latency_ms / 1000.0)
        
        # Add transmission metadata
        data['network_metadata'] = {
            'transmission_time': datetime.now().isoformat(),
            'latency_ms': self.latency_ms,
            'packet_id': self.total_packets
        }
        
        return data
    
    def get_stats(self) -> Dict:
        """Get network statistics"""
        loss_rate = self.packet_loss_counter / max(self.total_packets, 1)
        return {
            'total_packets': self.total_packets,
            'packet_loss': self.packet_loss_counter,
            'packet_loss_rate': loss_rate,
            'latency_ms': self.latency_ms,
            'reliability': self.reliability
        }

class DigitalTwinEngine:
    """Core Digital Twin inference engine"""
    
    def __init__(self, config: Dict, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Initialize Redis for state persistence
        self.redis_client = redis.Redis(
            host=config['system']['cloud']['redis_host'],
            port=config['system']['cloud']['redis_port'],
            db=config['system']['cloud']['redis_db'],
            decode_responses=True
        )
        
        # State tracking
        self.patient_state = {}
        self.inference_history = []
        
        # Performance monitoring
        self.monitor = SystemMonitor()
        
    def load_model(self, model_path: Optional[str] = None):
        """Load Trauma-Former model"""
        model_config = self.config['model']['transformer']
        model = TraumaFormer(
            input_dim=self.config['data_generation']['num_features'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_encoder_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout']
        )
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        return model
    
    def preprocess(self, data_window: List) -> torch.Tensor:
        """Preprocess data window for inference"""
        # Convert to numpy array
        data_array = np.array(data_window, dtype=np.float32)
        
        # Check window size
        seq_length = self.config['data_generation']['sequence_length']
        if data_array.shape[0] < seq_length:
            # Pad if necessary
            padding = np.zeros((seq_length - data_array.shape[0], data_array.shape[1]))
            data_array = np.vstack([padding, data_array])
        elif data_array.shape[0] > seq_length:
            # Truncate if necessary
            data_array = data_array[-seq_length:]
        
        # Normalize (simplified - should use trained normalization)
        # In practice, this should use the same normalization as training
        data_array = (data_array - data_array.mean(axis=0)) / (data_array.std(axis=0) + 1e-8)
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(data_array).unsqueeze(0)  # Add batch dimension
        return tensor_data.to(self.device)
    
    def infer(self, data_window: List, patient_id: str = "patient_001") -> Dict:
        """Perform real-time inference"""
        start_time = time.time()
        
        try:
            # Preprocess
            preprocess_start = time.time()
            input_tensor = self.preprocess(data_window)
            preprocess_time = (time.time() - preprocess_start) * 1000
            
            # Inference
            inference_start = time.time()
            with torch.no_grad():
                prediction = self.model(input_tensor)
                risk_score = prediction.item()
            inference_time = (time.time() - inference_start) * 1000
            
            # Post-process
            risk_threshold = self.config['evaluation']['early_warning_threshold']
            alert_triggered = risk_score > risk_threshold
            
            # Update patient state
            self.update_patient_state(patient_id, risk_score, alert_triggered)
            
            # Calculate total latency
            total_time = (time.time() - start_time) * 1000
            
            # Record inference
            inference_record = {
                'patient_id': patient_id,
                'timestamp': datetime.now().isoformat(),
                'risk_score': risk_score,
                'alert_triggered': alert_triggered,
                'latency_ms': {
                    'total': total_time,
                    'preprocessing': preprocess_time,
                    'inference': inference_time
                },
                'data_window_size': len(data_window)
            }
            
            self.inference_history.append(inference_record)
            
            # Monitor system performance
            self.monitor.record_inference(total_time)
            
            return inference_record
            
        except Exception as e:
            error_record = {
                'patient_id': patient_id,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }
            return error_record
    
    def update_patient_state(self, patient_id: str, risk_score: float, alert: bool):
        """Update patient state in Redis"""
        state_key = f"patient:{patient_id}:state"
        history_key = f"patient:{patient_id}:history"
        
        current_state = {
            'patient_id': patient_id,
            'last_update': datetime.now().isoformat(),
            'current_risk': risk_score,
            'alert_active': alert,
            'alert_count': int(alert)
        }
        
        # Get previous state
        prev_state = self.redis_client.get(state_key)
        if prev_state:
            prev_state = json.loads(prev_state)
            if alert and not prev_state.get('alert_active', False):
                current_state['alert_count'] = prev_state.get('alert_count', 0) + 1
        
        # Update state
        self.redis_client.set(state_key, json.dumps(current_state))
        
        # Add to history (keep last 1000 records)
        history_item = {
            'timestamp': datetime.now().isoformat(),
            'risk_score': risk_score,
            'alert': alert
        }
        self.redis_client.lpush(history_key, json.dumps(history_item))
        self.redis_client.ltrim(history_key, 0, 999)
        
        self.patient_state[patient_id] = current_state
    
    def get_performance_stats(self) -> Dict:
        """Get system performance statistics"""
        latencies = [record['latency_ms']['total'] for record in self.inference_history 
                    if 'latency_ms' in record]
        
        if not latencies:
            return {}
        
        return {
            'total_inferences': len(self.inference_history),
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'system_utilization': self.monitor.get_utilization_stats()
        }
    
    def save_results(self, output_dir: str):
        """Save inference results and performance statistics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save inference history
        with open(os.path.join(output_dir, 'inference_history.json'), 'w') as f:
            json.dump(self.inference_history, f, indent=2)
        
        # Save performance stats
        performance_stats = self.get_performance_stats()
        with open(os.path.join(output_dir, 'performance_stats.json'), 'w') as f:
            json.dump(performance_stats, f, indent=2)
        
        # Save system monitor data
        self.monitor.save_report(os.path.join(output_dir, 'system_report.json'))

def simulate_data_stream(duration_minutes: int = 30):
    """Simulate realistic vital sign data stream"""
    time_points = duration_minutes * 60  # Convert to seconds at 1 Hz
    
    for t in range(time_points):
        # Simulate different physiological patterns
        if t < 10 * 60:  # First 10 minutes: stable
            hr = np.random.normal(85, 5)
            sbp = np.random.normal(120, 8)
        elif t < 20 * 60:  # Next 10 minutes: early decompensation
            hr = 85 + (t - 10*60) / 600 * 20  # Gradually increase
            sbp = 120 - (t - 10*60) / 600 * 15  # Gradually decrease
        else:  # Final phase: severe decompensation
            hr = np.random.normal(110, 10)
            sbp = np.random.normal(90, 15)
        
        # Add noise and variability
        hr += np.random.normal(0, 3)
        sbp += np.random.normal(0, 5)
        
        # Calculate derived parameters
        dbp = sbp * 0.65 + np.random.normal(0, 3)
        spo2 = 98 - max(0, (t - 15*60) / 600 * 8) + np.random.normal(0, 1)
        spo2 = max(85, min(100, spo2))
        
        yield [hr, sbp, dbp, spo2]

def main():
    """Main simulation function"""
    parser = argparse.ArgumentParser(description="Trauma-Former Real-time Inference Simulator")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Configuration file")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--duration", type=int, default=30,
                       help="Simulation duration in minutes")
    parser.add_argument("--output", type=str, default="./simulation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("Trauma-Former Digital Twin Simulation")
    print("="*60)
    print(f"Simulation Duration: {args.duration} minutes")
    print(f"Model: {args.model}")
    print(f"Output Directory: {args.output}")
    print()
    
    # Initialize components
    print("Initializing system components...")
    edge = EdgeSimulator(buffer_size=config['system']['edge']['buffer_size'])
    network = NetworkSimulator(
        latency_ms=config['system']['network_latency_target_ms'],
        reliability=0.99999
    )
    digital_twin = DigitalTwinEngine(config, args.model)
    
    # Start data stream
    print("Starting sensor data stream...")
    data_generator = simulate_data_stream(args.duration)
    edge.start_sensor_stream(data_generator, interval_ms=1000)
    
    # Main simulation loop
    print("\nStarting real-time inference simulation...")
    print("-"*60)
    
    start_time = time.time()
    inference_count = 0
    
    try:
        while time.time() - start_time < args.duration * 60:
            # Get latest data from edge
            data_window = edge.get_latest_data(
                window_size=config['data_generation']['sequence_length']
            )
            
            if len(data_window) >= 10:  # Wait for some data to accumulate
                # Simulate network transmission
                transmission_data = {
                    'patient_id': 'simulated_patient_001',
                    'data_window': data_window,
                    'edge_timestamp': datetime.now().isoformat()
                }
                
                transmitted_data = network.transmit(transmission_data)
                
                if transmitted_data:
                    # Perform inference
                    result = digital_twin.infer(
                        transmitted_data['data_window'],
                        transmitted_data['patient_id']
                    )
                    
                    inference_count += 1
                    
                    # Display progress
                    if inference_count % 10 == 0:
                        risk_score = result.get('risk_score', 0)
                        alert = result.get('alert_triggered', False)
                        latency = result.get('latency_ms', {}).get('total', 0)
                        
                        alert_indicator = "🚨" if alert else "✓"
                        print(f"Inference {inference_count:4d} | "
                              f"Risk: {risk_score:.3f} {alert_indicator} | "
                              f"Latency: {latency:5.1f} ms")
            
            time.sleep(0.1)  # Small delay to prevent CPU overuse
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # Cleanup
        edge.stop()
        
        # Save results
        print("\n" + "="*60)
        print("Simulation Complete")
        print("="*60)
        
        digital_twin.save_results(args.output)
        
        # Print summary statistics
        perf_stats = digital_twin.get_performance_stats()
        network_stats = network.get_stats()
        
        print(f"\nPerformance Summary:")
        print(f"  Total Inferences: {perf_stats.get('total_inferences', 0)}")
        print(f"  Average Latency: {perf_stats.get('avg_latency_ms', 0):.2f} ms")
        print(f"  P95 Latency: {perf_stats.get('p95_latency_ms', 0):.2f} ms")
        print(f"  P99 Latency: {perf_stats.get('p99_latency_ms', 0):.2f} ms")
        
        print(f"\nNetwork Statistics:")
        print(f"  Packets Transmitted: {network_stats.get('total_packets', 0)}")
        print(f"  Packet Loss Rate: {network_stats.get('packet_loss_rate', 0)*100:.6f}%")
        
        print(f"\nResults saved to: {args.output}")
        print("="*60)

if __name__ == "__main__":
    main()