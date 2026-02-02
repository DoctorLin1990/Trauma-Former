"""
Edge Device Simulator for Trauma-Former Digital Twin
Simulates ambulance-based edge computing device for real-time vital sign processing
"""

import numpy as np
import pandas as pd
import asyncio
import aiohttp
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import socket
import struct
import hashlib
import zlib
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EdgeSimulator")

class SensorType(Enum):
    """Types of medical sensors"""
    ECG = "ecg"
    PPG = "ppg"
    NIBP = "nibp"  # Non-invasive blood pressure
    SPO2 = "spo2"
    TEMP = "temperature"
    RESP = "respiration"
    CO2 = "etco2"

@dataclass
class VitalSign:
    """Data structure for vital sign measurements"""
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: datetime
    quality: float  # 0.0 to 1.0, signal quality
    patient_id: str
    sensor_id: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['sensor_type'] = self.sensor_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VitalSign':
        """Create from dictionary"""
        data['sensor_type'] = SensorType(data['sensor_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class EdgeDeviceConfig:
    """Configuration for edge device simulation"""
    # Device identification
    device_id: str = "ambulance_001"
    ambulance_id: str = "AMBU-2024-001"
    hospital_id: str = "HOSP-FJMU-002"
    
    # Network configuration
    cloud_endpoint: str = "https://digitaltwin.hospital.com/api/v1/ingest"
    cloud_api_key: str = "trauma-former-api-key-2024"
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Sensor configuration
    sampling_rates: Dict[SensorType, float] = None
    sensor_accuracy: Dict[SensorType, float] = None
    
    # Processing configuration
    buffer_size: int = 1000
    preprocessing_enabled: bool = True
    compression_enabled: bool = True
    encryption_enabled: bool = True
    
    # Power management
    battery_capacity: float = 100.0  # Wh
    power_consumption: Dict[str, float] = None
    
    # QoS parameters
    target_latency_ms: float = 5.0
    max_packet_loss: float = 0.01
    min_signal_quality: float = 0.7
    
    def __post_init__(self):
        """Initialize default values"""
        if self.sampling_rates is None:
            self.sampling_rates = {
                SensorType.ECG: 250.0,   # Hz
                SensorType.PPG: 100.0,
                SensorType.NIBP: 0.033,  # ~ every 30 seconds
                SensorType.SPO2: 1.0,
                SensorType.TEMP: 0.1,    # every 10 seconds
                SensorType.RESP: 0.5,
                SensorType.CO2: 0.2,
            }
        
        if self.sensor_accuracy is None:
            self.sensor_accuracy = {
                SensorType.ECG: 0.95,
                SensorType.PPG: 0.90,
                SensorType.NIBP: 0.85,
                SensorType.SPO2: 0.92,
                SensorType.TEMP: 0.98,
                SensorType.RESP: 0.88,
                SensorType.CO2: 0.90,
            }
        
        if self.power_consumption is None:
            self.power_consumption = {
                'sensors': 5.0,      # Watts
                'processing': 8.0,   # Watts
                'wireless': 12.0,    # Watts
                'display': 3.0,      # Watts
            }

class EdgeDataProcessor:
    """Processes sensor data on the edge device"""
    
    def __init__(self, config: EdgeDeviceConfig):
        self.config = config
        self.data_buffer = queue.Queue(maxsize=config.buffer_size)
        self.processing_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'data_points_received': 0,
            'data_points_processed': 0,
            'data_points_transmitted': 0,
            'processing_errors': 0,
            'avg_processing_latency_ms': 0.0,
            'buffer_usage_percent': 0.0,
        }
        
        # Initialize filters
        self._init_filters()
    
    def _init_filters(self):
        """Initialize signal processing filters"""
        # Simple moving average filters for different sensor types
        self.filters = {
            SensorType.ECG: self._butterworth_filter,
            SensorType.PPG: self._median_filter,
            SensorType.SPO2: self._moving_average,
            SensorType.RESP: self._moving_average,
            SensorType.CO2: self._moving_average,
        }
    
    def start(self):
        """Start the data processor"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Edge data processor started")
    
    def stop(self):
        """Stop the data processor"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        logger.info("Edge data processor stopped")
    
    def add_sensor_data(self, vital_sign: VitalSign):
        """Add sensor data to processing queue"""
        try:
            if not self.data_buffer.full():
                self.data_buffer.put(vital_sign)
                self.stats['data_points_received'] += 1
                
                # Update buffer usage
                self.stats['buffer_usage_percent'] = (
                    self.data_buffer.qsize() / self.config.buffer_size * 100
                )
                
                return True
            else:
                logger.warning(f"Buffer full, dropping data from {vital_sign.sensor_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding sensor data: {e}")
            return False
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Process data from buffer
                if not self.data_buffer.empty():
                    vital_sign = self.data_buffer.get(timeout=0.1)
                    start_time = time.time()
                    
                    # Process the data
                    processed_data = self.process_vital_sign(vital_sign)
                    
                    # Calculate processing latency
                    processing_time = (time.time() - start_time) * 1000  # ms
                    self.stats['avg_processing_latency_ms'] = (
                        0.9 * self.stats['avg_processing_latency_ms'] + 
                        0.1 * processing_time
                    )
                    
                    self.stats['data_points_processed'] += 1
                    
                    # TODO: Send to cloud or local storage
                    
                else:
                    # Buffer empty, sleep briefly
                    time.sleep(0.01)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.stats['processing_errors'] += 1
                time.sleep(0.1)
    
    def process_vital_sign(self, vital_sign: VitalSign) -> Dict:
        """Process a single vital sign measurement"""
        try:
            # Apply sensor-specific processing
            raw_value = vital_sign.value
            
            # Check signal quality
            if vital_sign.quality < self.config.min_signal_quality:
                logger.warning(f"Low quality signal from {vital_sign.sensor_id}: {vital_sign.quality}")
            
            # Apply filtering if enabled
            if self.config.preprocessing_enabled and vital_sign.sensor_type in self.filters:
                processed_value = self.filters[vital_sign.sensor_type](
                    raw_value, vital_sign.sensor_type
                )
            else:
                processed_value = raw_value
            
            # Apply sensor calibration
            calibrated_value = self._apply_calibration(
                processed_value, vital_sign.sensor_type
            )
            
            # Create processed data packet
            processed_data = {
                **vital_sign.to_dict(),
                'processed_value': calibrated_value,
                'processing_timestamp': datetime.now().isoformat(),
                'edge_device_id': self.config.device_id,
                'metadata': {
                    'signal_quality': vital_sign.quality,
                    'sensor_accuracy': self.config.sensor_accuracy.get(vital_sign.sensor_type, 0.9),
                    'battery_level': self.get_battery_level(),
                    'network_strength': self.get_network_strength(),
                }
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing vital sign: {e}")
            raise
    
    def _butterworth_filter(self, value: float, sensor_type: SensorType) -> float:
        """Apply Butterworth filter (simplified implementation)"""
        # Simplified version - in reality would maintain state
        return value * 0.9 + np.random.normal(0, 0.01)  # Add small noise for realism
    
    def _median_filter(self, value: float, sensor_type: SensorType) -> float:
        """Apply median filter (simplified)"""
        return value  # Simplified
    
    def _moving_average(self, value: float, sensor_type: SensorType) -> float:
        """Apply moving average filter"""
        # Simplified - would use window of previous values
        return value * 0.95 + np.random.normal(0, 0.005)
    
    def _apply_calibration(self, value: float, sensor_type: SensorType) -> float:
        """Apply sensor calibration"""
        # In real implementation, would use calibration curves
        accuracy = self.config.sensor_accuracy.get(sensor_type, 0.9)
        
        # Simulate calibration error
        calibration_error = np.random.normal(0, (1 - accuracy) * 0.1)
        return value * (1 + calibration_error)
    
    def get_battery_level(self) -> float:
        """Get current battery level (simulated)"""
        # Simulate battery drain
        return max(0, 100 - (self.stats['data_points_processed'] * 0.0001))
    
    def get_network_strength(self) -> float:
        """Get network signal strength (simulated)"""
        # Simulate variable network conditions
        return 0.8 + 0.2 * np.sin(time.time() / 10)
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return self.stats.copy()

class NetworkManager:
    """Manages network connectivity for edge device"""
    
    def __init__(self, config: EdgeDeviceConfig):
        self.config = config
        self.connection_status = 'disconnected'
        self.last_heartbeat = None
        self.network_stats = {
            'upload_success': 0,
            'upload_failed': 0,
            'total_bytes_sent': 0,
            'avg_upload_latency_ms': 0.0,
            'current_rtt_ms': 100.0,
            'packet_loss_rate': 0.0,
        }
        
        # Network simulation
        self.network_conditions = {
            'latency_ms': 18.0,  # 5G URLLC target
            'reliability': 0.9999,
            'bandwidth_mbps': 100.0,
        }
    
    async def connect(self) -> bool:
        """Connect to cloud endpoint"""
        try:
            logger.info(f"Connecting to {self.config.cloud_endpoint}")
            
            # Simulate connection delay
            await asyncio.sleep(0.1)
            
            # Simulate network conditions
            if np.random.random() > 0.95:  # 5% chance of connection failure
                logger.error("Connection failed: Network unreachable")
                self.connection_status = 'disconnected'
                return False
            
            self.connection_status = 'connected'
            self.last_heartbeat = datetime.now()
            logger.info("Connected to cloud endpoint")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connection_status = 'error'
            return False
    
    async def disconnect(self):
        """Disconnect from cloud endpoint"""
        self.connection_status = 'disconnected'
        logger.info("Disconnected from cloud endpoint")
    
    async def send_data(self, data: Dict, priority: str = 'normal') -> Tuple[bool, float]:
        """Send data to cloud endpoint"""
        if self.connection_status != 'connected':
            logger.warning("Not connected, attempting to reconnect...")
            if not await self.connect():
                return False, 0.0
        
        start_time = time.time()
        
        try:
            # Prepare data for transmission
            payload = self._prepare_payload(data, priority)
            
            # Simulate network transmission
            transmission_success = await self._simulate_transmission(payload)
            
            if transmission_success:
                transmission_time = (time.time() - start_time) * 1000  # ms
                
                # Update statistics
                self.network_stats['upload_success'] += 1
                self.network_stats['total_bytes_sent'] += len(json.dumps(payload))
                self.network_stats['avg_upload_latency_ms'] = (
                    0.9 * self.network_stats['avg_upload_latency_ms'] + 
                    0.1 * transmission_time
                )
                
                logger.debug(f"Data sent successfully in {transmission_time:.2f} ms")
                return True, transmission_time
            else:
                self.network_stats['upload_failed'] += 1
                logger.warning("Data transmission failed")
                return False, 0.0
                
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            self.network_stats['upload_failed'] += 1
            return False, 0.0
    
    def _prepare_payload(self, data: Dict, priority: str) -> Dict:
        """Prepare data payload for transmission"""
        payload = {
            'device_id': self.config.device_id,
            'ambulance_id': self.config.ambulance_id,
            'hospital_id': self.config.hospital_id,
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'data': data,
            'metadata': {
                'battery_level': 85.5,  # Simulated
                'location': self._get_location(),
                'network_conditions': self.network_conditions,
                'qos_requirements': {
                    'max_latency_ms': 100.0,
                    'min_reliability': 0.999,
                }
            }
        }
        
        # Add digital signature
        if self.config.encryption_enabled:
            payload['signature'] = self._generate_signature(payload)
        
        # Compress if enabled
        if self.config.compression_enabled:
            payload = self._compress_payload(payload)
        
        return payload
    
    async def _simulate_transmission(self, payload: Dict) -> bool:
        """Simulate network transmission with realistic conditions"""
        # Simulate network latency
        latency = self.network_conditions['latency_ms']
        await asyncio.sleep(latency / 1000.0)  # Convert to seconds
        
        # Simulate packet loss
        if np.random.random() > self.network_conditions['reliability']:
            logger.warning("Packet lost during transmission")
            return False
        
        # Simulate bandwidth constraints
        payload_size = len(json.dumps(payload))
        transmission_time = payload_size / (self.network_conditions['bandwidth_mbps'] * 125000)  # Convert to seconds
        await asyncio.sleep(transmission_time)
        
        # Update RTT measurement
        self.network_stats['current_rtt_ms'] = 0.9 * self.network_stats['current_rtt_ms'] + 0.1 * (latency + transmission_time * 1000)
        
        return True
    
    def _get_location(self) -> Dict:
        """Get simulated device location"""
        # In real implementation, would use GPS
        return {
            'latitude': 24.907 + np.random.normal(0, 0.001),
            'longitude': 118.586 + np.random.normal(0, 0.001),
            'altitude': 15.0 + np.random.normal(0, 1.0),
            'accuracy': 10.0,  # meters
            'timestamp': datetime.now().isoformat(),
        }
    
    def _generate_signature(self, payload: Dict) -> str:
        """Generate digital signature for payload"""
        # Simplified signature generation
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode()).hexdigest()[:16]
    
    def _compress_payload(self, payload: Dict) -> Dict:
        """Compress payload (simplified)"""
        # In real implementation, would use gzip or similar
        return {
            'compressed': True,
            'original_size': len(json.dumps(payload)),
            'data': payload  # Simplified - no actual compression
        }
    
    async def send_heartbeat(self):
        """Send heartbeat to cloud"""
        heartbeat = {
            'device_id': self.config.device_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'statistics': self.get_statistics(),
        }
        
        success, latency = await self.send_data(heartbeat, priority='low')
        
        if success:
            self.last_heartbeat = datetime.now()
            logger.debug(f"Heartbeat sent successfully (latency: {latency:.2f} ms)")
        else:
            logger.warning("Heartbeat failed")
        
        return success
    
    def get_statistics(self) -> Dict:
        """Get network statistics"""
        return self.network_stats.copy()
    
    def get_connection_status(self) -> str:
        """Get current connection status"""
        return self.connection_status

class PowerManager:
    """Manages power consumption and battery life"""
    
    def __init__(self, config: EdgeDeviceConfig):
        self.config = config
        self.battery_level = config.battery_capacity  # Wh
        self.last_update = datetime.now()
        
        self.power_consumption = {
            'sensors': config.power_consumption['sensors'],
            'processing': config.power_consumption['processing'],
            'wireless': config.power_consumption['wireless'],
            'display': config.power_consumption['display'],
            'idle': 1.0,
        }
        
        self.current_mode = 'active'
        self.power_modes = {
            'active': 1.0,      # Full power
            'power_save': 0.5,  # 50% power
            'emergency': 0.3,   # 30% power
            'critical': 0.1,    # 10% power
        }
    
    def update(self, active_components: List[str] = None):
        """Update battery level based on active components"""
        if active_components is None:
            active_components = ['sensors', 'processing', 'wireless']
        
        current_time = datetime.now()
        time_delta = (current_time - self.last_update).total_seconds() / 3600  # hours
        
        # Calculate power consumption
        total_power = 0.0
        for component in active_components:
            if component in self.power_consumption:
                total_power += self.power_consumption[component]
        
        # Apply power mode
        total_power *= self.power_modes.get(self.current_mode, 1.0)
        
        # Update battery level
        energy_used = total_power * time_delta  # Wh
        self.battery_level = max(0, self.battery_level - energy_used)
        
        self.last_update = current_time
        
        # Check battery status and adjust mode if needed
        self._check_battery_status()
    
    def _check_battery_status(self):
        """Check battery level and adjust power mode"""
        battery_percent = (self.battery_level / self.config.battery_capacity) * 100
        
        if battery_percent < 10:
            self.current_mode = 'critical'
            logger.warning(f"Critical battery: {battery_percent:.1f}%")
        elif battery_percent < 25:
            self.current_mode = 'emergency'
            logger.warning(f"Low battery: {battery_percent:.1f}%")
        elif battery_percent < 50:
            self.current_mode = 'power_save'
            logger.info(f"Battery saving mode: {battery_percent:.1f}%")
        else:
            self.current_mode = 'active'
    
    def get_battery_info(self) -> Dict:
        """Get current battery information"""
        battery_percent = (self.battery_level / self.config.battery_capacity) * 100
        
        # Estimate remaining time
        current_power = sum([
            self.power_consumption['sensors'],
            self.power_consumption['processing'],
            self.power_consumption['wireless'],
        ]) * self.power_modes.get(self.current_mode, 1.0)
        
        if current_power > 0:
            remaining_hours = self.battery_level / current_power
        else:
            remaining_hours = float('inf')
        
        return {
            'level_percent': battery_percent,
            'level_wh': self.battery_level,
            'capacity_wh': self.config.battery_capacity,
            'power_mode': self.current_mode,
            'estimated_remaining_hours': remaining_hours,
            'charging': False,  # Simulated - would detect if charging
        }
    
    def set_power_mode(self, mode: str):
        """Set power management mode"""
        if mode in self.power_modes:
            self.current_mode = mode
            logger.info(f"Power mode set to: {mode}")
            return True
        else:
            logger.error(f"Invalid power mode: {mode}")
            return False

class SensorSimulator:
    """Simulates medical sensors for testing"""
    
    def __init__(self, patient_id: str = "patient_001"):
        self.patient_id = patient_id
        self.sensors = self._initialize_sensors()
        self.last_values = {}
        
        # Physiological parameters
        self.physiological_state = {
            'heart_rate': 85.0,      # bpm
            'systolic_bp': 125.0,    mmHg
            'diastolic_bp': 80.0,    mmHg
            'spo2': 98.0,           # %
            'respiration_rate': 16.0, # breaths/min
            'temperature': 37.0,     # °C
            'etco2': 35.0,          # mmHg
        }
        
        # Trend parameters (for simulating deterioration)
        self.trends = {
            'heart_rate': 0.0,      # bpm per minute
            'systolic_bp': 0.0,     # mmHg per minute
            'spo2': 0.0,            # % per minute
        }
    
    def _initialize_sensors(self) -> Dict[str, Dict]:
        """Initialize sensor configurations"""
        return {
            'ecg_001': {
                'type': SensorType.ECG,
                'unit': 'mV',
                'accuracy': 0.95,
                'noise_level': 0.01,
                'sampling_rate': 250.0,
            },
            'nibp_001': {
                'type': SensorType.NIBP,
                'unit': 'mmHg',
                'accuracy': 0.85,
                'noise_level': 2.0,
                'sampling_rate': 0.033,  # Every 30 seconds
            },
            'spo2_001': {
                'type': SensorType.SPO2,
                'unit': '%',
                'accuracy': 0.92,
                'noise_level': 0.5,
                'sampling_rate': 1.0,
            },
            'temp_001': {
                'type': SensorType.TEMP,
                'unit': '°C',
                'accuracy': 0.98,
                'noise_level': 0.1,
                'sampling_rate': 0.1,  # Every 10 seconds
            },
            'resp_001': {
                'type': SensorType.RESP,
                'unit': 'breaths/min',
                'accuracy': 0.88,
                'noise_level': 1.0,
                'sampling_rate': 0.5,
            },
            'co2_001': {
                'type': SensorType.CO2,
                'unit': 'mmHg',
                'accuracy': 0.90,
                'noise_level': 1.0,
                'sampling_rate': 0.2,
            },
        }
    
    def generate_vital_sign(self, sensor_id: str) -> Optional[VitalSign]:
        """Generate a vital sign reading from a sensor"""
        if sensor_id not in self.sensors:
            logger.error(f"Unknown sensor: {sensor_id}")
            return None
        
        sensor_config = self.sensors[sensor_id]
        sensor_type = sensor_config['type']
        
        # Get base value from physiological state
        if sensor_type == SensorType.ECG:
            base_value = self.physiological_state['heart_rate']
        elif sensor_type == SensorType.NIBP:
            base_value = self.physiological_state['systolic_bp']
        elif sensor_type == SensorType.SPO2:
            base_value = self.physiological_state['spo2']
        elif sensor_type == SensorType.TEMP:
            base_value = self.physiological_state['temperature']
        elif sensor_type == SensorType.RESP:
            base_value = self.physiological_state['respiration_rate']
        elif sensor_type == SensorType.CO2:
            base_value = self.physiological_state['etco2']
        else:
            base_value = 0.0
        
        # Apply trend
        param_name = self._get_parameter_name(sensor_type)
        if param_name in self.trends:
            trend_value = self.trends[param_name] * (time.time() / 60.0)  # Convert to minutes
            base_value += trend_value
        
        # Add noise
        noise = np.random.normal(0, sensor_config['noise_level'])
        value = base_value + noise
        
        # Add physiological variability
        if sensor_type == SensorType.ECG:
            # Add heart rate variability
            hr_variability = 5.0 * np.sin(time.time() * 2 * np.pi / 5.0)  # 5 second cycle
            value += hr_variability
        elif sensor_type == SensorType.RESP:
            # Add respiratory variability
            resp_variability = 2.0 * np.sin(time.time() * 2 * np.pi / 3.0)  # 3 second cycle
            value += resp_variability
        
        # Clip to physiological ranges
        value = self._clip_to_range(value, sensor_type)
        
        # Calculate signal quality (simulated)
        quality = sensor_config['accuracy'] * (0.9 + 0.1 * np.random.random())
        
        # Create VitalSign object
        vital_sign = VitalSign(
            sensor_type=sensor_type,
            value=value,
            unit=sensor_config['unit'],
            timestamp=datetime.now(),
            quality=quality,
            patient_id=self.patient_id,
            sensor_id=sensor_id,
        )
        
        # Store last value
        self.last_values[sensor_id] = value
        
        return vital_sign
    
    def _get_parameter_name(self, sensor_type: SensorType) -> Optional[str]:
        """Get physiological parameter name for sensor type"""
        mapping = {
            SensorType.ECG: 'heart_rate',
            SensorType.NIBP: 'systolic_bp',
            SensorType.SPO2: 'spo2',
            SensorType.TEMP: 'temperature',
            SensorType.RESP: 'respiration_rate',
            SensorType.CO2: 'etco2',
        }
        return mapping.get(sensor_type)
    
    def _clip_to_range(self, value: float, sensor_type: SensorType) -> float:
        """Clip value to physiological range"""
        ranges = {
            SensorType.ECG: (30, 250),       # bpm
            SensorType.NIBP: (40, 250),      # mmHg
            SensorType.SPO2: (70, 100),      # %
            SensorType.TEMP: (32, 42),       # °C
            SensorType.RESP: (4, 60),        # breaths/min
            SensorType.CO2: (10, 100),       # mmHg
        }
        
        if sensor_type in ranges:
            min_val, max_val = ranges[sensor_type]
            return max(min_val, min(max_val, value))
        
        return value
    
    def set_trend(self, parameter: str, trend_per_minute: float):
        """Set trend for a physiological parameter"""
        if parameter in self.trends:
            self.trends[parameter] = trend_per_minute
            logger.info(f"Set trend for {parameter}: {trend_per_minute} per minute")
            return True
        else:
            logger.error(f"Unknown parameter: {parameter}")
            return False
    
    def set_physiological_state(self, state: Dict):
        """Set physiological state"""
        for param, value in state.items():
            if param in self.physiological_state:
                self.physiological_state[param] = value
        
        logger.info(f"Updated physiological state: {state}")
    
    def simulate_deterioration(self, severity: str = 'moderate'):
        """Simulate patient deterioration"""
        trends = {
            'mild': {
                'heart_rate': 2.0,      # +2 bpm per minute
                'systolic_bp': -1.0,    # -1 mmHg per minute
                'spo2': -0.1,           # -0.1% per minute
            },
            'moderate': {
                'heart_rate': 5.0,
                'systolic_bp': -3.0,
                'spo2': -0.3,
            },
            'severe': {
                'heart_rate': 10.0,
                'systolic_bp': -8.0,
                'spo2': -0.8,
            },
        }
        
        if severity in trends:
            for param, trend in trends[severity].items():
                self.set_trend(param, trend)
            
            logger.info(f"Simulating {severity} deterioration")
            return True
        else:
            logger.error(f"Unknown severity level: {severity}")
            return False

class EdgeDeviceSimulator:
    """Main edge device simulator"""
    
    def __init__(self, config: EdgeDeviceConfig = None):
        self.config = config or EdgeDeviceConfig()
        
        # Initialize components
        self.data_processor = EdgeDataProcessor(self.config)
        self.network_manager = NetworkManager(self.config)
        self.power_manager = PowerManager(self.config)
        self.sensor_simulator = SensorSimulator()
        
        # State management
        self.running = False
        self.sensor_threads = {}
        self.heartbeat_task = None
        
        # Statistics
        self.start_time = None
        self.total_data_points = 0
    
    async def start(self):
        """Start the edge device simulator"""
        if self.running:
            logger.warning("Edge device already running")
            return
        
        logger.info("Starting edge device simulator...")
        self.running = True
        self.start_time = datetime.now()
        
        # Start components
        self.data_processor.start()
        
        # Connect to network
        await self.network_manager.connect()
        
        # Start sensor simulations
        self._start_sensor_simulations()
        
        # Start heartbeat
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start power management updates
        asyncio.create_task(self._power_management_loop())
        
        logger.info("Edge device simulator started successfully")
    
    async def stop(self):
        """Stop the edge device simulator"""
        if not self.running:
            return
        
        logger.info("Stopping edge device simulator...")
        self.running = False
        
        # Stop sensor simulations
        self._stop_sensor_simulations()
        
        # Stop heartbeat
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Stop components
        self.data_processor.stop()
        await self.network_manager.disconnect()
        
        logger.info("Edge device simulator stopped")
    
    def _start_sensor_simulations(self):
        """Start sensor simulation threads"""
        for sensor_id, sensor_config in self.sensor_simulator.sensors.items():
            thread = threading.Thread(
                target=self._sensor_simulation_loop,
                args=(sensor_id, sensor_config),
                daemon=True
            )
            thread.start()
            self.sensor_threads[sensor_id] = thread
    
    def _stop_sensor_simulations(self):
        """Stop all sensor simulation threads"""
        for sensor_id, thread in self.sensor_threads.items():
            if thread.is_alive():
                # Threads are daemon, will exit when main thread exits
                pass
    
    def _sensor_simulation_loop(self, sensor_id: str, sensor_config: Dict):
        """Simulate sensor data generation"""
        sampling_rate = sensor_config['sampling_rate']
        interval = 1.0 / sampling_rate
        
        logger.info(f"Starting sensor simulation: {sensor_id} ({sampling_rate} Hz)")
        
        while self.running:
            try:
                # Generate vital sign
                vital_sign = self.sensor_simulator.generate_vital_sign(sensor_id)
                
                if vital_sign:
                    # Add to data processor
                    self.data_processor.add_sensor_data(vital_sign)
                    self.total_data_points += 1
                    
                    # Send to cloud (async, non-blocking)
                    asyncio.run_coroutine_threadsafe(
                        self._send_to_cloud(vital_sign),
                        asyncio.get_event_loop()
                    )
                
                # Wait for next sample
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in sensor simulation {sensor_id}: {e}")
                time.sleep(1.0)
    
    async def _send_to_cloud(self, vital_sign: VitalSign):
        """Send vital sign data to cloud"""
        try:
            # Process the data
            processed_data = self.data_processor.process_vital_sign(vital_sign)
            
            # Determine priority based on vital sign
            priority = self._determine_priority(vital_sign)
            
            # Send to cloud
            success, latency = await self.network_manager.send_data(
                processed_data, priority
            )
            
            if success:
                logger.debug(f"Sent {vital_sign.sensor_type.value} data to cloud "
                           f"(priority: {priority}, latency: {latency:.2f} ms)")
            else:
                logger.warning(f"Failed to send {vital_sign.sensor_type.value} data")
                
        except Exception as e:
            logger.error(f"Error sending to cloud: {e}")
    
    def _determine_priority(self, vital_sign: VitalSign) -> str:
        """Determine transmission priority based on vital sign"""
        high_priority_sensors = {SensorType.ECG, SensorType.NIBP, SensorType.SPO2}
        
        if vital_sign.sensor_type in high_priority_sensors:
            # Check if value is critical
            if self._is_critical_value(vital_sign):
                return 'critical'
            else:
                return 'high'
        else:
            return 'normal'
    
    def _is_critical_value(self, vital_sign: VitalSign) -> bool:
        """Check if vital sign value is critical"""
        critical_ranges = {
            SensorType.ECG: (40, 140),     # bpm
            SensorType.NIBP: (70, 180),    # mmHg (systolic)
            SensorType.SPO2: (90, 100),    # %
        }
        
        if vital_sign.sensor_type in critical_ranges:
            min_val, max_val = critical_ranges[vital_sign.sensor_type]
            value = vital_sign.value
            
            # For NIBP, we might want to check both systolic and diastolic
            if vital_sign.sensor_type == SensorType.NIBP:
                # Simplified - in reality would have separate systolic/diastolic
                return value < min_val or value > max_val
            else:
                return value < min_val or value > max_val
        
        return False
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to cloud"""
        while self.running:
            try:
                await self.network_manager.send_heartbeat()
                await asyncio.sleep(30.0)  # Every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _power_management_loop(self):
        """Update power management periodically"""
        while self.running:
            try:
                self.power_manager.update()
                await asyncio.sleep(10.0)  # Update every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in power management loop: {e}")
                await asyncio.sleep(5.0)
    
    def get_status(self) -> Dict:
        """Get current device status"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'device_id': self.config.device_id,
            'status': 'running' if self.running else 'stopped',
            'uptime_seconds': uptime,
            'connection_status': self.network_manager.get_connection_status(),
            'battery': self.power_manager.get_battery_info(),
            'statistics': {
                'total_data_points': self.total_data_points,
                'data_processor': self.data_processor.get_statistics(),
                'network': self.network_manager.get_statistics(),
            },
            'sensors_active': len(self.sensor_threads),
            'timestamp': datetime.now().isoformat(),
        }
    
    def simulate_emergency(self, scenario: str = 'trauma'):
        """Simulate emergency scenarios"""
        scenarios = {
            'trauma': {
                'heart_rate': 120.0,
                'systolic_bp': 90.0,
                'spo2': 88.0,
                'respiration_rate': 28.0,
            },
            'cardiac': {
                'heart_rate': 150.0,
                'systolic_bp': 70.0,
                'spo2': 85.0,
                'respiration_rate': 32.0,
            },
            'respiratory': {
                'heart_rate': 110.0,
                'systolic_bp': 100.0,
                'spo2': 75.0,
                'respiration_rate': 40.0,
            },
        }
        
        if scenario in scenarios:
            self.sensor_simulator.set_physiological_state(scenarios[scenario])
            logger.info(f"Simulating {scenario} emergency scenario")
            return True
        else:
            logger.error(f"Unknown emergency scenario: {scenario}")
            return False
    
    def set_power_mode(self, mode: str):
        """Set power management mode"""
        return self.power_manager.set_power_mode(mode)

async def main():
    """Main function to run edge device simulator"""
    # Create configuration
    config = EdgeDeviceConfig(
        device_id="ambulance_001",
        ambulance_id="AMBU-FJMU-2024-001",
        hospital_id="HOSP-FJMU-002",
        cloud_endpoint="https://digitaltwin.hospital-fjmu.com/api/v1/ingest",
    )
    
    # Create edge device simulator
    edge_device = EdgeDeviceSimulator(config)
    
    try:
        # Start the edge device
        await edge_device.start()
        
        # Run for a specified duration or until interrupted
        print("\n" + "="*60)
        print("Trauma-Former Edge Device Simulator")
        print("="*60)
        print(f"Device ID: {config.device_id}")
        print(f"Ambulance: {config.ambulance_id}")
        print(f"Hospital: {config.hospital_id}")
        print(f"Cloud Endpoint: {config.cloud_endpoint}")
        print("\nCommands:")
        print("  status    - Show device status")
        print("  emergency - Simulate emergency scenario")
        print("  power     - Change power mode")
        print("  stop      - Stop the simulator")
        print("="*60)
        
        # Interactive command loop
        while edge_device.running:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\nCommand: "
                ).strip().lower()
                
                if command == 'status':
                    status = edge_device.get_status()
                    print("\n" + json.dumps(status, indent=2, default=str))
                
                elif command == 'emergency':
                    print("\nEmergency scenarios:")
                    print("  1. trauma (trauma patient)")
                    print("  2. cardiac (cardiac arrest)")
                    print("  3. respiratory (respiratory failure)")
                    
                    scenario = input("Select scenario (1-3): ").strip()
                    scenarios = {'1': 'trauma', '2': 'cardiac', '3': 'respiratory'}
                    
                    if scenario in scenarios:
                        edge_device.simulate_emergency(scenarios[scenario])
                        print(f"Started {scenarios[scenario]} emergency simulation")
                    else:
                        print("Invalid selection")
                
                elif command == 'power':
                    print("\nPower modes:")
                    print("  1. active (full power)")
                    print("  2. power_save (50% power)")
                    print("  3. emergency (30% power)")
                    print("  4. critical (10% power)")
                    
                    mode = input("Select mode (1-4): ").strip()
                    modes = {'1': 'active', '2': 'power_save', 
                            '3': 'emergency', '4': 'critical'}
                    
                    if mode in modes:
                        edge_device.set_power_mode(modes[mode])
                        print(f"Set power mode to {modes[mode]}")
                    else:
                        print("Invalid selection")
                
                elif command == 'stop':
                    print("Stopping edge device...")
                    await edge_device.stop()
                    break
                
                else:
                    print(f"Unknown command: {command}")
                
            except (KeyboardInterrupt, EOFError):
                print("\nStopping edge device...")
                await edge_device.stop()
                break
            except Exception as e:
                print(f"Error: {e}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        await edge_device.stop()
    
    print("\nEdge device simulator stopped.")

if __name__ == "__main__":
    # Run the edge device simulator
    asyncio.run(main())