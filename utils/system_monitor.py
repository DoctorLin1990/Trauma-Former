"""
System Monitoring for Trauma-Former Digital Twin
Monitors system performance, resources, and latency
"""

import psutil
import GPUtil
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import os
from collections import deque
import warnings

class ResourceMonitor:
    """Monitors system resource usage"""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize resource monitor
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Data storage
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.disk_io = deque(maxlen=1000)
        self.network_io = deque(maxlen=1000)
        self.gpu_usage = deque(maxlen=1000) if self.has_gpu() else None
        self.timestamps = deque(maxlen=1000)
        
        # Counters for I/O
        self.prev_disk_io = psutil.disk_io_counters()
        self.prev_net_io = psutil.net_io_counters()
        self.prev_time = time.time()
    
    def has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            GPUtil.getGPUs()
            return True
        except:
            return False
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            self.update_metrics()
            time.sleep(self.update_interval)
    
    def update_metrics(self):
        """Update all resource metrics"""
        timestamp = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_usage.append(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.memory_usage.append(memory_percent)
        
        # Disk I/O
        current_disk_io = psutil.disk_io_counters()
        current_time = time.time()
        time_diff = current_time - self.prev_time
        
        if time_diff > 0:
            read_mbps = (current_disk_io.read_bytes - self.prev_disk_io.read_bytes) / (1024**2 * time_diff)
            write_mbps = (current_disk_io.write_bytes - self.prev_disk_io.write_bytes) / (1024**2 * time_diff)
            
            self.disk_io.append({
                'read_mbps': read_mbps,
                'write_mbps': write_mbps,
                'read_iops': (current_disk_io.read_count - self.prev_disk_io.read_count) / time_diff,
                'write_iops': (current_disk_io.write_count - self.prev_disk_io.write_count) / time_diff
            })
            
            self.prev_disk_io = current_disk_io
        
        # Network I/O
        current_net_io = psutil.net_io_counters()
        
        if time_diff > 0:
            rx_mbps = (current_net_io.bytes_recv - self.prev_net_io.bytes_recv) / (1024**2 * time_diff)
            tx_mbps = (current_net_io.bytes_sent - self.prev_net_io.bytes_sent) / (1024**2 * time_diff)
            
            self.network_io.append({
                'rx_mbps': rx_mbps,
                'tx_mbps': tx_mbps,
                'rx_pps': (current_net_io.packets_recv - self.prev_net_io.packets_recv) / time_diff,
                'tx_pps': (current_net_io.packets_sent - self.prev_net_io.packets_sent) / time_diff
            })
            
            self.prev_net_io = current_net_io
        
        # GPU usage (if available)
        if self.gpu_usage is not None:
            try:
                gpus = GPUtil.getGPUs()
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,  # Percentage
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    })
                self.gpu_usage.append(gpu_info)
            except Exception as e:
                warnings.warn(f"Failed to get GPU metrics: {e}")
        
        # Store timestamp
        self.timestamps.append(timestamp)
        self.prev_time = current_time
    
    def get_current_stats(self) -> Dict:
        """Get current resource statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage': psutil.disk_usage('/').percent,
        }
        
        if self.gpu_usage and len(self.gpu_usage) > 0:
            stats['gpu_stats'] = self.gpu_usage[-1]
        
        return stats
    
    def get_historical_stats(self, last_n: Optional[int] = None) -> Dict:
        """Get historical statistics"""
        if last_n is None:
            last_n = len(self.cpu_usage)
        
        if last_n == 0:
            return {}
        
        # Get last N samples
        start_idx = max(0, len(self.cpu_usage) - last_n)
        
        cpu_array = np.array(list(self.cpu_usage)[start_idx:])
        memory_array = np.array(list(self.memory_usage)[start_idx:])
        
        stats = {
            'cpu': {
                'mean': float(np.mean(cpu_array)),
                'std': float(np.std(cpu_array)),
                'min': float(np.min(cpu_array)),
                'max': float(np.max(cpu_array)),
                'p95': float(np.percentile(cpu_array, 95)),
                'current': float(cpu_array[-1] if len(cpu_array) > 0 else 0)
            },
            'memory': {
                'mean': float(np.mean(memory_array)),
                'std': float(np.std(memory_array)),
                'min': float(np.min(memory_array)),
                'max': float(np.max(memory_array)),
                'p95': float(np.percentile(memory_array, 95)),
                'current': float(memory_array[-1] if len(memory_array) > 0 else 0)
            },
            'sample_count': len(cpu_array)
        }
        
        return stats
    
    def save_monitoring_data(self, output_path: str):
        """Save monitoring data to file"""
        data = {
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'cpu_usage': list(self.cpu_usage),
            'memory_usage': list(self.memory_usage),
            'disk_io': list(self.disk_io),
            'network_io': list(self.network_io),
            'gpu_usage': list(self.gpu_usage) if self.gpu_usage else []
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Monitoring data saved to {output_path}")

class LatencyMonitor:
    """Monitors system latency for real-time operations"""
    
    def __init__(self):
        self.latency_history = deque(maxlen=10000)
        self.component_latencies = {
            'edge_processing': deque(maxlen=1000),
            'network_uplink': deque(maxlen=1000),
            'cloud_inference': deque(maxlen=1000),
            'network_downlink': deque(maxlen=1000),
            'display': deque(maxlen=1000)
        }
        self.start_times = {}
    
    def start_timer(self, component: str):
        """Start timer for a component"""
        self.start_times[component] = time.time()
    
    def end_timer(self, component: str) -> float:
        """End timer for a component and return latency"""
        if component not in self.start_times:
            return 0.0
        
        latency = (time.time() - self.start_times[component]) * 1000  # Convert to ms
        self.component_latencies[component].append(latency)
        del self.start_times[component]
        
        return latency
    
    def record_end_to_end_latency(self, latency_ms: float):
        """Record end-to-end latency"""
        self.latency_history.append(latency_ms)
    
    def get_latency_stats(self, component: Optional[str] = None) -> Dict:
        """Get latency statistics for a component or overall"""
        if component:
            if component not in self.component_latencies:
                return {}
            latencies = list(self.component_latencies[component])
        else:
            latencies = list(self.latency_history)
        
        if not latencies:
            return {}
        
        latencies_array = np.array(latencies)
        
        return {
            'count': len(latencies),
            'mean_ms': float(np.mean(latencies_array)),
            'median_ms': float(np.median(latencies_array)),
            'std_ms': float(np.std(latencies_array)),
            'min_ms': float(np.min(latencies_array)),
            'max_ms': float(np.max(latencies_array)),
            'p50_ms': float(np.percentile(latencies_array, 50)),
            'p95_ms': float(np.percentile(latencies_array, 95)),
            'p99_ms': float(np.percentile(latencies_array, 99)),
            'latencies': latencies_array.tolist()
        }
    
    def check_sla_violation(self, sla_ms: float = 100.0) -> Dict:
        """Check for SLA violations"""
        if not self.latency_history:
            return {'violations': 0, 'violation_rate': 0.0}
        
        latencies = np.array(list(self.latency_history))
        violations = np.sum(latencies > sla_ms)
        violation_rate = violations / len(latencies)
        
        return {
            'total_samples': len(latencies),
            'violations': int(violations),
            'violation_rate': float(violation_rate),
            'sla_threshold_ms': sla_ms
        }
    
    def generate_latency_report(self) -> str:
        """Generate comprehensive latency report"""
        report = "System Latency Report\n"
        report += "=" * 50 + "\n\n"
        
        # End-to-end latency
        overall_stats = self.get_latency_stats()
        if overall_stats:
            report += "End-to-End Latency:\n"
            report += f"  Samples: {overall_stats['count']}\n"
            report += f"  Mean: {overall_stats['mean_ms']:.2f} ms\n"
            report += f"  Median: {overall_stats['median_ms']:.2f} ms\n"
            report += f"  P95: {overall_stats['p95_ms']:.2f} ms\n"
            report += f"  P99: {overall_stats['p99_ms']:.2f} ms\n"
            report += f"  Min: {overall_stats['min_ms']:.2f} ms\n"
            report += f"  Max: {overall_stats['max_ms']:.2f} ms\n\n"
        
        # Component latencies
        report += "Component Latencies:\n"
        for component in self.component_latencies.keys():
            stats = self.get_latency_stats(component)
            if stats:
                report += f"  {component}:\n"
                report += f"    Mean: {stats['mean_ms']:.2f} ms\n"
                report += f"    P95: {stats['p95_ms']:.2f} ms\n"
        
        # SLA compliance
        sla_check = self.check_sla_violation(100.0)
        report += f"\nSLA Compliance (100 ms threshold):\n"
        report += f"  Violation Rate: {sla_check['violation_rate']*100:.2f}%\n"
        report += f"  Violations: {sla_check['violations']}/{sla_check['total_samples']}\n"
        
        return report
    
    def save_latency_data(self, output_path: str):
        """Save latency data to file"""
        data = {
            'end_to_end_latencies': list(self.latency_history),
            'component_latencies': {
                component: list(latencies)
                for component, latencies in self.component_latencies.items()
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Latency data saved to {output_path}")

class PerformanceMonitor:
    """Monitors model and system performance"""
    
    def __init__(self):
        self.inference_times = deque(maxlen=10000)
        self.batch_sizes = deque(maxlen=10000)
        self.throughput_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        
        # Performance thresholds
        self.thresholds = {
            'inference_time_ms': 50.0,
            'throughput_ips': 20.0,  # inferences per second
            'memory_usage_gb': 8.0
        }
    
    def record_inference(self, inference_time_ms: float, batch_size: int = 1):
        """Record inference performance"""
        self.inference_times.append(inference_time_ms)
        self.batch_sizes.append(batch_size)
        
        # Calculate throughput
        throughput = (batch_size * 1000) / inference_time_ms if inference_time_ms > 0 else 0
        self.throughput_history.append(throughput)
    
    def record_memory_usage(self, memory_gb: float):
        """Record memory usage"""
        self.memory_history.append(memory_gb)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        inference_array = np.array(self.inference_times)
        throughput_array = np.array(self.throughput_history) if self.throughput_history else np.array([0])
        
        stats = {
            'inference_time_ms': {
                'mean': float(np.mean(inference_array)),
                'median': float(np.median(inference_array)),
                'std': float(np.std(inference_array)),
                'p95': float(np.percentile(inference_array, 95)),
                'count': len(inference_array)
            },
            'throughput_ips': {
                'mean': float(np.mean(throughput_array)),
                'current': float(throughput_array[-1] if len(throughput_array) > 0 else 0),
                'max': float(np.max(throughput_array) if len(throughput_array) > 0 else 0)
            }
        }
        
        # Check thresholds
        stats['threshold_violations'] = {
            'inference_time': {
                'threshold': self.thresholds['inference_time_ms'],
                'violations': np.sum(inference_array > self.thresholds['inference_time_ms']),
                'violation_rate': np.mean(inference_array > self.thresholds['inference_time_ms'])
            }
        }
        
        return stats
    
    def estimate_scalability(self, target_concurrent_patients: int) -> Dict:
        """
        Estimate system scalability for given number of concurrent patients
        """
        stats = self.get_performance_stats()
        
        if not stats:
            return {}
        
        mean_inference_ms = stats['inference_time_ms']['mean']
        mean_throughput = stats['throughput_ips']['mean']
        
        # Estimate requirements
        inference_per_patient_per_second = 1.0  # 1 Hz monitoring
        total_inferences_per_second = target_concurrent_patients * inference_per_patient_per_second
        
        # Required throughput
        required_throughput = total_inferences_per_second
        
        # Can current system handle it?
        can_handle = mean_throughput >= required_throughput
        
        # Estimate resources needed
        if mean_throughput > 0:
            scale_factor = required_throughput / mean_throughput
        else:
            scale_factor = float('inf')
        
        return {
            'target_concurrent_patients': target_concurrent_patients,
            'current_throughput': mean_throughput,
            'required_throughput': required_throughput,
            'can_handle': can_handle,
            'scale_factor': scale_factor,
            'estimated_servers_needed': max(1, int(np.ceil(scale_factor))),
            'estimated_inference_latency_ms': mean_inference_ms * scale_factor
        }

class SystemHealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.latency_monitor = LatencyMonitor()
        self.performance_monitor = PerformanceMonitor()
        
        # Health status
        self.health_status = {
            'overall': 'healthy',
            'components': {},
            'last_check': None
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'inference_latency_ms': 100.0,
            'disk_usage_percent': 90.0,
            'sla_violation_rate': 0.01  # 1%
        }
        
        # Alert history
        self.alerts = deque(maxlen=1000)
    
    def start_monitoring(self):
        """Start all monitoring components"""
        self.resource_monitor.start_monitoring()
        print("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components"""
        self.resource_monitor.stop_monitoring()
        print("System health monitoring stopped")
    
    def check_system_health(self) -> Dict:
        """Check overall system health"""
        current_time = datetime.now()
        
        # Get current stats
        resource_stats = self.resource_monitor.get_current_stats()
        latency_stats = self.latency_monitor.get_latency_stats()
        performance_stats = self.performance_monitor.get_performance_stats()
        
        # Check components
        component_health = {}
        
        # CPU health
        cpu_percent = resource_stats.get('cpu_percent', 0)
        component_health['cpu'] = {
            'status': 'warning' if cpu_percent > self.alert_thresholds['cpu_percent'] else 'healthy',
            'value': cpu_percent,
            'threshold': self.alert_thresholds['cpu_percent']
        }
        
        # Memory health
        memory_percent = resource_stats.get('memory_percent', 0)
        component_health['memory'] = {
            'status': 'warning' if memory_percent > self.alert_thresholds['memory_percent'] else 'healthy',
            'value': memory_percent,
            'threshold': self.alert_thresholds['memory_percent']
        }
        
        # Latency health
        if latency_stats:
            mean_latency = latency_stats.get('mean_ms', 0)
            component_health['latency'] = {
                'status': 'warning' if mean_latency > self.alert_thresholds['inference_latency_ms'] else 'healthy',
                'value': mean_latency,
                'threshold': self.alert_thresholds['inference_latency_ms']
            }
        
        # Determine overall health
        warning_components = [c for c in component_health.values() if c['status'] == 'warning']
        
        if warning_components:
            overall_status = 'warning'
            # Check for critical alerts
            critical_count = 0
            for component in warning_components:
                if component['value'] > component['threshold'] * 1.5:  # 50% above threshold
                    critical_count += 1
            
            if critical_count > 0:
                overall_status = 'critical'
        else:
            overall_status = 'healthy'
        
        # Update health status
        self.health_status = {
            'overall': overall_status,
            'components': component_health,
            'last_check': current_time.isoformat(),
            'resource_stats': resource_stats,
            'latency_stats': latency_stats,
            'performance_stats': performance_stats
        }
        
        # Generate alerts if needed
        if overall_status in ['warning', 'critical']:
            self.generate_alert(overall_status, component_health)
        
        return self.health_status
    
    def generate_alert(self, severity: str, components: Dict):
        """Generate system alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'components': components,
            'message': f"System health {severity}: {len([c for c in components.values() if c['status'] in ['warning', 'critical']])} components affected"
        }
        
        self.alerts.append(alert)
        
        # Log alert
        print(f"ALERT [{severity.upper()}]: {alert['message']}")
        
        # Here you could add email/SMS/other notification mechanisms
    
    def get_health_report(self) -> str:
        """Generate comprehensive health report"""
        health = self.check_system_health()
        
        report = "System Health Report\n"
        report += "=" * 50 + "\n\n"
        report += f"Overall Status: {health['overall'].upper()}\n"
        report += f"Last Check: {health['last_check']}\n\n"
        
        report += "Component Health:\n"
        for component, status in health['components'].items():
            report += f"  {component.upper()}:\n"
            report += f"    Status: {status['status']}\n"
            report += f"    Value: {status['value']:.1f} (Threshold: {status['threshold']})\n"
        
        report += "\nRecent Alerts:\n"
        for alert in list(self.alerts)[-5:]:  # Last 5 alerts
            report += f"  [{alert['timestamp']}] {alert['severity']}: {alert['message']}\n"
        
        return report
    
    def save_system_report(self, output_dir: str):
        """Save comprehensive system report"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save health status
        health_status = self.check_system_health()
        health_path = os.path.join(output_dir, f"health_status_{timestamp}.json")
        with open(health_path, 'w') as f:
            json.dump(health_status, f, indent=2)
        
        # Save resource data
        resource_path = os.path.join(output_dir, f"resource_data_{timestamp}.json")
        self.resource_monitor.save_monitoring_data(resource_path)
        
        # Save latency data
        latency_path = os.path.join(output_dir, f"latency_data_{timestamp}.json")
        self.latency_monitor.save_latency_data(latency_path)
        
        # Generate and save report
        report = self.get_health_report()
        report_path = os.path.join(output_dir, f"system_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"System report saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Create system monitor
    system_monitor = SystemHealthMonitor()
    
    # Start monitoring
    system_monitor.start_monitoring()
    
    # Simulate some operations
    print("Simulating system operations...")
    time.sleep(2)
    
    # Record some latencies
    latency_monitor = system_monitor.latency_monitor
    
    for i in range(10):
        # Simulate edge processing
        latency_monitor.start_timer('edge_processing')
        time.sleep(0.001)  # 1 ms
        edge_latency = latency_monitor.end_timer('edge_processing')
        
        # Simulate network uplink
        latency_monitor.start_timer('network_uplink')
        time.sleep(0.018)  # 18 ms
        network_latency = latency_monitor.end_timer('network_uplink')
        
        # Simulate inference
        latency_monitor.start_timer('cloud_inference')
        time.sleep(0.015)  # 15 ms
        inference_latency = latency_monitor.end_timer('cloud_inference')
        
        # Record end-to-end latency
        total_latency = edge_latency + network_latency + inference_latency
        latency_monitor.record_end_to_end_latency(total_latency)
        
        # Record performance
        system_monitor.performance_monitor.record_inference(inference_latency)
    
    # Check system health
    health_status = system_monitor.check_system_health()
    print("\nSystem Health Status:")
    print(f"Overall: {health_status['overall']}")
    
    # Generate report
    report = system_monitor.get_health_report()
    print("\n" + report)
    
    # Stop monitoring
    system_monitor.stop_monitoring()
    
    # Save report
    system_monitor.save_system_report("./system_reports")