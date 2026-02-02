"""
Shock Index Calculator
Rule-based baseline as described in Section 3.4
"""

import numpy as np
from typing import Union, Dict, List, Optional

class ShockIndexCalculator:
    """
    Shock Index (SI) calculator for TIC prediction
    
    Shock Index = Heart Rate / Systolic Blood Pressure
    Threshold: SI > 1.0 indicates potential shock (as per literature)
    """
    
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        
    def calculate_si(self, heart_rate: Union[float, np.ndarray], 
                    systolic_bp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Shock Index
        
        Args:
            heart_rate: Heart rate in bpm
            systolic_bp: Systolic blood pressure in mmHg
        Returns:
            Shock Index value(s)
        """
        # Avoid division by zero
        systolic_bp = np.maximum(systolic_bp, 1.0)
        return heart_rate / systolic_bp
    
    def predict_tic(self, heart_rate: Union[float, np.ndarray], 
                   systolic_bp: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Predict TIC based on Shock Index threshold
        
        Args:
            heart_rate: Heart rate in bpm
            systolic_bp: Systolic blood pressure in mmHg
        Returns:
            Boolean prediction(s)
        """
        si = self.calculate_si(heart_rate, systolic_bp)
        return si > self.threshold
    
    def predict_tic_probability(self, heart_rate: Union[float, np.ndarray], 
                               systolic_bp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert binary prediction to probability-like score
        
        Args:
            heart_rate: Heart rate in bpm
            systolic_bp: Systolic blood pressure in mmHg
        Returns:
            Probability score between 0 and 1
        """
        si = self.calculate_si(heart_rate, systolic_bp)
        
        # Sigmoid-like transformation around threshold
        # This gives a smooth probability instead of binary prediction
        probability = 1 / (1 + np.exp(-10 * (si - self.threshold)))
        return probability
    
    def analyze_trajectory(self, heart_rate_traj: np.ndarray, 
                          systolic_bp_traj: np.ndarray) -> Dict:
        """
        Analyze a time-series trajectory using Shock Index
        
        Args:
            heart_rate_traj: Time-series of heart rates
            systolic_bp_traj: Time-series of systolic BP
        Returns:
            Dictionary with analysis results
        """
        # Calculate SI trajectory
        si_traj = self.calculate_si(heart_rate_traj, systolic_bp_traj)
        
        # Binary predictions
        predictions = self.predict_tic(heart_rate_traj, systolic_bp_traj)
        
        # Probability scores
        probabilities = self.predict_tic_probability(heart_rate_traj, systolic_bp_traj)
        
        # Statistics
        mean_si = np.mean(si_traj)
        max_si = np.max(si_traj)
        time_above_threshold = np.sum(predictions) / len(predictions) * 100
        
        # Early warning: first time SI crosses threshold
        crossing_indices = np.where(predictions)[0]
        if len(crossing_indices) > 0:
            first_crossing = crossing_indices[0]
            early_warning_time = len(si_traj) - first_crossing
        else:
            first_crossing = -1
            early_warning_time = 0
        
        return {
            'shock_index_trajectory': si_traj,
            'binary_predictions': predictions,
            'probability_scores': probabilities,
            'statistics': {
                'mean_shock_index': float(mean_si),
                'max_shock_index': float(max_si),
                'percent_time_above_threshold': float(time_above_threshold),
                'first_crossing_index': int(first_crossing),
                'early_warning_time_steps': int(early_warning_time)
            }
        }

class ModifiedShockIndex(ShockIndexCalculator):
    """
    Modified Shock Index with age adjustment
    
    Age-adjusted SI = HR / (SBP * age_factor)
    Age factor: 1.0 for age < 65, 0.9 for age >= 65
    """
    
    def __init__(self, threshold: float = 1.0):
        super().__init__(threshold)
        
    def calculate_age_adjusted_si(self, heart_rate: float, 
                                 systolic_bp: float, age: float) -> float:
        """
        Calculate age-adjusted Shock Index
        
        Args:
            heart_rate: Heart rate in bpm
            systolic_bp: Systolic blood pressure in mmHg
            age: Patient age in years
        Returns:
            Age-adjusted Shock Index
        """
        age_factor = 0.9 if age >= 65 else 1.0
        adjusted_sbp = systolic_bp * age_factor
        return self.calculate_si(heart_rate, adjusted_sbp)
    
    def predict_tic_with_age(self, heart_rate: float, systolic_bp: float, 
                            age: float) -> bool:
        """
        Predict TIC with age adjustment
        """
        adjusted_si = self.calculate_age_adjusted_si(heart_rate, systolic_bp, age)
        return adjusted_si > self.threshold

class DynamicShockIndex(ShockIndexCalculator):
    """
    Dynamic Shock Index that considers rate of change
    
    Dynamic SI = SI + α * d(SI)/dt
    where α is a weighting factor for the derivative
    """
    
    def __init__(self, threshold: float = 1.0, alpha: float = 0.5):
        super().__init__(threshold)
        self.alpha = alpha  # Weight for derivative term
    
    def calculate_dynamic_si(self, heart_rate_traj: np.ndarray, 
                            systolic_bp_traj: np.ndarray) -> np.ndarray:
        """
        Calculate dynamic Shock Index
        
        Args:
            heart_rate_traj: Time-series of heart rates
            systolic_bp_traj: Time-series of systolic BP
        Returns:
            Dynamic Shock Index trajectory
        """
        # Calculate standard SI
        si_traj = self.calculate_si(heart_rate_traj, systolic_bp_traj)
        
        # Calculate derivative (rate of change)
        # Use central difference for interior points
        derivative = np.zeros_like(si_traj)
        if len(si_traj) > 2:
            derivative[1:-1] = (si_traj[2:] - si_traj[:-2]) / 2
            # Forward difference for first point
            derivative[0] = si_traj[1] - si_traj[0]
            # Backward difference for last point
            derivative[-1] = si_traj[-1] - si_traj[-2]
        
        # Combine SI with its derivative
        dynamic_si = si_traj + self.alpha * derivative
        
        return dynamic_si
    
    def analyze_dynamic_trajectory(self, heart_rate_traj: np.ndarray, 
                                  systolic_bp_traj: np.ndarray) -> Dict:
        """
        Analyze trajectory using dynamic Shock Index
        """
        # Calculate dynamic SI
        dynamic_si = self.calculate_dynamic_si(heart_rate_traj, systolic_bp_traj)
        
        # Standard SI for comparison
        standard_si = self.calculate_si(heart_rate_traj, systolic_bp_traj)
        
        # Predictions
        dynamic_predictions = dynamic_si > self.threshold
        standard_predictions = standard_si > self.threshold
        
        # Compare timing of alerts
        dynamic_crossings = np.where(dynamic_predictions)[0]
        standard_crossings = np.where(standard_predictions)[0]
        
        if len(dynamic_crossings) > 0 and len(standard_crossings) > 0:
            early_warning_advantage = standard_crossings[0] - dynamic_crossings[0]
        else:
            early_warning_advantage = 0
        
        return {
            'standard_shock_index': standard_si,
            'dynamic_shock_index': dynamic_si,
            'standard_predictions': standard_predictions,
            'dynamic_predictions': dynamic_predictions,
            'early_warning_advantage_steps': int(early_warning_advantage),
            'alert_timing': {
                'first_standard_alert': int(standard_crossings[0]) if len(standard_crossings) > 0 else -1,
                'first_dynamic_alert': int(dynamic_crossings[0]) if len(dynamic_crossings) > 0 else -1
            }
        }

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_timesteps = 30
    heart_rate = np.random.normal(100, 15, n_timesteps)
    systolic_bp = np.random.normal(110, 20, n_timesteps)
    
    # Standard Shock Index
    si_calculator = ShockIndexCalculator(threshold=1.0)
    results = si_calculator.analyze_trajectory(heart_rate, systolic_bp)
    
    print("Standard Shock Index Analysis")
    print("="*50)
    print(f"Mean SI: {results['statistics']['mean_shock_index']:.3f}")
    print(f"Max SI: {results['statistics']['max_shock_index']:.3f}")
    print(f"Time above threshold: {results['statistics']['percent_time_above_threshold']:.1f}%")
    print(f"Early warning time: {results['statistics']['early_warning_time_steps']} steps")
    
    # Dynamic Shock Index
    dynamic_si = DynamicShockIndex(threshold=1.0, alpha=0.5)
    dynamic_results = dynamic_si.analyze_dynamic_trajectory(heart_rate, systolic_bp)
    
    print("\nDynamic Shock Index Analysis")
    print("="*50)
    print(f"Early warning advantage: {dynamic_results['early_warning_advantage_steps']} steps")
    
    # Compare with paper results
    print("\nComparison with Paper Results (Table 2)")
    print("="*50)
    print("Shock Index performance in paper:")
    print("  AUROC: 0.785")
    print("  Early Warning Time: 5.2 ± 2.8 min")
    print("  Computational Latency: < 1 ms")