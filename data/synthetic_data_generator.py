"""
Synthetic Trauma Data Generation Engine
Generates physiologically-informed synthetic dataset as described in Section 3.2
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import json
import os

class PhysiologicalModel:
    """Models physiological dynamics for trauma patients"""
    
    def __init__(self):
        # Physiological parameter ranges from literature
        self.param_ranges = {
            'hr': {'normal': (60, 100), 'tachycardia': (100, 140), 'severe': (140, 180)},
            'sbp': {'normal': (100, 140), 'hypotension': (70, 100), 'severe': (40, 70)},
            'dbp': {'normal': (60, 90), 'low': (40, 60), 'severe': (20, 40)},
            'spo2': {'normal': (95, 100), 'mild': (90, 95), 'severe': (80, 90)},
            'resp_rate': {'normal': (12, 20), 'tachypnea': (20, 30), 'severe': (30, 40)},
            'temp': {'normal': (36.5, 37.5), 'fever': (37.5, 39.5), 'hypo': (35, 36.5)}
        }
        
        # Correlation matrix between vital signs (based on clinical knowledge)
        self.correlations = {
            'hr_sbp': -0.65,  # Inverse relationship in shock
            'hr_spo2': -0.45, # Tachycardia with hypoxia
            'sbp_spo2': 0.30, # Hypotension may accompany hypoxia
            'hr_temp': 0.40,  # Fever causes tachycardia
        }
    
    def generate_compensated_shock_trajectory(self, duration_sec: int = 1800) -> Dict:
        """Generate trajectory for compensated shock phase"""
        time_points = np.arange(0, duration_sec, 1)
        
        # Base values
        hr_base = 85
        sbp_base = 125
        spo2_base = 98
        
        # Create trends
        # Early: subtle changes in pulse pressure
        early_phase = time_points[time_points < 600]  # First 10 minutes
        hr_trend = hr_base + 0.02 * early_phase  # Gradual increase
        sbp_trend = sbp_base - 0.015 * early_phase  # Gradual decrease
        
        # Middle: compensation begins to fail
        middle_phase = time_points[(time_points >= 600) & (time_points < 1200)]
        hr_trend = np.concatenate([hr_trend, 95 + 0.04 * (middle_phase - 600)])
        sbp_trend = np.concatenate([sbp_trend, 115 - 0.03 * (middle_phase - 600)])
        
        # Late: decompensation
        late_phase = time_points[time_points >= 1200]
        hr_trend = np.concatenate([hr_trend, 115 + 0.06 * (late_phase - 1200)])
        sbp_trend = np.concatenate([sbp_trend, 100 - 0.05 * (late_phase - 1200)])
        
        # Add physiological variability
        hr_variability = np.sin(time_points / 30) * 3 + np.random.normal(0, 2, len(time_points))
        sbp_variability = np.cos(time_points / 45) * 2 + np.random.normal(0, 3, len(time_points))
        
        # Combine trends with variability
        hr = hr_trend + hr_variability
        sbp = sbp_trend + sbp_variability
        dbp = sbp * 0.65 + np.random.normal(0, 4, len(time_points))
        
        # SpO2: starts normal, then declines
        spo2 = spo2_base * np.ones_like(time_points)
        spo2[time_points >= 900] -= 0.008 * (time_points[time_points >= 900] - 900)  # Decline after 15 min
        spo2 += np.random.normal(0, 0.5, len(time_points))
        spo2 = np.clip(spo2, 85, 100)
        
        # Calculate derived parameters
        pulse_pressure = sbp - dbp
        shock_index = hr / (sbp + 1e-6)  # Avoid division by zero
        
        return {
            'time': time_points,
            'hr': hr,
            'sbp': sbp,
            'dbp': dbp,
            'spo2': spo2,
            'pulse_pressure': pulse_pressure,
            'shock_index': shock_index,
            'phase': 'compensated_shock'
        }
    
    def generate_stable_trajectory(self, duration_sec: int = 1800) -> Dict:
        """Generate trajectory for stable patient"""
        time_points = np.arange(0, duration_sec, 1)
        
        # Normal ranges with natural variability
        hr = 75 + 8 * np.sin(time_points / 60) + np.random.normal(0, 3, len(time_points))
        sbp = 120 + 5 * np.cos(time_points / 90) + np.random.normal(0, 4, len(time_points))
        dbp = 80 + 4 * np.sin(time_points / 75) + np.random.normal(0, 3, len(time_points))
        spo2 = 98 + 0.5 * np.sin(time_points / 120) + np.random.normal(0, 0.3, len(time_points))
        spo2 = np.clip(spo2, 96, 100)
        
        # Calculate derived parameters
        pulse_pressure = sbp - dbp
        shock_index = hr / (sbp + 1e-6)
        
        return {
            'time': time_points,
            'hr': hr,
            'sbp': sbp,
            'dbp': dbp,
            'spo2': spo2,
            'pulse_pressure': pulse_pressure,
            'shock_index': shock_index,
            'phase': 'stable'
        }

class SyntheticTraumaDataset:
    """Main dataset generator"""
    
    def __init__(self, num_samples: int = 1240, seq_length: int = 30, 
                 num_features: int = 4, random_seed: int = 42):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.random_seed = random_seed
        self.physio_model = PhysiologicalModel()
        
        np.random.seed(random_seed)
        
        # Demographic distributions (simulated)
        self.demographics = {
            'age': {'mean': 45, 'std': 15, 'min': 18, 'max': 85},
            'gender': {'male': 0.65, 'female': 0.35},
            'injury_severity': {'mild': 0.3, 'moderate': 0.4, 'severe': 0.3}
        }
    
    def generate_patient_metadata(self, patient_id: str, is_tic: bool) -> Dict:
        """Generate realistic patient metadata"""
        age = np.random.normal(self.demographics['age']['mean'], 
                              self.demographics['age']['std'])
        age = np.clip(age, self.demographics['age']['min'], 
                     self.demographics['age']['max'])
        
        gender = 'male' if np.random.random() < self.demographics['gender']['male'] else 'female'
        
        # Adjust distributions based on TIC status
        if is_tic:
            injury_severity = np.random.choice(
                ['mild', 'moderate', 'severe'],
                p=[0.1, 0.3, 0.6]  # Higher probability of severe injury for TIC
            )
            initial_hr = np.random.normal(96, 15)
            initial_sbp = np.random.normal(118, 18)
            initial_spo2 = np.random.normal(96, 2.1)
        else:
            injury_severity = np.random.choice(
                ['mild', 'moderate', 'severe'],
                p=self.demographics['injury_severity'].values()
            )
            initial_hr = np.random.normal(88, 12)
            initial_sbp = np.random.normal(125, 14)
            initial_spo2 = np.random.normal(98, 1.5)
        
        shock_index = initial_hr / (initial_sbp + 1e-6)
        
        return {
            'patient_id': patient_id,
            'age': float(age),
            'gender': gender,
            'injury_severity': injury_severity,
            'tic_status': bool(is_tic),
            'initial_vitals': {
                'heart_rate': float(initial_hr),
                'systolic_bp': float(initial_sbp),
                'spo2': float(initial_spo2),
                'shock_index': float(shock_index)
            }
        }
    
    def generate_trajectory(self, metadata: Dict, duration_min: int = 30) -> np.ndarray:
        """Generate full patient trajectory"""
        duration_sec = duration_min * 60
        is_tic = metadata['tic_status']
        
        if is_tic:
            # Generate TIC-positive trajectory with decompensation
            trajectory = self.physio_model.generate_compensated_shock_trajectory(duration_sec)
            
            # Ensure it meets TIC criteria
            # Add more variability and ensure downward trend
            min_hr = np.min(trajectory['hr'][-300:])  # Last 5 minutes
            max_hr = np.max(trajectory['hr'][-300:])
            
            # Adjust if not showing clear deterioration
            if max_hr - min_hr < 20:
                trajectory['hr'][-300:] *= 1.1
            
        else:
            # Generate stable trajectory
            trajectory = self.physio_model.generate_stable_trajectory(duration_sec)
        
        # Extract the features we need
        features = np.column_stack([
            trajectory['hr'],
            trajectory['sbp'],
            trajectory['dbp'],
            trajectory['spo2']
        ])
        
        # Resample to 1 Hz if needed
        if features.shape[0] > duration_sec:
            indices = np.linspace(0, features.shape[0]-1, duration_sec, dtype=int)
            features = features[indices]
        
        return features
    
    def add_noise_and_artifacts(self, data: np.ndarray, 
                               noise_level: float = 0.05,
                               missing_prob: float = 0.02) -> np.ndarray:
        """Add realistic noise and missing data artifacts"""
        noisy_data = data.copy()
        n_timesteps, n_features = data.shape
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * np.std(data, axis=0), data.shape)
        noisy_data += noise
        
        # Simulate missing data (sensor dropouts)
        missing_mask = np.random.random(data.shape) < missing_prob
        noisy_data[missing_mask] = np.nan
        
        # Simulate motion artifacts (sudden spikes)
        artifact_mask = np.random.random(data.shape) < 0.01
        noisy_data[artifact_mask] += np.random.normal(0, 10, np.sum(artifact_mask))
        
        return noisy_data
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete dataset"""
        print(f"Generating synthetic trauma dataset (N={self.num_samples})...")
        
        X_list = []
        y_list = []
        metadata_list = []
        
        # Ensure balanced dataset (50% TIC positive)
        n_tic = self.num_samples // 2
        n_control = self.num_samples - n_tic
        
        patient_counter = 0
        
        # Generate TIC-positive cases
        for i in range(n_tic):
            patient_id = f"PT_TIC_{i+1:04d}"
            metadata = self.generate_patient_metadata(patient_id, is_tic=True)
            trajectory = self.generate_trajectory(metadata)
            
            # Add noise and artifacts
            trajectory = self.add_noise_and_artifacts(trajectory)
            
            # Split into 30-second windows (as in paper)
            n_windows = trajectory.shape[0] // self.seq_length
            for j in range(n_windows):
                window = trajectory[j*self.seq_length:(j+1)*self.seq_length]
                
                # Handle missing values (simple interpolation)
                if np.any(np.isnan(window)):
                    # Linear interpolation for missing values
                    for k in range(window.shape[1]):
                        col = window[:, k]
                        if np.any(np.isnan(col)):
                            mask = np.isnan(col)
                            col[mask] = np.interp(
                                np.where(mask)[0],
                                np.where(~mask)[0],
                                col[~mask]
                            )
                            window[:, k] = col
                
                X_list.append(window)
                y_list.append(1.0)  # TIC positive
                metadata_list.append({
                    **metadata,
                    'window_id': j,
                    'window_start': j * self.seq_length,
                    'window_end': (j + 1) * self.seq_length
                })
            
            patient_counter += 1
            if patient_counter % 50 == 0:
                print(f"  Generated {patient_counter}/{self.num_samples} patients")
        
        # Generate control cases
        for i in range(n_control):
            patient_id = f"PT_CTL_{i+1:04d}"
            metadata = self.generate_patient_metadata(patient_id, is_tic=False)
            trajectory = self.generate_trajectory(metadata)
            
            # Add noise (less aggressive for controls)
            trajectory = self.add_noise_and_artifacts(trajectory, noise_level=0.03)
            
            # Split into windows
            n_windows = trajectory.shape[0] // self.seq_length
            for j in range(n_windows):
                window = trajectory[j*self.seq_length:(j+1)*self.seq_length]
                
                # Handle missing values
                if np.any(np.isnan(window)):
                    for k in range(window.shape[1]):
                        col = window[:, k]
                        if np.any(np.isnan(col)):
                            mask = np.isnan(col)
                            col[mask] = np.interp(
                                np.where(mask)[0],
                                np.where(~mask)[0],
                                col[~mask]
                            )
                            window[:, k] = col
                
                X_list.append(window)
                y_list.append(0.0)  # TIC negative
                metadata_list.append({
                    **metadata,
                    'window_id': j,
                    'window_start': j * self.seq_length,
                    'window_end': (j + 1) * self.seq_length
                })
            
            patient_counter += 1
            if patient_counter % 50 == 0:
                print(f"  Generated {patient_counter}/{self.num_samples} patients")
        
        # Convert to arrays
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        metadata_list = [metadata_list[i] for i in indices]
        
        print(f"\nDataset generation complete!")
        print(f"  Total windows: {len(X)}")
        print(f"  TIC positive: {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
        print(f"  TIC negative: {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")
        print(f"  Shape: {X.shape}")
        
        return X, y, metadata_list
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, 
                    metadata: List[Dict], output_dir: str):
        """Save dataset to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numpy arrays
        np.save(os.path.join(output_dir, 'X_synthetic.npy'), X)
        np.save(os.path.join(output_dir, 'y_synthetic.npy'), y)
        
        # Save metadata
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save summary statistics
        summary = {
            'num_samples': len(X),
            'num_features': X.shape[2],
            'sequence_length': X.shape[1],
            'class_distribution': {
                'tic_positive': int(np.sum(y == 1)),
                'tic_negative': int(np.sum(y == 0)),
                'tic_percentage': float(np.mean(y == 1))
            },
            'feature_statistics': {
                'heart_rate': {
                    'mean': float(np.nanmean(X[:, :, 0])),
                    'std': float(np.nanstd(X[:, :, 0])),
                    'min': float(np.nanmin(X[:, :, 0])),
                    'max': float(np.nanmax(X[:, :, 0]))
                },
                'systolic_bp': {
                    'mean': float(np.nanmean(X[:, :, 1])),
                    'std': float(np.nanstd(X[:, :, 1])),
                    'min': float(np.nanmin(X[:, :, 1])),
                    'max': float(np.nanmax(X[:, :, 1]))
                },
                'diastolic_bp': {
                    'mean': float(np.nanmean(X[:, :, 2])),
                    'std': float(np.nanstd(X[:, :, 2])),
                    'min': float(np.nanmin(X[:, :, 2])),
                    'max': float(np.nanmax(X[:, :, 2]))
                },
                'spo2': {
                    'mean': float(np.nanmean(X[:, :, 3])),
                    'std': float(np.nanstd(X[:, :, 3])),
                    'min': float(np.nanmin(X[:, :, 3])),
                    'max': float(np.nanmax(X[:, :, 3]))
                }
            },
            'generation_parameters': {
                'num_samples': self.num_samples,
                'seq_length': self.seq_length,
                'num_features': self.num_features,
                'random_seed': self.random_seed
            }
        }
        
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        
        # Generate validation plots
        self.generate_validation_plots(X, y, output_dir)
    
    def generate_validation_plots(self, X: np.ndarray, y: np.ndarray, 
                                output_dir: str):
        """Generate validation plots similar to Figure 2 in paper"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Distribution comparison
        tic_mask = y == 1
        control_mask = y == 0
        
        # Heart rate distribution
        hr_tic = X[tic_mask, :, 0].flatten()
        hr_control = X[control_mask, :, 0].flatten()
        
        axes[0, 0].hist(hr_control, bins=50, alpha=0.5, label='Control', density=True)
        axes[0, 0].hist(hr_tic, bins=50, alpha=0.5, label='TIC', density=True)
        axes[0, 0].set_xlabel('Heart Rate (bpm)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Heart Rate Distribution')
        axes[0, 0].legend()
        
        # 2. Example trajectories
        tic_example_idx = np.where(tic_mask)[0][0]
        control_example_idx = np.where(control_mask)[0][0]
        
        time = np.arange(self.seq_length)
        axes[0, 1].plot(time, X[tic_example_idx, :, 0], 'r-', label='TIC HR', linewidth=2)
        axes[0, 1].plot(time, X[control_example_idx, :, 0], 'b-', label='Control HR', linewidth=2)
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Heart Rate (bpm)')
        axes[0, 1].set_title('Example Heart Rate Trajectories')
        axes[0, 1].legend()
        
        # 3. Correlation heatmap
        import seaborn as sns
        from scipy.stats import pearsonr
        
        # Compute correlation matrix
        corr_matrix = np.zeros((4, 4))
        feature_names = ['HR', 'SBP', 'DBP', 'SpO2']
        
        for i in range(4):
            for j in range(4):
                corr, _ = pearsonr(X[:, :, i].flatten(), X[:, :, j].flatten())
                corr_matrix[i, j] = corr
        
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(4))
        axes[1, 0].set_yticks(range(4))
        axes[1, 0].set_xticklabels(feature_names)
        axes[1, 0].set_yticklabels(feature_names)
        axes[1, 0].set_title('Correlation Matrix')
        
        # Add correlation values
        for i in range(4):
            for j in range(4):
                text = axes[1, 0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="black")
        
        # 4. Shock Index comparison
        shock_index_tic = X[tic_mask, :, 0].flatten() / (X[tic_mask, :, 1].flatten() + 1e-6)
        shock_index_control = X[control_mask, :, 0].flatten() / (X[control_mask, :, 1].flatten() + 1e-6)
        
        # Take samples for visualization
        n_samples = min(1000, len(shock_index_tic), len(shock_index_control))
        shock_index_tic_sample = np.random.choice(shock_index_tic, n_samples, replace=False)
        shock_index_control_sample = np.random.choice(shock_index_control, n_samples, replace=False)
        
        axes[1, 1].boxplot([shock_index_control_sample, shock_index_tic_sample], 
                          labels=['Control', 'TIC'])
        axes[1, 1].set_ylabel('Shock Index (HR/SBP)')
        axes[1, 1].set_title('Shock Index Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_validation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Validation plots saved to {output_dir}/data_validation_plots.png")

def main():
    """Command-line interface for data generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic trauma dataset")
    parser.add_argument("--num_samples", type=int, default=1240,
                       help="Number of patient samples to generate")
    parser.add_argument("--seq_length", type=int, default=30,
                       help="Sequence length in seconds")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic",
                       help="Output directory for generated data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticTraumaDataset(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        num_features=4,
        random_seed=args.seed
    )
    
    # Generate dataset
    X, y, metadata = generator.generate()
    
    # Save dataset
    generator.save_dataset(X, y, metadata, args.output_dir)
    
    print(f"\nDataset successfully generated and saved to {args.output_dir}")
    print(f"To use this dataset in training, run:")
    print(f"  python train.py --data_dir {args.output_dir}")

if __name__ == "__main__":
    main()