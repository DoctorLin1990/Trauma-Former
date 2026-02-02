"""
Data Preprocessing Module for Trauma-Former
Preprocessing utilities as described in Section 3.2.2 of the paper
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import warnings

class VitalSignsPreprocessor:
    """Preprocesses vital sign data for Trauma-Former"""
    
    def __init__(self, sampling_rate: float = 1.0,  # Hz
                 window_size: int = 30,  # seconds
                 stride: int = 1,  # seconds
                 features: List[str] = None):
        
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.stride = stride
        self.features = features or ['HR', 'SBP', 'DBP', 'SpO2']
        self.num_features = len(self.features)
        
        # Initialize scalers (fit on training data)
        self.scalers = {}
        self.fitted = False
    
    def sliding_window_segmentation(self, data: np.ndarray, 
                                   labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment continuous time-series into sliding windows
        as described in Section 3.2.2
        
        Args:
            data: [total_time_points, num_features]
            labels: [total_time_points] or None
        Returns:
            windows: [num_windows, window_size, num_features]
            window_labels: [num_windows]
        """
        total_points = data.shape[0]
        
        # Calculate number of windows
        num_windows = (total_points - self.window_size) // self.stride + 1
        
        windows = []
        window_labels = []
        
        for i in range(0, num_windows * self.stride, self.stride):
            end_idx = i + self.window_size
            if end_idx > total_points:
                break
                
            window = data[i:end_idx]
            
            # Check for NaN values
            if np.any(np.isnan(window)):
                window = self.handle_missing_values(window)
            
            windows.append(window)
            
            if labels is not None:
                # Label for window is the label at the end of the window
                window_labels.append(labels[end_idx - 1])
        
        windows = np.array(windows, dtype=np.float32)
        
        if labels is not None:
            window_labels = np.array(window_labels, dtype=np.float32)
            return windows, window_labels
        else:
            return windows, None
    
    def handle_missing_values(self, window: np.ndarray, 
                             method: str = 'linear') -> np.ndarray:
        """
        Handle missing values in a time window
        Options: 'linear', 'cubic', 'nearest', 'mean'
        """
        cleaned = window.copy()
        
        for feat_idx in range(window.shape[1]):
            column = window[:, feat_idx]
            
            if np.any(np.isnan(column)):
                nan_mask = np.isnan(column)
                valid_indices = np.where(~nan_mask)[0]
                valid_values = column[valid_indices]
                
                if len(valid_values) == 0:
                    # If all values are NaN, fill with column mean or 0
                    cleaned[:, feat_idx] = 0
                elif len(valid_values) == 1:
                    # If only one valid value, fill all with that value
                    cleaned[:, feat_idx] = valid_values[0]
                else:
                    # Interpolate missing values
                    if method == 'linear':
                        # Linear interpolation
                        cleaned[:, feat_idx] = np.interp(
                            np.arange(len(column)),
                            valid_indices,
                            valid_values
                        )
                    elif method == 'cubic':
                        # Cubic spline interpolation
                        interp_func = interp1d(valid_indices, valid_values, 
                                              kind='cubic', 
                                              bounds_error=False,
                                              fill_value='extrapolate')
                        cleaned[:, feat_idx] = interp_func(np.arange(len(column)))
                    elif method == 'mean':
                        # Fill with mean of valid values
                        mean_val = np.mean(valid_values)
                        cleaned[nan_mask, feat_idx] = mean_val
                    elif method == 'nearest':
                        # Nearest neighbor interpolation
                        interp_func = interp1d(valid_indices, valid_values,
                                              kind='nearest',
                                              bounds_error=False,
                                              fill_value='extrapolate')
                        cleaned[:, feat_idx] = interp_func(np.arange(len(column)))
        
        return cleaned
    
    def filter_noise(self, data: np.ndarray, 
                    filter_type: str = 'butterworth',
                    cutoff_freq: float = 0.5,
                    order: int = 3) -> np.ndarray:
        """
        Apply noise filtering to vital sign data
        """
        filtered_data = data.copy()
        
        # Normalize cutoff frequency (Nyquist frequency = sampling_rate/2)
        nyquist = self.sampling_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        
        if filter_type == 'butterworth':
            # Butterworth low-pass filter
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            
            for i in range(data.shape[1]):
                # Forward-backward filtering to avoid phase shift
                filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        elif filter_type == 'median':
            # Median filter (good for spike noise)
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.medfilt(data[:, i], kernel_size=5)
        
        elif filter_type == 'savgol':
            # Savitzky-Golay filter (preserves features)
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.savgol_filter(
                    data[:, i], 
                    window_length=5, 
                    polyorder=2
                )
        
        return filtered_data
    
    def extract_features(self, window: np.ndarray) -> Dict:
        """
        Extract additional features from vital sign window
        """
        features = {}
        
        # Basic statistics
        for i, feat_name in enumerate(self.features):
            col = window[:, i]
            features[f'{feat_name}_mean'] = np.mean(col)
            features[f'{feat_name}_std'] = np.std(col)
            features[f'{feat_name}_min'] = np.min(col)
            features[f'{feat_name}_max'] = np.max(col)
            features[f'{feat_name}_median'] = np.median(col)
        
        # Derived features
        if 'HR' in self.features and 'SBP' in self.features:
            hr_idx = self.features.index('HR')
            sbp_idx = self.features.index('SBP')
            
            hr = window[:, hr_idx]
            sbp = window[:, sbp_idx]
            
            # Shock Index
            si = hr / (sbp + 1e-6)
            features['shock_index_mean'] = np.mean(si)
            features['shock_index_max'] = np.max(si)
            
            # Pulse Pressure (if DBP available)
            if 'DBP' in self.features:
                dbp_idx = self.features.index('DBP')
                dbp = window[:, dbp_idx]
                pulse_pressure = sbp - dbp
                features['pulse_pressure_mean'] = np.mean(pulse_pressure)
                features['pulse_pressure_trend'] = self.calculate_trend(pulse_pressure)
        
        # Rate of change features
        for i, feat_name in enumerate(self.features):
            col = window[:, i]
            if len(col) > 1:
                diff = np.diff(col)
                features[f'{feat_name}_roc_mean'] = np.mean(diff)
                features[f'{feat_name}_roc_std'] = np.std(diff)
                features[f'{feat_name}_roc_max'] = np.max(np.abs(diff))
        
        # Frequency domain features (for HR)
        if 'HR' in self.features:
            hr_idx = self.features.index('HR')
            hr = window[:, hr_idx]
            
            # Simple HR variability
            if len(hr) > 10:
                features['hr_variability'] = np.std(hr)
                
                # Approximate RMSSD (Root Mean Square of Successive Differences)
                diff_hr = np.diff(hr)
                features['hr_rmssd'] = np.sqrt(np.mean(diff_hr ** 2))
        
        return features
    
    def calculate_trend(self, data: np.ndarray) -> float:
        """Calculate linear trend coefficient"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        return slope
    
    def z_score_normalization(self, data: np.ndarray, 
                             fit: bool = False) -> np.ndarray:
        """
        Z-score normalization as described in Section 3.2.2
        """
        normalized = data.copy()
        
        if fit:
            # Fit new scalers
            self.scalers = {}
            for i in range(data.shape[1]):
                scaler = StandardScaler()
                # Reshape for sklearn
                col = data[:, i].reshape(-1, 1)
                scaler.fit(col)
                self.scalers[i] = scaler
            self.fitted = True
        
        # Apply normalization
        if self.fitted:
            for i in range(data.shape[1]):
                if i in self.scalers:
                    scaler = self.scalers[i]
                    col = data[:, i].reshape(-1, 1)
                    normalized[:, i] = scaler.transform(col).flatten()
        else:
            warnings.warn("Scalers not fitted. Using per-window normalization.")
            # Per-window normalization (for inference)
            for i in range(data.shape[1]):
                col = data[:, i]
                normalized[:, i] = (col - np.mean(col)) / (np.std(col) + 1e-8)
        
        return normalized
    
    def min_max_normalization(self, data: np.ndarray, 
                             feature_ranges: Optional[Dict] = None) -> np.ndarray:
        """Min-max normalization to [0, 1] range"""
        normalized = data.copy()
        
        if feature_ranges is None:
            # Use per-window normalization
            for i in range(data.shape[1]):
                col = data[:, i]
                min_val = np.min(col)
                max_val = np.max(col)
                if max_val > min_val:
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
                else:
                    normalized[:, i] = 0
        else:
            # Use predefined ranges
            for i, feat_name in enumerate(self.features):
                if feat_name in feature_ranges:
                    min_val, max_val = feature_ranges[feat_name]
                    col = data[:, i]
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
                    # Clip to [0, 1]
                    normalized[:, i] = np.clip(normalized[:, i], 0, 1)
        
        return normalized
    
    def create_sequence_dataset(self, data: np.ndarray, 
                               labels: np.ndarray,
                               augmentation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create final dataset ready for training
        """
        # Segment into windows
        windows, window_labels = self.sliding_window_segmentation(data, labels)
        
        if augmentation and window_labels is not None:
            # Data augmentation for minority class
            windows, window_labels = self.augment_data(windows, window_labels)
        
        # Normalize windows
        normalized_windows = np.zeros_like(windows)
        for i in range(len(windows)):
            normalized_windows[i] = self.z_score_normalization(windows[i])
        
        return normalized_windows, window_labels
    
    def augment_data(self, windows: np.ndarray, 
                    labels: np.ndarray,
                    target_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment minority class data
        """
        # Count samples per class
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        if len(class_counts) < 2:
            return windows, labels
        
        # Find minority class
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        
        minority_count = class_counts[minority_class]
        majority_count = class_counts[majority_class]
        
        # Calculate how many samples to generate
        if minority_count == 0:
            return windows, labels
        
        target_count = int(majority_count * target_ratio)
        samples_needed = max(0, target_count - minority_count)
        
        if samples_needed == 0:
            return windows, labels
        
        # Get minority samples
        minority_indices = np.where(labels == minority_class)[0]
        minority_windows = windows[minority_indices]
        
        # Generate augmented samples
        augmented_windows = []
        augmented_labels = []
        
        for _ in range(samples_needed):
            # Randomly select a minority sample
            idx = np.random.randint(0, len(minority_windows))
            sample = minority_windows[idx].copy()
            
            # Apply random augmentation
            aug_sample = self.apply_augmentation(sample)
            augmented_windows.append(aug_sample)
            augmented_labels.append(minority_class)
        
        # Combine with original data
        if len(augmented_windows) > 0:
            augmented_windows = np.array(augmented_windows, dtype=np.float32)
            augmented_labels = np.array(augmented_labels, dtype=np.float32)
            
            windows = np.concatenate([windows, augmented_windows], axis=0)
            labels = np.concatenate([labels, augmented_labels], axis=0)
        
        return windows, labels
    
    def apply_augmentation(self, window: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a window"""
        augmented = window.copy()
        
        # Randomly choose augmentation method
        method = np.random.choice(['noise', 'scaling', 'time_warp', 'combination'])
        
        if method == 'noise':
            # Add Gaussian noise
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, window.shape)
            augmented += noise
            
        elif method == 'scaling':
            # Random scaling
            scale_factor = np.random.uniform(0.9, 1.1, window.shape[1])
            for i in range(window.shape[1]):
                augmented[:, i] = window[:, i] * scale_factor[i]
        
        elif method == 'time_warp':
            # Time warping (speed up/slow down)
            original_length = window.shape[0]
            warp_factor = np.random.uniform(0.8, 1.2)
            new_length = int(original_length * warp_factor)
            
            if new_length != original_length:
                # Resample each feature
                for i in range(window.shape[1]):
                    original_signal = window[:, i]
                    x_original = np.linspace(0, 1, original_length)
                    x_new = np.linspace(0, 1, new_length)
                    
                    # Interpolate to new length
                    interp_func = interp1d(x_original, original_signal, 
                                          kind='linear', 
                                          fill_value='extrapolate')
                    warped_signal = interp_func(x_new)
                    
                    # Resample back to original length
                    if new_length > original_length:
                        # Downsample
                        indices = np.linspace(0, new_length-1, original_length).astype(int)
                        augmented[:, i] = warped_signal[indices]
                    else:
                        # Upsample
                        x_warped = np.linspace(0, 1, new_length)
                        x_target = np.linspace(0, 1, original_length)
                        interp_back = interp1d(x_warped, warped_signal,
                                              kind='linear',
                                              fill_value='extrapolate')
                        augmented[:, i] = interp_back(x_target)
        
        elif method == 'combination':
            # Combine multiple augmentations
            # Add noise
            noise_level = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_level, window.shape)
            augmented += noise
            
            # Add scaling
            scale_factor = np.random.uniform(0.95, 1.05, window.shape[1])
            for i in range(window.shape[1]):
                augmented[:, i] = augmented[:, i] * scale_factor[i]
        
        return augmented
    
    def preprocess_realtime(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Preprocess data for real-time inference
        """
        # Handle missing values
        if np.any(np.isnan(raw_data)):
            raw_data = self.handle_missing_values(raw_data)
        
        # Apply noise filtering
        filtered_data = self.filter_noise(raw_data, 
                                         filter_type='butterworth',
                                         cutoff_freq=0.5)
        
        # Normalize
        if self.fitted:
            normalized_data = self.z_score_normalization(filtered_data)
        else:
            # If scalers not fitted, use per-window normalization
            normalized_data = np.zeros_like(filtered_data)
            for i in range(filtered_data.shape[1]):
                col = filtered_data[:, i]
                normalized_data[:, i] = (col - np.mean(col)) / (np.std(col) + 1e-8)
        
        return normalized_data
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state"""
        import pickle
        
        state = {
            'sampling_rate': self.sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'features': self.features,
            'scalers': self.scalers,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load_preprocessor(cls, filepath: str):
        """Load preprocessor from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            sampling_rate=state['sampling_rate'],
            window_size=state['window_size'],
            stride=state['stride'],
            features=state['features']
        )
        
        preprocessor.scalers = state['scalers']
        preprocessor.fitted = state['fitted']
        
        return preprocessor

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = np.random.randn(300, 4)  # 5 minutes at 1 Hz
    sample_labels = np.random.randint(0, 2, 300)
    
    # Initialize preprocessor
    preprocessor = VitalSignsPreprocessor(
        sampling_rate=1.0,
        window_size=30,
        stride=1,
        features=['HR', 'SBP', 'DBP', 'SpO2']
    )
    
    # Create dataset
    windows, window_labels = preprocessor.create_sequence_dataset(
        sample_data, sample_labels, augmentation=True
    )
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Windowed data shape: {windows.shape}")
    print(f"Window labels shape: {window_labels.shape}")
    
    # Test real-time preprocessing
    realtime_data = np.random.randn(30, 4)
    preprocessed = preprocessor.preprocess_realtime(realtime_data)
    print(f"\nReal-time preprocessing test:")
    print(f"Input shape: {realtime_data.shape}")
    print(f"Output shape: {preprocessed.shape}")