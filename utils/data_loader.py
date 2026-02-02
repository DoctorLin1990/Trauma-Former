"""
Data Loading Utilities for Trauma-Former
Handles data loading, batching, and dataset creation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
import json
import os
import h5py
from pathlib import Path

class TraumaDataset(Dataset):
    """PyTorch Dataset for trauma patient data"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray,
                metadata: Optional[List[Dict]] = None,
                transform=None):
        """
        Initialize dataset
        
        Args:
            data: [n_samples, seq_len, n_features]
            labels: [n_samples]
            metadata: Optional list of metadata dictionaries
            transform: Optional transform to apply
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.metadata = metadata
        self.transform = transform
        
        # Validate shapes
        assert len(self.data) == len(self.labels), \
            "Data and labels must have same length"
        
        if metadata is not None:
            assert len(self.data) == len(metadata), \
                "Data and metadata must have same length"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Get item by index
        
        Returns:
            data, label, metadata
        """
        sample_data = self.data[idx]
        sample_label = self.labels[idx]
        
        if self.transform:
            sample_data = self.transform(sample_data)
        
        if self.metadata is not None:
            sample_metadata = self.metadata[idx]
            return sample_data, sample_label, sample_metadata
        else:
            return sample_data, sample_label
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets"""
        class_counts = np.bincount(self.labels.int().numpy())
        total_samples = len(self)
        
        # Compute weights inversely proportional to class frequencies
        weights = total_samples / (len(class_counts) * class_counts)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return torch.FloatTensor(weights)
    
    def get_class_distribution(self) -> Dict:
        """Get class distribution statistics"""
        unique, counts = np.unique(self.labels.numpy(), return_counts=True)
        
        distribution = {}
        for cls, count in zip(unique, counts):
            distribution[int(cls)] = {
                'count': int(count),
                'percentage': float(count / len(self) * 100)
            }
        
        return distribution
    
    def split_dataset(self, train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     stratify: bool = True,
                     seed: int = 42) -> Dict[str, 'TraumaDataset']:
        """
        Split dataset into train/val/test subsets
        
        Args:
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion
            stratify: Whether to maintain class distribution
            seed: Random seed
        Returns:
            Dictionary with split datasets
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1, got {total_ratio}")
        
        if stratify:
            # Stratified split
            from sklearn.model_selection import train_test_split
            
            # First split: train vs temp (val+test)
            indices = np.arange(len(self))
            
            train_idx, temp_idx = train_test_split(
                indices,
                test_size=(1 - train_ratio),
                stratify=self.labels.numpy(),
                random_state=seed
            )
            
            # Second split: val vs test from temp
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=(1 - val_ratio_adjusted),
                stratify=self.labels[temp_idx].numpy(),
                random_state=seed
            )
        else:
            # Random split
            n_samples = len(self)
            indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed))
            
            train_size = int(train_ratio * n_samples)
            val_size = int(val_ratio * n_samples)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
        
        # Create split datasets
        splits = {}
        
        for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            split_data = self.data[idx]
            split_labels = self.labels[idx]
            
            if self.metadata is not None:
                split_metadata = [self.metadata[i] for i in idx]
            else:
                split_metadata = None
            
            splits[name] = TraumaDataset(
                split_data.numpy(),
                split_labels.numpy(),
                split_metadata,
                self.transform
            )
        
        return splits

class DataLoaderFactory:
    """Factory for creating data loaders with various configurations"""
    
    @staticmethod
    def create_dataloader(dataset: Dataset,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         pin_memory: bool = True,
                         drop_last: bool = False,
                         sampler=None) -> DataLoader:
        """
        Create a DataLoader with specified configuration
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler
        )
    
    @staticmethod
    def create_balanced_dataloader(dataset: TraumaDataset,
                                  batch_size: int = 32,
                                  num_workers: int = 4) -> DataLoader:
        """
        Create a DataLoader with balanced class sampling
        """
        from torch.utils.data import WeightedRandomSampler
        
        # Get class weights
        class_weights = dataset.get_class_weights()
        sample_weights = class_weights[dataset.labels.long()]
        
        # Create sampler
        sampler = WeightedRandomSampler(
            sample_weights,
            len(dataset),
            replacement=True
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    
    @staticmethod
    def create_time_series_dataloader(dataset: Dataset,
                                     batch_size: int = 32,
                                     seq_len: int = 30,
                                     stride: int = 1,
                                     shuffle: bool = True) -> DataLoader:
        """
        Create DataLoader specifically for time-series data
        """
        # Time-series specific settings
        collate_fn = None  # Can add custom collate function if needed
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,  # Fewer workers for time-series data
            pin_memory=True,
            collate_fn=collate_fn
        )

class DataIO:
    """Data input/output utilities"""
    
    @staticmethod
    def load_numpy_data(data_path: str, 
                       labels_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load data from numpy files
        
        Args:
            data_path: Path to .npy file with data
            labels_path: Optional path to .npy file with labels
        Returns:
            data array and optional labels array
        """
        data = np.load(data_path)
        
        if labels_path:
            labels = np.load(labels_path)
            return data, labels
        else:
            return data, None
    
    @staticmethod
    def load_hdf5_data(filepath: str,
                      data_key: str = 'data',
                      labels_key: str = 'labels') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from HDF5 file
        """
        with h5py.File(filepath, 'r') as f:
            data = f[data_key][:]
            labels = f[labels_key][:]
        
        return data, labels
    
    @staticmethod
    def load_csv_data(filepath: str,
                     feature_columns: List[str],
                     label_column: str,
                     time_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from CSV file
        """
        df = pd.read_csv(filepath)
        
        # Extract features and labels
        X = df[feature_columns].values
        y = df[label_column].values
        
        return X, y
    
    @staticmethod
    def save_dataset(data: np.ndarray,
                    labels: np.ndarray,
                    output_dir: str,
                    format: str = 'numpy') -> Dict[str, str]:
        """
        Save dataset to disk
        
        Args:
            data: Data array
            labels: Labels array
            output_dir: Output directory
            format: Save format ('numpy', 'hdf5', 'csv')
        Returns:
            Dictionary with saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        if format == 'numpy':
            data_path = os.path.join(output_dir, 'data.npy')
            labels_path = os.path.join(output_dir, 'labels.npy')
            
            np.save(data_path, data)
            np.save(labels_path, labels)
            
            saved_files['data'] = data_path
            saved_files['labels'] = labels_path
            
        elif format == 'hdf5':
            filepath = os.path.join(output_dir, 'dataset.h5')
            
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data', data=data)
                f.create_dataset('labels', data=labels)
            
            saved_files['hdf5'] = filepath
            
        elif format == 'csv':
            # Combine data and labels
            df_data = pd.DataFrame(data)
            df_labels = pd.DataFrame(labels, columns=['label'])
            df = pd.concat([df_data, df_labels], axis=1)
            
            filepath = os.path.join(output_dir, 'dataset.csv')
            df.to_csv(filepath, index=False)
            
            saved_files['csv'] = filepath
        
        return saved_files
    
    @staticmethod
    def load_synthetic_dataset(data_dir: str = './data/synthetic') -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load synthetic dataset generated by the data generator
        """
        data_path = os.path.join(data_dir, 'X_synthetic.npy')
        labels_path = os.path.join(data_dir, 'y_synthetic.npy')
        metadata_path = os.path.join(data_dir, 'metadata.json')
        
        # Load data and labels
        X = np.load(data_path)
        y = np.load(labels_path)
        
        # Load metadata if exists
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = []
        
        print(f"Loaded dataset: X={X.shape}, y={y.shape}, metadata={len(metadata)}")
        
        return X, y, metadata

class BatchGenerator:
    """Generate batches for training and evaluation"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray,
                 batch_size: int = 32, shuffle: bool = True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)
        self.indices = np.arange(self.n_samples)
        
        if shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_idx >= self.n_samples:
            # Reset for next epoch
            self.current_idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration
        
        # Get batch indices
        end_idx = min(self.current_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[self.current_idx:end_idx]
        
        # Get batch data
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        # Update index
        self.current_idx = end_idx
        
        return batch_data, batch_labels
    
    def __len__(self) -> int:
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def reset(self):
        """Reset the generator"""
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

class StreamingDataLoader:
    """
    Data loader for streaming/real-time data
    Simulates continuous data stream for inference testing
    """
    
    def __init__(self, data_source: Union[str, np.ndarray],
                 window_size: int = 30,
                 stride: int = 1,
                 sampling_rate: float = 1.0):
        """
        Initialize streaming data loader
        
        Args:
            data_source: Path to data file or data array
            window_size: Size of sliding window (seconds)
            stride: Stride between windows (seconds)
            sampling_rate: Data sampling rate (Hz)
        """
        # Load data
        if isinstance(data_source, str):
            if data_source.endswith('.npy'):
                self.data = np.load(data_source)
            elif data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
                self.data = df.values
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        else:
            self.data = data_source
        
        # Configuration
        self.window_size = int(window_size * sampling_rate)
        self.stride = int(stride * sampling_rate)
        self.sampling_rate = sampling_rate
        
        # State
        self.current_pos = 0
        self.total_points = len(self.data)
        
        print(f"Streaming data loader initialized:")
        print(f"  Total points: {self.total_points}")
        print(f"  Window size: {self.window_size} points ({window_size}s)")
        print(f"  Stride: {self.stride} points ({stride}s)")
    
    def get_next_window(self) -> Optional[np.ndarray]:
        """
        Get next window of data
        Returns None when no more data is available
        """
        if self.current_pos + self.window_size > self.total_points:
            return None
        
        # Extract window
        window = self.data[self.current_pos:self.current_pos + self.window_size]
        
        # Update position
        self.current_pos += self.stride
        
        return window
    
    def has_next(self) -> bool:
        """Check if more data is available"""
        return self.current_pos + self.window_size <= self.total_points
    
    def reset(self):
        """Reset to beginning of data"""
        self.current_pos = 0
    
    def get_current_time(self) -> float:
        """Get current time in seconds"""
        return self.current_pos / self.sampling_rate
    
    def get_total_duration(self) -> float:
        """Get total duration of data in seconds"""
        return self.total_points / self.sampling_rate

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    seq_len = 30
    n_features = 4
    
    X = np.random.randn(n_samples, seq_len, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Create dataset
    dataset = TraumaDataset(X, y)
    
    print(f"Dataset created: {len(dataset)} samples")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Create data loader
    dataloader = DataLoaderFactory.create_dataloader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    
    # Test batch generation
    print("\nTesting batch generation:")
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        print(f"Batch {batch_idx}: X={batch_X.shape}, y={batch_y.shape}")
        if batch_idx >= 2:
            break
    
    # Test streaming loader
    print("\nTesting streaming loader:")
    streaming_data = np.random.randn(1000, 4)  # 1000 time points, 4 features
    stream_loader = StreamingDataLoader(streaming_data, window_size=30, stride=1)
    
    window_count = 0
    while stream_loader.has_next():
        window = stream_loader.get_next_window()
        if window is not None:
            window_count += 1
            if window_count <= 3:
                print(f"Window {window_count}: shape={window.shape}")
    
    print(f"Total windows extracted: {window_count}")