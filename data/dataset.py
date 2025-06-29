import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wfdb
import os
from scipy import signal
from typing import Dict, List, Tuple, Optional
import warnings

class MITBIHDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH ECG classification following the paper:
    "A Tiny Matched Filter-Based CNN for Inter-Patient ECG Classification and Arrhythmia Detection at the Edge"
    """
    
    # MIT-BIH to AAMI mapping as per Table 1 in the paper
    MITBIH_TO_AAMI = {
        # Normal Beat (N)
        'N': 'N',    # Normal beat
        'L': 'N',    # Left bundle branch block beat
        'R': 'N',    # Right bundle branch block beat
        'e': 'N',    # Atrial escape beat
        'j': 'N',    # Nodal (junctional) escape beat
        
        # Supraventricular Ectopic Beat (SVEB)
        'A': 'SVEB', # Atrial premature beat
        'a': 'SVEB', # Aberrated atrial premature beat
        'S': 'SVEB', # Premature or ectopic supraventricular beat
        'J': 'SVEB', # Nodal (junctional) premature beat
        
        # Ventricular Ectopic Beat (VEB)
        'V': 'VEB',  # Premature ventricular contraction
        'E': 'VEB',  # Ventricular escape beat
        
        # Fusion Beat (F)
        'F': 'F',    # Fusion of ventricular and normal beat
        
        # Unknown (Q)
        'Q': 'Q',    # Unclassifiable beat
        'f': 'Q',    # Fusion of paced and normal beat
        '/': 'Q',    # Paced beat
    }
    
    AAMI_TO_IDX = {'N': 0, 'SVEB': 1, 'VEB': 2, 'F': 3, 'Q': 4}
    
    # Inter-patient division as per Table 2 in the paper
    DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 
                   201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
    
    DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 
                   213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',  # 'train' (DS1) or 'test' (DS2)
                 num_classes: int = 3,  # 3, 4, or 5 classes
                 use_derivative: bool = True,
                 target_fs: int = 128,
                 segment_length: float = 0.5,  # seconds
                 local_rr_window: int = 80,
                 global_rr_window: int = 400,
                 lead: str = 'MLII',
                 cache_dir: Optional[str] = None):
        """
        Initialize MIT-BIH ECG Dataset
        
        Args:
            data_dir: Path to MIT-BIH database
            split: 'train' for DS1 or 'test' for DS2
            num_classes: Number of classes (3: N,SVEB,VEB; 4: +F; 5: +Q)
            use_derivative: Whether to use first derivative as input
            target_fs: Target sampling frequency (Hz)
            segment_length: Length of ECG segments (seconds)
            local_rr_window: Window size for local RR normalization
            global_rr_window: Window size for global RR normalization
            lead: ECG lead to use ('MLII' or 'V1')
            cache_dir: Directory to cache processed data
        """
        self.data_dir = data_dir
        self.split = split
        self.num_classes = num_classes
        self.use_derivative = use_derivative
        self.target_fs = target_fs
        self.segment_samples = int(segment_length * target_fs)  # 64 samples
        self.local_rr_window = local_rr_window
        self.global_rr_window = global_rr_window
        self.lead = lead
        self.cache_dir = cache_dir
        
        # Select records based on split
        if split == 'train':
            self.records = self.DS1_RECORDS
        elif split == 'test':
            self.records = self.DS2_RECORDS
        else:
            raise ValueError("Split must be 'train' or 'test'")
        
        # Filter classes based on num_classes
        if num_classes == 3:
            self.valid_classes = ['N', 'SVEB', 'VEB']
        elif num_classes == 4:
            self.valid_classes = ['N', 'SVEB', 'VEB', 'F']
        elif num_classes == 5:
            self.valid_classes = ['N', 'SVEB', 'VEB', 'F', 'Q']
        else:
            raise ValueError("num_classes must be 3, 4, or 5")
        
        # Load and process data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess ECG data directly from WFDB files"""
        print("Processing ECG data from WFDB files...")
        self.ecg_segments = []
        self.rr_intervals = []
        self.labels = []
        
        for record_num in self.records:
            #print(f"Processing record {record_num}")
            try:
                record_data = self._process_record(record_num)
                if record_data:
                    segments, rr_features, record_labels = record_data
                    self.ecg_segments.extend(segments)
                    self.rr_intervals.extend(rr_features)
                    self.labels.extend(record_labels)
            except Exception as e:
                warnings.warn(f"Error processing record {record_num}: {e}")
                continue
        
        # Convert to numpy arrays
        self.ecg_segments = np.array(self.ecg_segments, dtype=np.float32)
        self.rr_intervals = np.array(self.rr_intervals, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"Loaded {len(self.ecg_segments)} segments")
        print(f"Class distribution: {np.bincount(self.labels)}")
        
    def _process_record(self, record_num: int) -> Optional[Tuple[List, List, List]]:
        """Process a single MIT-BIH record"""
        try:
            # Load record using the same approach as the notebook
            record_path = os.path.join(self.data_dir, str(record_num))
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            # Get ECG signal (MLII lead)
            if self.lead == 'MLII':
                # Usually channel 0 is MLII
                ecg_signal = record.p_signal[:, 0]
            else:
                # Usually channel 1 is V1
                ecg_signal = record.p_signal[:, 1]
            
            # Resample to target frequency
            if record.fs != self.target_fs:
                ecg_signal = signal.resample(ecg_signal, 
                                           int(len(ecg_signal) * self.target_fs / record.fs))
                # Adjust annotation sample indices
                ann_samples = (annotation.sample * self.target_fs / record.fs).astype(int)
            else:
                ann_samples = annotation.sample
            
            # Filter valid annotations
            valid_mask = np.array([
                ann in self.MITBIH_TO_AAMI and 
                self.MITBIH_TO_AAMI[ann] in self.valid_classes 
                for ann in annotation.symbol
            ])
            
            valid_samples = ann_samples[valid_mask]
            valid_symbols = np.array(annotation.symbol)[valid_mask]
            
            if len(valid_samples) == 0:
                return None
            
            # Extract segments and RR intervals
            segments = []
            rr_features = []
            labels = []
            
            # Compute RR intervals
            rr_intervals = np.diff(valid_samples) / self.target_fs  # in seconds
            
            for i, (sample_idx, symbol) in enumerate(zip(valid_samples, valid_symbols)):
                # Extract ECG segment centered on R-peak
                half_segment = self.segment_samples // 2
                start_idx = sample_idx - half_segment
                end_idx = sample_idx + half_segment
                
                # Skip if segment extends beyond signal boundaries
                if start_idx < 0 or end_idx >= len(ecg_signal):
                    continue
                
                segment = ecg_signal[start_idx:end_idx]
                
                # Edge-pad if necessary (shouldn't happen with proper bounds check)
                if len(segment) < self.segment_samples:
                    pad_width = self.segment_samples - len(segment)
                    segment = np.pad(segment, (0, pad_width), mode='edge')
                
                # Compute first derivative if requested
                if self.use_derivative:
                    segment = np.gradient(segment)
                
                # Extract RR interval features
                rr_feature = self._extract_rr_features(rr_intervals, i)
                
                # Map to AAMI class
                aami_class = self.MITBIH_TO_AAMI[symbol]
                label = self.AAMI_TO_IDX[aami_class]
                
                segments.append(segment)
                rr_features.append(rr_feature)
                labels.append(label)
            
            return segments, rr_features, labels
            
        except Exception as e:
            print(f"Error processing record {record_num}: {e}")
            return None
    
    def _extract_rr_features(self, rr_intervals: np.ndarray, beat_idx: int) -> np.ndarray:
        """Extract normalized RR interval features"""
        if len(rr_intervals) == 0:
            return np.zeros(4, dtype=np.float32)
        
        # Get preceding and following RR intervals
        pre_rr = rr_intervals[beat_idx - 1] if beat_idx > 0 else rr_intervals[0] if len(rr_intervals) > 0 else 0
        post_rr = rr_intervals[beat_idx] if beat_idx < len(rr_intervals) else rr_intervals[-1] if len(rr_intervals) > 0 else 0
        
        # Compute local mean (last local_rr_window intervals)
        start_local = max(0, beat_idx - self.local_rr_window)
        end_local = min(beat_idx, len(rr_intervals))
        if end_local > start_local:
            local_mean = np.mean(rr_intervals[start_local:end_local])
        else:
            local_mean = rr_intervals[0] if len(rr_intervals) > 0 else 1.0
        
        # Compute global mean (last global_rr_window intervals)  
        start_global = max(0, beat_idx - self.global_rr_window)
        end_global = min(beat_idx, len(rr_intervals))
        if end_global > start_global:
            global_mean = np.mean(rr_intervals[start_global:end_global])
        else:
            global_mean = local_mean
        
        # Normalize RR intervals
        pre_rr_norm = pre_rr / local_mean if local_mean > 0 else 0
        post_rr_norm = post_rr / local_mean if local_mean > 0 else 0
        local_norm = local_mean / global_mean if global_mean > 0 else 1
        
        return np.array([pre_rr_norm, post_rr_norm, local_norm, global_mean], dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.ecg_segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ecg_segment: ECG segment tensor of shape (segment_samples,)
            rr_features: RR interval features tensor of shape (4,)
            label: Class label tensor
        """
        ecg_segment = torch.from_numpy(self.ecg_segments[idx]).float()
        rr_features = torch.from_numpy(self.rr_intervals[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Add channel dimension for ECG segment
        ecg_segment = ecg_segment.unsqueeze(0)  # Shape: (1, segment_samples)
        
        return ecg_segment, rr_features, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling class imbalance"""
        class_counts = np.bincount(self.labels, minlength=self.num_classes)
        class_weights = len(self.labels) / (self.num_classes * class_counts + 1e-8)
        return torch.from_numpy(class_weights).float()
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution"""
        class_names = list(self.valid_classes)
        class_counts = np.bincount(self.labels, minlength=self.num_classes)
        return {class_names[i]: class_counts[i] for i in range(len(class_names))}
    
    def get_templates_by_class(self) -> Dict[str, np.ndarray]:
        """Extract matched filter templates by averaging segments per class"""
        templates = {}
        
        for class_idx, class_name in enumerate(self.valid_classes):
            class_mask = self.labels == class_idx
            if np.sum(class_mask) > 0:
                class_segments = self.ecg_segments[class_mask]
                template = np.mean(class_segments, axis=0)  # Average waveform
                templates[class_name] = template
        
        return templates


class ECGDataModule:
    """Data module for ECG classification"""
    
    def __init__(self,
                 data_dir: str,
                 num_classes: int = 3,
                 use_derivative: bool = True,
                 batch_size: int = 512,
                 num_workers: int = 4,
                 cache_dir: Optional[str] = None):
        
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.use_derivative = use_derivative
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        
    def setup(self):
        """Setup train and test datasets"""
        self.train_dataset = MITBIHDataset(
            data_dir=self.data_dir,
            split='train',
            num_classes=self.num_classes,
            use_derivative=self.use_derivative,
            cache_dir=self.cache_dir
        )
        
        self.test_dataset = MITBIHDataset(
            data_dir=self.data_dir,
            split='test',
            num_classes=self.num_classes,
            use_derivative=self.use_derivative,
            cache_dir=self.cache_dir
        )
        
        print("Dataset setup complete!")
        print(f"Train set: {len(self.train_dataset)} samples")
        print(f"Test set: {len(self.test_dataset)} samples")
        print(f"Train class distribution: {self.train_dataset.get_class_distribution()}")
        print(f"Test class distribution: {self.test_dataset.get_class_distribution()}")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights from training dataset"""
        return self.train_dataset.get_class_weights()


# Example usage
if __name__ == "__main__":
    # Example usage with the correct data directory path
    data_dir = "mitbih"  # Path to MIT-BIH data directory
    
    # Create data module
    data_module = ECGDataModule(
        data_dir=data_dir,
        num_classes=3,  # N, SVEB, VEB
        use_derivative=True,
        batch_size=512,
        cache_dir="./cache"
    )
    
    # Setup datasets
    data_module.setup()
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Example: iterate through one batch
    for ecg_segments, rr_features, labels in train_loader:
        print(f"ECG segments shape: {ecg_segments.shape}")  # [batch_size, 1, 64]
        print(f"RR features shape: {rr_features.shape}")    # [batch_size, 4]
        print(f"Labels shape: {labels.shape}")              # [batch_size]
        break