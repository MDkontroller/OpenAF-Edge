#!/usr/bin/env python3
"""
Training Pipeline Compatibility Analyzer

Shows:
1. Exact tensor shapes and dtypes for model compatibility
2. Data distribution analysis for normalization decisions
3. Training pipeline flow verification
4. Normalization recommendations
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Add data directory to path
sys.path.append(str(Path(__file__).parent.parent / "data"))

class TrainingPipelineAnalyzer:
    """Analyze training pipeline compatibility and normalization needs"""
    
    def __init__(self, data_dir="data/mitbih"):
        self.data_dir = data_dir
        self.dataset = None
        self.dataloader = None
        
    def load_data(self):
        """Load dataset and dataloader"""
        print("🔄 Loading dataset for pipeline analysis...")
        
        from dataset import MITBIHDataset
        
        self.dataset = MITBIHDataset(
            data_dir=self.data_dir,
            split='train',
            num_classes=3,
            use_derivative=True,
            cache_dir="data/cache"
        )
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=32, 
            shuffle=True,
            num_workers=0
        )
        
        print(f"✅ Loaded dataset: {len(self.dataset)} samples")
        return self.dataset, self.dataloader
    
    def analyze_tensor_shapes_and_types(self):
        """Analyze exact tensor shapes and data types"""
        if self.dataset is None:
            self.load_data()
        
        print("\n" + "="*60)
        print("🔍 TENSOR SHAPES & TYPES ANALYSIS")
        print("="*60)
        
        # Single sample analysis
        print("\n📋 SINGLE SAMPLE:")
        ecg, rr, label = self.dataset[0]
        
        print(f"ECG tensor:")
        print(f"  Shape: {ecg.shape}")
        print(f"  Dtype: {ecg.dtype}")
        print(f"  Device: {ecg.device}")
        print(f"  Memory: {ecg.element_size() * ecg.nelement()} bytes")
        print(f"  Min: {ecg.min():.4f}, Max: {ecg.max():.4f}")
        print(f"  Mean: {ecg.mean():.4f}, Std: {ecg.std():.4f}")
        
        print(f"\nRR tensor:")
        print(f"  Shape: {rr.shape}")
        print(f"  Dtype: {rr.dtype}")
        print(f"  Values: {rr.numpy()}")
        print(f"  Min: {rr.min():.4f}, Max: {rr.max():.4f}")
        
        print(f"\nLabel tensor:")
        print(f"  Shape: {label.shape}")
        print(f"  Dtype: {label.dtype}")
        print(f"  Value: {label.item()}")
        
        # Batch analysis
        print("\n📦 BATCH ANALYSIS:")
        batch_ecg, batch_rr, batch_labels = next(iter(self.dataloader))
        
        print(f"Batch ECG:")
        print(f"  Shape: {batch_ecg.shape}  <- [batch_size, channels, sequence_length]")
        print(f"  Dtype: {batch_ecg.dtype}")
        print(f"  Memory: {batch_ecg.element_size() * batch_ecg.nelement() / 1024:.1f} KB")
        print(f"  Range: [{batch_ecg.min():.4f}, {batch_ecg.max():.4f}]")
        
        print(f"\nBatch RR:")
        print(f"  Shape: {batch_rr.shape}  <- [batch_size, features]")
        print(f"  Dtype: {batch_rr.dtype}")
        print(f"  Range: [{batch_rr.min():.4f}, {batch_rr.max():.4f}]")
        
        print(f"\nBatch Labels:")
        print(f"  Shape: {batch_labels.shape}  <- [batch_size]")
        print(f"  Dtype: {batch_labels.dtype}")
        print(f"  Unique values: {torch.unique(batch_labels).tolist()}")
        print(f"  Class distribution: {torch.bincount(batch_labels).tolist()}")
        
        return batch_ecg, batch_rr, batch_labels
    
    def analyze_data_distributions(self):
        """Analyze data distributions for normalization decisions"""
        if self.dataset is None:
            self.load_data()
        
        print("\n" + "="*60)
        print("📊 DATA DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Collect statistics from multiple batches
        all_ecg_data = []
        all_rr_data = []
        
        print("🔄 Analyzing data distributions across batches...")
        
        for i, (batch_ecg, batch_rr, batch_labels) in enumerate(self.dataloader):
            all_ecg_data.append(batch_ecg.numpy())
            all_rr_data.append(batch_rr.numpy())
            
            if i >= 10:  # Analyze first 10 batches for speed
                break
        
        # Combine data
        ecg_data = np.concatenate(all_ecg_data, axis=0)  # [samples, channels, sequence]
        rr_data = np.concatenate(all_rr_data, axis=0)    # [samples, features]
        
        print(f"📈 Analyzed {ecg_data.shape[0]} samples")
        
        # ECG distribution analysis
        print(f"\n🫀 ECG SIGNAL DISTRIBUTION:")
        ecg_flat = ecg_data.flatten()
        
        print(f"  Overall statistics:")
        print(f"    Mean: {np.mean(ecg_flat):.6f}")
        print(f"    Std:  {np.std(ecg_flat):.6f}")
        print(f"    Min:  {np.min(ecg_flat):.6f}")
        print(f"    Max:  {np.max(ecg_flat):.6f}")
        print(f"    Median: {np.median(ecg_flat):.6f}")
        
        # Per-sample statistics
        sample_means = np.mean(ecg_data, axis=(1, 2))  # Mean per sample
        sample_stds = np.std(ecg_data, axis=(1, 2))    # Std per sample
        
        print(f"  Per-sample variation:")
        print(f"    Mean range: [{np.min(sample_means):.4f}, {np.max(sample_means):.4f}]")
        print(f"    Std range:  [{np.min(sample_stds):.4f}, {np.max(sample_stds):.4f}]")
        
        # RR distribution analysis
        print(f"\n💓 RR FEATURES DISTRIBUTION:")
        for i in range(rr_data.shape[1]):
            feature_data = rr_data[:, i]
            print(f"  RR Feature {i}:")
            print(f"    Mean: {np.mean(feature_data):.4f}")
            print(f"    Std:  {np.std(feature_data):.4f}")
            print(f"    Range: [{np.min(feature_data):.4f}, {np.max(feature_data):.4f}]")
        
        # Create visualizations
        self._plot_distributions(ecg_data, rr_data, sample_means, sample_stds)
        
        return ecg_data, rr_data
    
    def _plot_distributions(self, ecg_data, rr_data, sample_means, sample_stds):
        """Plot distribution visualizations"""
        print("\n📊 Creating distribution plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # ECG overall distribution
        ecg_flat = ecg_data.flatten()
        axes[0, 0].hist(ecg_flat, bins=100, alpha=0.7, color='blue', density=True)
        axes[0, 0].set_title('ECG Signal Distribution (All Data)', fontweight='bold')
        axes[0, 0].set_xlabel('Amplitude')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].axvline(np.mean(ecg_flat), color='red', linestyle='--', label=f'Mean: {np.mean(ecg_flat):.3f}')
        axes[0, 0].legend()
        
        # Per-sample means
        axes[0, 1].hist(sample_means, bins=50, alpha=0.7, color='green', density=True)
        axes[0, 1].set_title('Per-Sample Mean Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Sample Mean')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].axvline(np.mean(sample_means), color='red', linestyle='--', label=f'Mean: {np.mean(sample_means):.3f}')
        axes[0, 1].legend()
        
        # Per-sample stds
        axes[0, 2].hist(sample_stds, bins=50, alpha=0.7, color='orange', density=True)
        axes[0, 2].set_title('Per-Sample Std Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Sample Std')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].axvline(np.mean(sample_stds), color='red', linestyle='--', label=f'Mean: {np.mean(sample_stds):.3f}')
        axes[0, 2].legend()
        
        # RR feature distributions
        colors = ['red', 'blue', 'green', 'orange']
        for i in range(min(3, rr_data.shape[1])):
            ax = axes[1, i]
            feature_data = rr_data[:, i]
            ax.hist(feature_data, bins=50, alpha=0.7, color=colors[i], density=True)
            ax.set_title(f'RR Feature {i} Distribution', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.axvline(np.mean(feature_data), color='black', linestyle='--', 
                      label=f'Mean: {np.mean(feature_data):.3f}')
            ax.legend()
        
        plt.tight_layout()
        plt.suptitle('DATA DISTRIBUTION ANALYSIS', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def test_model_compatibility(self):
        """Test compatibility with actual model architecture"""
        if self.dataset is None:
            self.load_data()
        
        print("\n" + "="*60)
        print("🧪 MODEL COMPATIBILITY TEST")
        print("="*60)
        
        # Test with actual model architecture (simplified version)
        print("🏗️ Creating test model architecture...")
        
        class TestMatchedFilterCNN(nn.Module):
            def __init__(self, num_classes=3, num_filters=11, kernel_size=64):
                super().__init__()
                # Conv1D layer
                self.conv1d = nn.Conv1d(1, num_filters, kernel_size, padding=0)
                self.bn = nn.BatchNorm1d(num_filters)
                self.activation = nn.Tanh()
                self.global_max_pool = nn.AdaptiveMaxPool1d(1)
                
                # RR processing
                self.rr_fc1 = nn.Linear(4, 32)
                self.rr_fc2 = nn.Linear(32, 16)
                self.rr_fc3 = nn.Linear(16, 8)
                
                # Output
                self.output = nn.Linear(num_filters + 8, num_classes)
                
            def forward(self, ecg, rr):
                # ECG branch
                x = self.conv1d(ecg)
                x = self.bn(x)
                x = self.activation(x)
                x = self.global_max_pool(x)
                x = x.squeeze(-1)
                
                # RR branch
                rr_out = torch.relu(self.rr_fc1(rr))
                rr_out = torch.relu(self.rr_fc2(rr_out))
                rr_out = torch.relu(self.rr_fc3(rr_out))
                
                # Combine
                combined = torch.cat([x, rr_out], dim=1)
                output = self.output(combined)
                
                return output
        
        model = TestMatchedFilterCNN()
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        
        batch_ecg, batch_rr, batch_labels = next(iter(self.dataloader))
        
        print(f"Input shapes:")
        print(f"  ECG: {batch_ecg.shape}")
        print(f"  RR:  {batch_rr.shape}")
        print(f"  Labels: {batch_labels.shape}")
        
        try:
            with torch.no_grad():
                outputs = model(batch_ecg, batch_rr)
            
            print(f"\n✅ Forward pass successful!")
            print(f"Output shape: {outputs.shape}")
            print(f"Expected: [batch_size, num_classes] = [{batch_ecg.shape[0]}, 3]")
            
            # Test loss computation
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, batch_labels)
            
            print(f"\n✅ Loss computation successful!")
            print(f"Loss value: {loss.item():.4f}")
            
            # Test predictions
            predictions = torch.argmax(outputs, dim=1)
            print(f"\nPrediction shapes:")
            print(f"  Predictions: {predictions.shape}")
            print(f"  Labels:      {batch_labels.shape}")
            print(f"  Sample predictions: {predictions[:5].tolist()}")
            print(f"  Sample labels:      {batch_labels[:5].tolist()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model compatibility test failed: {e}")
            return False
    
    def normalization_recommendations(self, ecg_data, rr_data):
        """Provide normalization recommendations"""
        print("\n" + "="*60)
        print("💡 NORMALIZATION RECOMMENDATIONS")
        print("="*60)
        
        ecg_flat = ecg_data.flatten()
        
        print("🫀 ECG SIGNALS:")
        print(f"  Current range: [{np.min(ecg_flat):.4f}, {np.max(ecg_flat):.4f}]")
        print(f"  Current mean: {np.mean(ecg_flat):.6f}")
        print(f"  Current std:  {np.std(ecg_flat):.6f}")
        
        # Check if already well-normalized
        mean_close_to_zero = abs(np.mean(ecg_flat)) < 0.1
        std_close_to_one = 0.5 < np.std(ecg_flat) < 2.0
        
        print(f"\n📊 Current normalization status:")
        print(f"  Mean ≈ 0: {'✅' if mean_close_to_zero else '❌'} (|{np.mean(ecg_flat):.4f}| < 0.1)")
        print(f"  Std ≈ 1:  {'✅' if std_close_to_one else '❌'} ({np.std(ecg_flat):.4f} in [0.5, 2.0])")
        
        if mean_close_to_zero and std_close_to_one:
            print("\n✅ ECG signals appear reasonably normalized already!")
            print("   First derivative + existing preprocessing seems sufficient.")
        else:
            print("\n⚠️  ECG signals could benefit from normalization:")
            
            # Z-score normalization example
            ecg_normalized = (ecg_flat - np.mean(ecg_flat)) / np.std(ecg_flat)
            print(f"   After z-score: mean={np.mean(ecg_normalized):.6f}, std={np.std(ecg_normalized):.6f}")
            
            # Per-sample normalization example
            sample_means = np.mean(ecg_data, axis=(1, 2), keepdims=True)
            sample_stds = np.std(ecg_data, axis=(1, 2), keepdims=True)
            ecg_per_sample_norm = (ecg_data - sample_means) / (sample_stds + 1e-8)
            
            print(f"   Per-sample norm: mean={np.mean(ecg_per_sample_norm):.6f}, std={np.std(ecg_per_sample_norm):.6f}")
        
        print(f"\n💓 RR FEATURES:")
        for i in range(rr_data.shape[1]):
            feature_data = rr_data[:, i]
            print(f"  Feature {i}: mean={np.mean(feature_data):.4f}, std={np.std(feature_data):.4f}")
        
        rr_well_normalized = all(0.1 < np.std(rr_data[:, i]) < 10 for i in range(rr_data.shape[1]))
        
        if rr_well_normalized:
            print("✅ RR features appear reasonably scaled.")
        else:
            print("⚠️  RR features might benefit from scaling.")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        
        if not (mean_close_to_zero and std_close_to_one):
            print("1. 🫀 ECG Normalization Options:")
            print("   a) Z-score normalization: (x - mean) / std")
            print("   b) Per-sample normalization: (x - sample_mean) / sample_std")
            print("   c) Min-max scaling: (x - min) / (max - min)")
            print("   d) Robust scaling: (x - median) / IQR")
            
            print("\n   💡 Recommendation: Try per-sample z-score normalization")
            print("      Reason: ECG amplitude varies between patients")
        
        if not rr_well_normalized:
            print("\n2. 💓 RR Feature Scaling:")
            print("   Consider StandardScaler or MinMaxScaler for RR features")
        
        print("\n3. 🔄 Implementation Strategy:")
        print("   a) Start without additional normalization")
        print("   b) If training is unstable, add per-sample ECG normalization")
        print("   c) Monitor gradient magnitudes and loss convergence")
        print("   d) Use validation performance to decide")
        
        return mean_close_to_zero and std_close_to_one, rr_well_normalized
    
    def run_full_analysis(self):
        """Run complete pipeline analysis"""
        print("🚀 RUNNING COMPLETE TRAINING PIPELINE ANALYSIS")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # 1. Tensor shapes and types
        batch_ecg, batch_rr, batch_labels = self.analyze_tensor_shapes_and_types()
        
        # 2. Data distributions
        ecg_data, rr_data = self.analyze_data_distributions()
        
        # 3. Model compatibility
        model_compatible = self.test_model_compatibility()
        
        # 4. Normalization recommendations
        ecg_normalized, rr_normalized = self.normalization_recommendations(ecg_data, rr_data)
        
        # Final summary
        print("\n" + "="*70)
        print("📋 FINAL PIPELINE COMPATIBILITY SUMMARY")
        print("="*70)
        
        print(f"✅ Dataset loaded: {len(self.dataset):,} samples")
        print(f"✅ Tensor shapes compatible: {batch_ecg.shape} -> CNN")
        print(f"✅ Model forward pass: {'✅ Works' if model_compatible else '❌ Failed'}")
        print(f"✅ ECG normalization: {'✅ Good' if ecg_normalized else '⚠️  Consider'}")
        print(f"✅ RR normalization: {'✅ Good' if rr_normalized else '⚠️  Consider'}")
        
        if model_compatible and ecg_normalized and rr_normalized:
            print(f"\n🎉 TRAINING PIPELINE READY!")
            print(f"   Your data is compatible and well-normalized for training.")
        else:
            print(f"\n⚠️  TRAINING PIPELINE NEEDS ATTENTION:")
            if not model_compatible:
                print(f"   - Fix model compatibility issues")
            if not ecg_normalized:
                print(f"   - Consider ECG normalization")
            if not rr_normalized:
                print(f"   - Consider RR feature scaling")


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training pipeline compatibility')
    parser.add_argument('--data-dir', type=str, default='data/mitbih',
                        help='Path to MIT-BIH data directory')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = TrainingPipelineAnalyzer(args.data_dir)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()