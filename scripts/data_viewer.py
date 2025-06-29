#!/usr/bin/env python3
"""
Simple Data Viewer - Just show me the damn data!

No menus, no complexity - just SHOW the ECG data visually
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add data directory to path
sys.path.append(str(Path(__file__).parent.parent / "data"))

def show_me_the_data():
    """Just show the ECG data - no BS"""
    
    print("🫀 LOADING ECG DATA...")
    
    # Load dataset
    from dataset import MITBIHDataset
    
    dataset = MITBIHDataset(
        data_dir="data/mitbih",
        split='train',
        num_classes=3,
        use_derivative=True,
        cache_dir="data/cache"
    )
    
    print(f"✅ Loaded {len(dataset)} ECG samples")
    
    # SHOW BASIC INFO
    print("\n📊 WHAT'S IN THE DATA:")
    print("-" * 30)
    
    class_dist = dataset.get_class_distribution()
    total = sum(class_dist.values())
    
    for class_name, count in class_dist.items():
        percentage = count / total * 100
        print(f"{class_name:5}: {count:6,} samples ({percentage:5.1f}%)")
    
    # Get first sample to show shapes
    ecg, rr, label = dataset[0]
    print(f"\nDATA SHAPES:")
    print(f"  ECG shape: {ecg.shape}")
    print(f"  RR shape:  {rr.shape}")
    print(f"  Label:     {label} (class: {list(class_dist.keys())[label]})")
    
    # SHOW ACTUAL ECG SIGNALS
    print("\n🫀 SHOWING ACTUAL ECG SIGNALS...")
    
    # Create big plot
    fig = plt.figure(figsize=(20, 12))
    
    # Find examples of each class
    class_examples = {'N': [], 'SVEB': [], 'VEB': []}
    class_names = list(class_dist.keys())
    
    # Get 4 examples of each class
    for i in range(min(1000, len(dataset))):
        ecg, rr, label = dataset[i]
        class_name = class_names[label.item()]
        if len(class_examples[class_name]) < 4:
            class_examples[class_name].append((ecg[0].numpy(), rr.numpy(), i))
    
    # Plot grid: 3 classes × 4 examples = 12 subplots
    colors = {'N': 'blue', 'SVEB': 'orange', 'VEB': 'red'}
    
    for class_idx, (class_name, examples) in enumerate(class_examples.items()):
        for example_idx in range(4):
            subplot_idx = class_idx * 4 + example_idx + 1
            ax = plt.subplot(3, 4, subplot_idx)
            
            if example_idx < len(examples):
                ecg_signal, rr_features, sample_idx = examples[example_idx]
                
                # Plot ECG
                ax.plot(ecg_signal, color=colors[class_name], linewidth=2)
                ax.set_title(f'{class_name} - Sample {sample_idx}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-3, 3)  # Standard ECG range
                
                # Show some stats
                stats = f"Min: {ecg_signal.min():.2f}\nMax: {ecg_signal.max():.2f}\nMean: {ecg_signal.mean():.2f}"
                ax.text(0.02, 0.98, stats, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{class_name} - No sample')
            
            if class_idx == 2:  # Bottom row
                ax.set_xlabel('Time (samples)')
            if example_idx == 0:  # First column
                ax.set_ylabel('Amplitude')
    
    plt.suptitle('ACTUAL ECG SIGNALS FROM YOUR DATASET', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # SHOW CLASS IMBALANCE VISUALLY
    print("\n📊 CLASS IMBALANCE VISUALIZATION...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart (log scale)
    bars = ax1.bar(class_dist.keys(), class_dist.values(), 
                   color=['blue', 'orange', 'red'], alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_title('Class Counts (Log Scale)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    
    # Add numbers on bars
    for bar, count in zip(bars, class_dist.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(class_dist.values(), labels=class_dist.keys(), autopct='%1.1f%%',
           colors=['blue', 'orange', 'red'], alpha=0.7, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')
    
    plt.suptitle('YOUR DATA IS SEVERELY IMBALANCED!', fontsize=16, fontweight='bold', color='red')
    plt.tight_layout()
    plt.show()
    
    # SHOW MATCHED FILTER TEMPLATES
    print("\n🎯 MATCHED FILTER TEMPLATES...")
    
    templates = dataset.get_templates_by_class()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (class_name, template) in enumerate(templates.items()):
        ax = axes[i]
        ax.plot(template, color=colors[class_name], linewidth=3)
        ax.set_title(f'{class_name} Template\n(Average of all {class_name} beats)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Stats
        stats = f"Mean: {template.mean():.3f}\nStd: {template.std():.3f}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('TEMPLATES FOR MATCHED FILTER CNN', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # SHOW WHAT GOES INTO THE MODEL
    print("\n🔄 WHAT ACTUALLY GOES INTO YOUR MODEL...")
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_ecg, batch_rr, batch_labels = next(iter(loader))
    
    print(f"BATCH SHAPES:")
    print(f"  ECG batch:    {batch_ecg.shape}    <- This goes to CNN")
    print(f"  RR batch:     {batch_rr.shape}      <- This goes to FC layers")
    print(f"  Label batch:  {batch_labels.shape}           <- These are targets")
    print(f"  Label values: {batch_labels.tolist()}   <- Actual class indices")
    
    # Plot the batch
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    class_names_list = list(class_dist.keys())
    
    for i in range(4):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        ecg_signal = batch_ecg[i, 0].numpy()  # [batch, channel, samples]
        rr_features = batch_rr[i].numpy()
        label_idx = batch_labels[i].item()
        class_name = class_names_list[label_idx]
        
        ax.plot(ecg_signal, color=colors[class_name], linewidth=2)
        ax.set_title(f'Batch[{i}] -> {class_name} (label={label_idx})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Show RR features
        rr_text = f"RR features: [{rr_features[0]:.2f}, {rr_features[1]:.2f}, {rr_features[2]:.2f}, {rr_features[3]:.2f}]"
        ax.text(0.02, 0.02, rr_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('ACTUAL BATCH GOING INTO YOUR MODEL', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # FINAL SUMMARY
    print("\n" + "="*60)
    print("🎯 SUMMARY - WHAT YOU HAVE:")
    print("="*60)
    print(f"✅ {len(dataset):,} ECG heartbeat samples")
    print(f"✅ 3 classes: Normal ({class_dist['N']:,}), SVEB ({class_dist['SVEB']:,}), VEB ({class_dist['VEB']:,})")
    print(f"✅ Each sample: ECG signal (64 points) + RR features (4 values)")
    print(f"✅ Severely imbalanced data (89% normal, 11% arrhythmias)")
    print(f"✅ Ready for matched filter CNN training")
    print(f"✅ Caching enabled for fast loading")
    print("\n🚀 YOUR DATA IS READY FOR TRAINING!")
    print("="*60)

if __name__ == "__main__":
    show_me_the_data()