# Scripts Directory

## Testing & Analysis Scripts
- **`test_mitbih_dataset.py`** - Comprehensive test suite for MITBIHDataset. Tests data loading, class mapping, shapes, performance, and integrity.
- **`dataset_test.py`** - Training pipeline compatibility analyzer. Analyzes tensor shapes, data distributions, and provides normalization recommendations.
- **`data_viewer.py`** - Simple ECG data visualizer. Shows actual ECG signals, class imbalance, and matched filter templates.
- **`visual_data_explorer.py`** - Enhanced data explorer with detailed visualizations of ECG signals, class distributions, and model input analysis.

## Key Features
- All scripts fixed to work with updated dataset structure (no more pickle dependencies)
- Direct WFDB file loading from `data/mitbih/`
- Comprehensive testing and visualization capabilities 