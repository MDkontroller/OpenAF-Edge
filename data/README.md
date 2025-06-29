# Data Directory

## Files
- **`dataset.py`** - PyTorch dataset for MIT-BIH ECG classification. Loads WFDB files directly, supports AAMI class mapping, inter-patient division (DS1/DS2), RR interval features, and class balancing.
- **`download_mitbih.py`** - Downloads MIT-BIH Arrhythmia Database from PhysioNet
- **`download_afdb.py`** - Downloads MIT-BIH AFDB (Atrial Fibrillation Database) from PhysioNet

## Directories
- **`mitbih/`** - Contains MIT-BIH arrhythmia database WFDB files (.dat, .hea, .atr)
- **`afdb/`** - Contains MIT-BIH AFDB database WFDB files 