# scripts/download_mitbih_arrhythmia.py
import wfdb
import os

OUTPUT_DIR = "data/mitbih"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading MIT-BIH Arrhythmia Database from PhysioNet...")
wfdb.dl_database("mitdb", dl_dir=OUTPUT_DIR)

print("Download complete. Files saved to:", OUTPUT_DIR)
