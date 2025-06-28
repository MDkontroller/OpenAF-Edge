# scripts/download_afdb.py
import wfdb
import os

OUTPUT_DIR = "data/afdb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading entire AFDB dataset from PhysioNet...")
wfdb.dl_database("afdb", dl_dir=OUTPUT_DIR)
