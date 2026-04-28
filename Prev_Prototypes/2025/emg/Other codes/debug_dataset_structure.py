#!/usr/bin/env python3
"""
Debug Dataset Structure
======================

This script checks the structure of the combined_emg_dorsiflex.csv file
to understand the column names and data format.
"""

import pandas as pd
from pathlib import Path

def main():
    # Load the dataset
    dataset_path = Path(__file__).parent.parent / "dataset" / "combined_emg_dorsiflex.csv"
    
    print(f"Loading dataset: {dataset_path}")
    
    # Read just the first few rows to check structure
    df = pd.read_csv(dataset_path, nrows=10)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Column dtypes: {df.dtypes.tolist()}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nUnique values in Subject: {sorted(df['Subject'].unique())}")
    print(f"Unique values in MVC: {sorted(df['MVC'].unique())}")
    print(f"Unique values in Trial: {sorted(df['Trial'].unique())}")
    
    # Check for any missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Check EMG signal statistics
    print(f"\nEMG signal statistics:")
    print(df['fwEMG 3'].describe())

if __name__ == "__main__":
    main()
