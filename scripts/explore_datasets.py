import pandas as pd
import os

# Define paths
DATA_RAW_DIR = "data/raw/"

# Ensure raw data directory exists
os.makedirs(DATA_RAW_DIR, exist_ok=True)

# Load datasets
# Note: Replace paths if datasets are elsewhere
try:
    # Load Cell2Cell subset (10,000 records)
    cell2cell = pd.read_csv(os.path.join(DATA_RAW_DIR, "cell2celltrain.csv"), nrows=10000)
    print("Cell2Cell loaded successfully.")
except FileNotFoundError:
    print("Error: cell2celltrain.csv not found in data/raw/. Please download from Kaggle.")

try:
    # Load IBM Telco full dataset
    telco = pd.read_csv(os.path.join(DATA_RAW_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    print("IBM Telco loaded successfully.")
except FileNotFoundError:
    print("Error: WA_Fn-UseC_-Telco-Customer-Churn.csv not found in data/raw/. Please download from Kaggle.")

# Function to explore dataset
def explore_dataset(data, name):
    print(f"\n=== {name} Dataset Exploration ===")
    print("Shape:", data.shape)
    print("\nFeatures:")
    print(data.columns.tolist())
    print("\nData Types:")
    print(data.dtypes)
    print("\nFirst 5 Rows:")
    print(data.head())
    print("\nChurn Distribution:")
    try:
        print(data['Churn'].value_counts(normalize=True))
    except KeyError:
        print("Churn column not found. Available columns:", data.columns.tolist())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nSummary Statistics:")
    print(data.describe(include='all'))

# Explore both datasets
explore_dataset(cell2cell, "Cell2Cell")
explore_dataset(telco, "IBM Telco")