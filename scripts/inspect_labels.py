import pandas as pd
import os

# Define paths
DATA_PROCESSED_DIR = "data/processed/"

# Load the label files
y_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_cell2cell_val.csv"))
y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_cell2cell_test.csv"))
y_telco = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_telco.csv"))

# Inspect unique values and NaN counts
print("=== Cell2Cell Validation Labels ===")
print("Unique values:", y_val['Churn'].unique())
print("NaN count:", y_val['Churn'].isna().sum())

print("\n=== Cell2Cell Test Labels ===")
print("Unique values:", y_test['Churn'].unique())
print("NaN count:", y_test['Churn'].isna().sum())

print("\n=== IBM Telco Labels ===")
print("Unique values:", y_telco['Churn'].unique())
print("NaN count:", y_telco['Churn'].isna().sum())