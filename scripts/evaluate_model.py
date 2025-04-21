import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os

# Define paths
DATA_PROCESSED_DIR = "data/processed/"

# Load Cell2Cell validation and test data
X_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_cell2cell_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_cell2cell_val.csv"))
X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_cell2cell_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_cell2cell_test.csv"))

# Load IBM Telco preprocessed data
X_telco = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_telco_preprocessed.csv"))
y_telco = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_telco.csv"))

# Function to clean and map Churn labels
def clean_and_map_labels(df, dataset_name):
    # Extract Churn as a Series
    churn = df['Churn']
    
    # Check unique values to determine if mapping is needed
    unique_values = churn.unique()
    print(f"Unique {dataset_name} Churn values before processing:", unique_values)
    
    # Check for NaN values
    nan_count = churn.isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in {dataset_name} Churn column. Dropping these rows.")
        df = df.dropna(subset=['Churn'])
        churn = df['Churn']
    
    # Determine if mapping is needed (i.e., if values are Yes/No)
    if set(unique_values).issubset({1, 0, 1.0, 0.0}):
        # Labels are already binary (1/0), no mapping needed
        churn_mapped = churn.astype(int)
    else:
        # Standardize case (e.g., convert to title case)
        churn = churn.str.title()
        # Map to binary values
        mapping = {'Yes': 1, 'No': 0}
        churn_mapped = churn.map(mapping)
        
        # Check for NaN values after mapping (indicating unexpected values)
        nan_after_mapping = churn_mapped.isna().sum()
        if nan_after_mapping > 0:
            unexpected_values = churn[churn_mapped.isna()].unique()
            raise ValueError(f"Unexpected values {unexpected_values} in {dataset_name} Churn column after mapping.")
    
    return churn_mapped, df

# Clean and map labels for all datasets
y_val, val_df = clean_and_map_labels(y_val, "Cell2Cell Validation")
X_val = X_val.loc[val_df.index]  # Align X_val with dropped rows

y_test, test_df = clean_and_map_labels(y_test, "Cell2Cell Test")
X_test = X_test.loc[test_df.index]  # Align X_test with dropped rows

y_telco, telco_df = clean_and_map_labels(y_telco, "IBM Telco")
X_telco = X_telco.loc[telco_df.index]  # Align X_telco with dropped rows

# Load the trained model
stacking_model = joblib.load(os.path.join(DATA_PROCESSED_DIR, "stacking_model_cell2cell.pkl"))

# Evaluate on Cell2Cell validation set
y_pred_val = stacking_model.predict(X_val)
y_pred_proba_val = stacking_model.predict_proba(X_val)[:, 1]

print("\n=== Cell2Cell Validation Set Evaluation ===")
print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Precision: {precision_score(y_val, y_pred_val):.4f}")
print(f"Recall: {recall_score(y_val, y_pred_val):.4f}")
print(f"F1-Score: {f1_score(y_val, y_pred_val):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_val, y_pred_proba_val):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val))

# Evaluate on Cell2Cell test set
y_pred_test = stacking_model.predict(X_test)
y_pred_proba_test = stacking_model.predict_proba(X_test)[:, 1]

print("\n=== Cell2Cell Test Set Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_test):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_test):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Attempt validation on IBM Telco
try:
    y_pred_telco = stacking_model.predict(X_telco)
    y_pred_proba_telco = stacking_model.predict_proba(X_telco)[:, 1]

    print("\n=== IBM Telco Validation ===")
    print(f"Accuracy: {accuracy_score(y_telco, y_pred_telco):.4f}")
    print(f"Precision: {precision_score(y_telco, y_pred_telco):.4f}")
    print(f"Recall: {recall_score(y_telco, y_pred_telco):.4f}")
    print(f"F1-Score: {f1_score(y_telco, y_pred_telco):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_telco, y_pred_proba_telco):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_telco, y_pred_telco))
except ValueError as e:
    print("\nWarning: IBM Telco validation failed due to feature mismatch:", e)
    print("Consider training a separate model for IBM Telco or aligning features.")