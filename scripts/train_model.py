import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os

# Define paths
DATA_PROCESSED_DIR = "data/processed/"

# Load preprocessed Cell2Cell data
X_cell2cell = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_cell2cell_preprocessed.csv"))
y_cell2cell = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_cell2cell.csv"))['Churn']

# Convert Churn to binary (Yes=1, No=0)
y_cell2cell = y_cell2cell.map({'Yes': 1, 'No': 0})

# Step 1: Split into 72% (train+validation) and 28% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_cell2cell, y_cell2cell, test_size=0.28, random_state=42, stratify=y_cell2cell
)

# Step 2: Split the 72% into 63% train (63/72 = 87.5%) and 9% validation (9/72 = 12.5%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp  # 9/(63+9) = 0.125
)

# Verify split proportions
total_samples = len(X_cell2cell)
print(f"Total samples: {total_samples}")
print(f"Training set: {len(X_train)} ({len(X_train)/total_samples:.3f})")
print(f"Validation set: {len(X_val)} ({len(X_val)/total_samples:.3f})")
print(f"Test set: {len(X_test)} ({len(X_test)/total_samples:.3f})")

# Calculate scale_pos_weight for XGBoost (ratio of negative to positive instances)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale Pos Weight for XGBoost: {scale_pos_weight:.2f}")

# Define base models
base_models = [
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('xgb', XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42)),
    ('svm', SVC(class_weight='balanced', probability=True, random_state=42))
]

# Define stacking classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)

# Train the model on the training set
stacking_model.fit(X_train, y_train)

# Save the model
joblib.dump(stacking_model, os.path.join(DATA_PROCESSED_DIR, "stacking_model_cell2cell.pkl"))
print("Model trained and saved successfully.")

# Save the splits for evaluation
X_train.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_cell2cell_train.csv"), index=False)
y_train.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_cell2cell_train.csv"), index=False)
X_val.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_cell2cell_val.csv"), index=False)
y_val.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_cell2cell_val.csv"), index=False)
X_test.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_cell2cell_test.csv"), index=False)
y_test.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_cell2cell_test.csv"), index=False)
print("Train, validation, and test sets saved for evaluation.")