import pandas as pd
import numpy as np
from textblob import TextBlob
import os

# Define paths
DATA_RAW_DIR = "data/raw/"
DATA_PROCESSED_DIR = "data/processed/"

# Ensure processed data directory exists
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Load datasets (same as exploration)
cell2cell = pd.read_csv(os.path.join(DATA_RAW_DIR, "cell2celltrain.csv"), nrows=10000)
telco = pd.read_csv(os.path.join(DATA_RAW_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))

# Cell2Cell IoB augmentation
def add_iob_features_cell2cell(data):
    def generate_feedback(row):
        if row['RetentionCalls'] > 0 or row['DroppedCalls'] > 10:
            return "Frustrated with frequent dropped calls and poor support."
        elif row['MonthlyMinutes'] > 500:
            return "Satisfied with heavy usage and reliable service."
        else:
            return "Average experience, no major issues."
    
    data['Feedback'] = data.apply(generate_feedback, axis=1)
    data['SentimentScore'] = data['Feedback'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data['SocialMediaActivity'] = np.random.randint(0, 10, size=len(data))
    data['ComplaintFrequency'] = np.random.randint(0, 5, size=len(data))
    data['CompetitorSearches'] = np.random.choice([0, 1], size=len(data), p=[0.8, 0.2])
    return data.drop(columns=['Feedback'])  # Drop text to avoid encoding issues

# IBM Telco IoB augmentation
def add_iob_features_telco(data):
    def generate_feedback(row):
        if row['TechSupport'] == 'Yes' and row['Contract'] == 'Two year':
            return "Great service, very reliable support!"
        elif row['TechSupport'] == 'No' and row['MonthlyCharges'] > 80:
            return "Expensive and no support, considering switching."
        else:
            return "Average experience, could be better."
    
    data['Feedback'] = data.apply(generate_feedback, axis=1)
    data['SentimentScore'] = data['Feedback'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data['SocialMediaActivity'] = np.random.randint(0, 10, size=len(data))
    data['ComplaintFrequency'] = np.random.randint(0, 5, size=len(data))
    data['CompetitorSearches'] = np.random.choice([0, 1], size=len(data), p=[0.8, 0.2])
    return data.drop(columns=['Feedback'])

# Apply IoB augmentation
cell2cell_augmented = add_iob_features_cell2cell(cell2cell)
telco_augmented = add_iob_features_telco(telco)

# Save augmented datasets
cell2cell_augmented.to_csv(os.path.join(DATA_PROCESSED_DIR, "cell2cell_augmented.csv"), index=False)
telco_augmented.to_csv(os.path.join(DATA_PROCESSED_DIR, "telco_augmented.csv"), index=False)

# Verify new features
print("\n=== Cell2Cell: New IoB Features ===")
print(cell2cell_augmented[['SentimentScore', 'SocialMediaActivity', 'ComplaintFrequency', 'CompetitorSearches']].head())
print("\n=== IBM Telco: New IoB Features ===")
print(telco_augmented[['SentimentScore', 'SocialMediaActivity', 'ComplaintFrequency', 'CompetitorSearches']].head())