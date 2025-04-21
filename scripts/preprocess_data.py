import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

# Define paths
DATA_PROCESSED_DIR = "data/processed/"

# Ensure processed data directory exists
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Load augmented datasets
cell2cell = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "cell2cell_augmented.csv"))
telco = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "telco_augmented.csv"))

# Type conversion for IBM Telco: Convert TotalCharges to float
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')

# Drop CustomerID for privacy
cell2cell = cell2cell.drop(columns=['CustomerID'])
telco = telco.drop(columns=['customerID'])

# Define numerical and categorical columns (exclude Churn as it's the target)
# Cell2Cell
cell2cell_numerical = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 'DirectorAssistedCalls',
                       'OverageMinutes', 'RoamingCalls', 'PercChangeMinutes', 'PercChangeRevenues',
                       'DroppedCalls', 'BlockedCalls', 'UnansweredCalls', 'CustomerCareCalls',
                       'ThreewayCalls', 'ReceivedCalls', 'OutboundCalls', 'InboundCalls',
                       'PeakCallsInOut', 'OffPeakCallsInOut', 'DroppedBlockedCalls', 'CallForwardingCalls',
                       'CallWaitingCalls', 'MonthsInService', 'UniqueSubs', 'ActiveSubs',
                       'Handsets', 'HandsetModels', 'CurrentEquipmentDays', 'AgeHH1', 'AgeHH2',
                       'RetentionCalls', 'RetentionOffersAccepted', 'SentimentScore',
                       'SocialMediaActivity', 'ComplaintFrequency', 'CompetitorSearches']
cell2cell_categorical = ['ServiceArea', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable',
                         'TruckOwner', 'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers',
                         'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'HasCreditCard',
                         'NewCellphoneUser', 'NotNewCellphoneUser', 'ReferralsMadeBySubscriber',
                         'IncomeGroup', 'OwnsMotorcycle', 'AdjustmentsToCreditRating', 'MadeCallToRetentionTeam',
                         'CreditRating', 'PrizmCode', 'Occupation', 'MaritalStatus']

# IBM Telco
telco_numerical = ['tenure', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
                   'SentimentScore', 'SocialMediaActivity', 'ComplaintFrequency', 'CompetitorSearches']
telco_categorical = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                     'PaperlessBilling', 'PaymentMethod']

# Validate columns exist
def validate_columns(data, columns, dataset_name):
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        print(f"Warning: Columns {missing_cols} not found in {dataset_name} dataset.")
    return [col for col in columns if col in data.columns]

cell2cell_numerical = validate_columns(cell2cell, cell2cell_numerical, "Cell2Cell")
cell2cell_categorical = validate_columns(cell2cell, cell2cell_categorical, "Cell2Cell")
telco_numerical = validate_columns(telco, telco_numerical, "IBM Telco")
telco_categorical = validate_columns(telco, telco_categorical, "IBM Telco")

# Define preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Cell2Cell preprocessor
cell2cell_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, cell2cell_numerical),
        ('cat', categorical_transformer, cell2cell_categorical)
    ])

# IBM Telco preprocessor
telco_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, telco_numerical),
        ('cat', categorical_transformer, telco_categorical)
    ])

# Fit and transform datasets
# Cell2Cell
X_cell2cell = cell2cell.drop(columns=['Churn'])
y_cell2cell = cell2cell['Churn']
X_cell2cell_preprocessed = cell2cell_preprocessor.fit_transform(X_cell2cell)
X_cell2cell_preprocessed_df = pd.DataFrame(
    X_cell2cell_preprocessed,
    columns=cell2cell_preprocessor.get_feature_names_out()
)

# IBM Telco
X_telco = telco.drop(columns=['Churn'])
y_telco = telco['Churn']
X_telco_preprocessed = telco_preprocessor.fit_transform(X_telco)
X_telco_preprocessed_df = pd.DataFrame(
    X_telco_preprocessed,
    columns=telco_preprocessor.get_feature_names_out()
)

# Save preprocessed data and preprocessors
X_cell2cell_preprocessed_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_cell2cell_preprocessed.csv"), index=False)
y_cell2cell.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_cell2cell.csv"), index=False)
X_telco_preprocessed_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_telco_preprocessed.csv"), index=False)
y_telco.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_telco.csv"), index=False)

joblib.dump(cell2cell_preprocessor, os.path.join(DATA_PROCESSED_DIR, "cell2cell_preprocessor.pkl"))
joblib.dump(telco_preprocessor, os.path.join(DATA_PROCESSED_DIR, "telco_preprocessor.pkl"))

print("\nPreprocessed datasets and preprocessors saved successfully.")
print("Cell2Cell preprocessed shape:", X_cell2cell_preprocessed_df.shape)
print("IBM Telco preprocessed shape:", X_telco_preprocessed_df.shape)