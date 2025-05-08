import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page config
st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

# Load model, preprocessor, and feature names
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load('models/telco_churn_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    with open('models/feature_names.txt', 'r') as f:
        transformed_feature_names = f.read().splitlines()
    # Get expected input columns from preprocessor
    expected_input_columns = preprocessor.feature_names_in_
    return model, preprocessor, transformed_feature_names, expected_input_columns

model, preprocessor, transformed_feature_names, expected_input_columns = load_model_and_preprocessor()

# Load dataset for EDA visualizations
@st.cache_data
def load_data():
    df = pd.read_csv('Data/telco_customer_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[(df['TotalCharges'].isnull()) & (df['tenure'] == 0), 'TotalCharges'] = 0
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    for col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

# Feature engineering function
def engineer_features(input_data):
    input_data['tenure_bin'] = pd.cut(input_data['tenure'], bins=[-1, 12, 24, 72], labels=['0-12', '12-24', '>24'])
    input_data['MonthlyCharges_per_tenure'] = input_data['MonthlyCharges'] / (input_data['tenure'] + 1)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_data['ServiceCount'] = input_data[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    input_data['Fiber_no_Security'] = ((input_data['InternetService'] == 'Fiber optic') & (input_data['OnlineSecurity'] == 'No')).astype(int)
    return input_data

# Sidebar for user inputs
st.sidebar.header("Customer Data Input")
def get_user_input():
    input_dict = {
        'tenure': st.sidebar.slider("Tenure (months)", 0, 72, 12),
        'MonthlyCharges': st.sidebar.slider("Monthly Charges ($)", 18.0, 118.0, 50.0),
        'TotalCharges': st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 600.0),
        'Contract': st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year']),
        'PaymentMethod': st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
        'InternetService': st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No']),
        'OnlineSecurity': st.sidebar.selectbox("Online Security", ['Yes', 'No']),
        'OnlineBackup': st.sidebar.selectbox("Online Backup", ['Yes', 'No']),
        'DeviceProtection': st.sidebar.selectbox("Device Protection", ['Yes', 'No']),
        'TechSupport': st.sidebar.selectbox("Tech Support", ['Yes', 'No']),
        'StreamingTV': st.sidebar.selectbox("Streaming TV", ['Yes', 'No']),
        'StreamingMovies': st.sidebar.selectbox("Streaming Movies", ['Yes', 'No']),
        'gender': st.sidebar.selectbox("Gender", ['Female', 'Male']),
        'SeniorCitizen': st.sidebar.selectbox("Senior Citizen", [0, 1]),
        'Partner': st.sidebar.selectbox("Partner", ['Yes', 'No']),
        'Dependents': st.sidebar.selectbox("Dependents", ['Yes', 'No']),
        'PhoneService': st.sidebar.selectbox("Phone Service", ['Yes', 'No']),
        'MultipleLines': st.sidebar.selectbox("Multiple Lines", ['Yes', 'No']),
        'PaperlessBilling': st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])
    }
    input_df = pd.DataFrame([input_dict])
    # Engineer features
    input_df = engineer_features(input_df)
    # Ensure only expected columns are included and in correct order
    missing_cols = [col for col in expected_input_columns if col not in input_df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return None
    extra_cols = [col for col in input_df.columns if col not in expected_input_columns]
    if extra_cols:
        st.warning(f"Removing extra columns: {extra_cols}")
        input_df = input_df.drop(columns=extra_cols)
    # Reorder columns to match preprocessor
    input_df = input_df[expected_input_columns]
    return input_df

# Main panel
st.title("Telco Customer Churn Prediction Dashboard")
st.markdown("Enter customer details in the sidebar to predict churn risk and view insights.")

# Display expected input columns for debugging
st.write("Expected Input Columns:", expected_input_columns)

# Get user input and predict
input_df = get_user_input()
if input_df is not None:
    st.write("Input Data Columns:", input_df.columns.tolist())
    try:
        X_processed = preprocessor.transform(input_df)
        st.write(f"Transformed Data Shape: {X_processed.shape} (Expected {len(transformed_feature_names)} features)")
        pred = model.predict(X_processed)[0]
        pred_proba = model.predict_proba(X_processed)[0]

        # Display prediction
        st.subheader("Churn Prediction")
        col1, col2 = st.columns(2)
        col1.metric("Churn Prediction", "Yes" if pred == 1 else "No")
        col2.metric("Churn Probability", f"{pred_proba[1]:.2%}")

        # SHAP explanation
        st.subheader("Prediction Explanation (SHAP)")
        explainer = shap.TreeExplainer(model.named_steps['model'] if 'Stacking' not in str(model) else model.named_steps['model'].estimators_[0])
        shap_values = explainer.shap_values(X_processed)
        # Convert X_processed to DataFrame with transformed feature names
        X_processed_df = pd.DataFrame(X_processed, columns=transformed_feature_names)
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_processed_df, matplotlib=True, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# EDA Visualizations
st.subheader("Key Insights from Data")
st.markdown("The following plots highlight factors influencing churn, based on exploratory data analysis.")

# Churn rate by Contract
fig = px.bar(df.groupby('Contract')['Churn'].mean().reset_index(), x='Contract', y='Churn', title="Churn Rate by Contract Type",
             labels={'Churn': 'Churn Rate'}, color='Contract')
fig.update_layout(yaxis_tickformat='.0%')
st.plotly_chart(fig)

# Churn rate by Payment Method
fig = px.bar(df.groupby('PaymentMethod')['Churn'].mean().reset_index(), x='PaymentMethod', y='Churn', title="Churn Rate by Payment Method",
             labels={'Churn': 'Churn Rate'}, color='PaymentMethod')
fig.update_layout(yaxis_tickformat='.0%')
st.plotly_chart(fig)

# Recommendations
st.subheader("Retention Recommendations")
if input_df is not None and 'pred' in locals() and pred == 1:
    st.markdown("""
    **This customer is at high risk of churning. Consider the following actions:**
    - **Offer a longer-term contract**: Customers with one- or two-year contracts have lower churn rates (see plot above).
    - **Switch to automatic payments**: Electronic check users have higher churn rates; suggest bank transfer or credit card.
    - **Bundle services**: Increasing the number of services (e.g., adding Online Security) may improve retention.
    - **Review pricing**: High Monthly Charges are a key churn driver; consider discounts or promotions.
    """)
elif input_df is not None and 'pred' in locals():
    st.markdown("**This customer is likely to stay. Continue monitoring and offer loyalty rewards to maintain satisfaction.**")
else:
    st.markdown("**Please provide valid input data to generate recommendations.**")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data: Telco Customer Churn | Model: Trained using LightGBM/Random Forest")