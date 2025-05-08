---

# 📉 Customer Churn Prediction

This repository contains a complete pipeline for predicting customer churn in the telecommunications sector. It helps identify customers likely to leave and offers interpretable insights and retention strategies using machine learning and explainable AI.

---

## 🚀 Project Overview

This project uses the **Telco Customer Churn dataset** to:

* Perform in-depth **exploratory data analysis (EDA)** to uncover churn patterns.
* Train multiple **machine learning models** (Random Forest, LightGBM, and Stacking) for high-recall churn prediction.
* Deploy an interactive **Streamlit dashboard** that allows users to input customer data, generate predictions, and interpret results using SHAP visualizations.

### 🔑 Key Features

* Preprocessing of 23 raw features (5 numerical, 18 categorical) into 28 model-ready features using `ColumnTransformer`.
* Achieves **F1-score = 0.58** and **ROC-AUC = 0.824** using the Random Forest model.
* Model explanations and customer-specific insights using **SHAP**.
* Interactive dashboard for churn risk prediction and stakeholder decision-making.

---

## 🗂 Repository Structure

```
Customer-Churn-Prediction-/
├── Data/
│   └── telco_customer_churn.csv
├── models/
│   ├── telco_churn_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_names.txt
│   ├── input_feature_names.txt
├── EDA.ipynb                 # Exploratory Data Analysis
├── Model_Development.ipynb   # Model training and evaluation
├── app.py                    # Streamlit dashboard application
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 💻 Prerequisites

* **Python**: 3.12.3
* **RAM**: Minimum of 4GB for model training and dashboard execution
* **OS**: Windows, macOS, or Linux
* **Dataset**: Download `telco_customer_churn.csv` from Kaggle

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Akobabs/Customer-Churn-Prediction-.git
cd Customer-Churn-Prediction-
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key libraries** used:

* `pandas`, `numpy`, `scikit-learn`
* `lightgbm`, `imblearn`, `shap`
* `streamlit`, `plotly`, `seaborn`, `matplotlib`
* `joblib`

---

## 📥 Download the Dataset

1. Visit the Kaggle dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Rename extracted file to `telco_customer_churn.csv`
3. Place `telco_customer_churn.csv` in the `Data/` directory.

**Optional (using Kaggle API):**

```bash
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d Data/
```

---

## ▶️ Running the Project

### 1. Exploratory Data Analysis (EDA)

Launch Jupyter Notebook and open:

```bash
jupyter notebook EDA.ipynb
```

* Visualize churn rates, contract types, payment methods, etc.
* Validate feature engineering steps like `tenure_bin` and `Fiber_no_Security`.

---

### 2. Model Training

Open the notebook:

```bash
jupyter notebook Model_Development.ipynb
```

* Preprocess the dataset and engineer features.
* Train and evaluate models.
* Generate SHAP plots for model explainability.

**Artifacts Saved:**

* `models/telco_churn_model.pkl`
* `models/preprocessor.pkl`
* `models/feature_names.txt`
* `models/input_feature_names.txt`

---

### 3. Run the Streamlit App

Start the dashboard:

```bash
streamlit run app.py
```

* Access via browser: `http://localhost:8501`
* Input customer data in the sidebar.
* View predictions, SHAP-based explanations, and EDA insights.

---

## 🧰 Troubleshooting

### ❌ `ValueError: X has 28 features, but ColumnTransformer is expecting 23 features`

* Ensure `input_feature_names.txt` lists only the 23 original raw features.
* Re-run `Model_Development.ipynb` to regenerate all model artifacts.

### ❌ Missing `.pkl` files or `feature_names.txt`

* Execute `Model_Development.ipynb` fully.
* Confirm the `models/` directory contains all outputs.

### ❌ Dataset Errors

* Confirm `Data/telco_customer_churn.csv` exists and is formatted correctly.

### ❌ Dependency Issues

* Double-check Python version: `python --version`
* Reinstall packages: `pip install -r requirements.txt`

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork** the repository
2. **Create a new branch**
   `git checkout -b feature/your-feature`
3. **Commit your changes**
   `git commit -m "Add your feature"`
4. **Push** to GitHub
   `git push origin feature/your-feature`
5. **Open a Pull Request**

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

* GitHub: [@Akobabs](https://github.com/Akobabs)
* Email: `akorede.ademola[at]yahoo[dot]com`

---