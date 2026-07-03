# 🏦 Smart Banking Risk Analytics Platform

A unified web application combining four machine learning models to support key banking decisions — loan approval, fraud detection, customer segmentation, and churn prediction — all in one interactive dashboard.

**🔗 Live App:** https://smartbankingriskanalyticsplatform-7zvbs35b9cpcxca9abdbvu.streamlit.app/
---

## 📌 What This Project Does

Banks make thousands of data-driven decisions every day. This platform brings together four independently trained machine learning models into a single, easy-to-use tool:

| Module | Model | Purpose |
|---|---|---|
| 💰 **Loan Approval Prediction** | Logistic Regression | Predicts whether a loan application is likely to be approved based on applicant income, credit history, and demographics |
| 🕵️ **Fraud Detection** | Random Forest | Scans credit card transaction data to flag potentially fraudulent activity |
| 📊 **Customer Segmentation** | K-Means Clustering | Groups customers into behavioral segments based on balance, spending, and credit usage |
| 📉 **Churn Prediction** | Stacking Classifier (LightGBM + XGBoost + Random Forest) | Predicts whether a customer is likely to leave, based on billing, contract, and service usage |

## 🎯 Why It's Useful

Instead of four disconnected notebooks, this project packages real banking use cases into one usable product:
- **Approve loans faster** with a data-backed risk assessment
- **Catch fraud** before it causes financial loss
- **Understand customers** to target the right products to the right people
- **Reduce churn** by identifying at-risk customers before they leave

## 🛠️ Tech Stack

- **Frontend/Deployment:** Streamlit (multi-page app)
- **ML/Data:** scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE), pandas, NumPy
- **Model Persistence:** joblib

## 📂 Project Structure

```
Smart_banking_risk_analytics_platform/
├── Home.py                          # Landing page
├── pages/
│   ├── 1_Loan_Approval.py
│   ├── 2_Fraud_Detection.py
│   ├── 3_Customer_Segmentation.py
│   └── 4_Churn_Prediction.py
├── models/                          # Trained models & scalers (.pkl)
├── notebooks/                       # Training notebooks for all 4 models
└── requirements.txt
```

## 🚀 Running Locally

```bash
git clone https://github.com/ruhabtahir12/Smart_banking_risk_analytics_platform.git
cd Smart_banking_risk_analytics_platform
pip install -r requirements.txt
streamlit run Home.py
```

## 📊 Datasets Used

- Loan Approval: [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) (Kaggle)
- Fraud Detection: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle)
- Customer Segmentation: Credit Card Customer Dataset
- Churn Prediction: Telco Customer Churn Dataset

## ✨ Features

- Manual data entry forms for individual predictions
- CSV batch upload for bulk processing
- Confidence scores on every prediction
- Auto-generated, human-readable customer segment labels

## 👤 Author

**Ruhab Tahir**
GitHub: [@ruhabtahir12](https://github.com/ruhabtahir12)
