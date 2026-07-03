import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Fraud Detection", page_icon="🕵️")

st.title("🕵️ Credit Card Fraud Detection")
st.write("Upload transaction data to flag potentially fraudulent transactions.")

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_detection_model.pkl")
AMOUNT_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_amount_scaler.pkl")
TIME_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_time_scaler.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    amount_scaler = joblib.load(AMOUNT_SCALER_PATH)
    time_scaler = joblib.load(TIME_SCALER_PATH)
    return model, amount_scaler, time_scaler

try:
    model, amount_scaler, time_scaler = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Could not find a required model/scaler file: {e}\n\n"
        "This page needs fraud_detection_model.pkl, fraud_amount_scaler.pkl, "
        "and fraud_time_scaler.pkl in your models/ folder. See the notebook "
        "fix for saving these three files together."
    )
    st.stop()

# ----------------------------------------------------
# Exact column order used during training: Time, V1..V28, Amount
# ----------------------------------------------------
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_ORDER = ["Time"] + V_COLS + ["Amount"]

# ----------------------------------------------------
# File uploader
# ----------------------------------------------------
st.caption(
    "Expected columns: Time, V1–V28, Amount "
    "(the same format as the original creditcard.csv dataset). "
    "A 'Class' column, if present, is ignored."
)
uploaded_file = st.file_uploader("Upload Transaction Data (CSV)", type="csv")

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(raw_df.head())

    if st.button("🔍 Scan for Fraud"):
        try:
            data = raw_df.copy()
            if "Class" in data.columns:
                data = data.drop("Class", axis=1)

            missing_cols = [c for c in FEATURE_ORDER if c not in data.columns]
            if missing_cols:
                st.error(f"Uploaded CSV is missing required column(s): {missing_cols}")
                st.stop()

            data = data[FEATURE_ORDER].copy()

            # Fill any missing numeric values with column median
            if data.isnull().any().any():
                data = data.fillna(data.median(numeric_only=True))

            # Scale Amount and Time using the SAME scalers fit during training
            data["Amount"] = amount_scaler.transform(data[["Amount"]])
            data["Time"] = time_scaler.transform(data[["Time"]])

            predictions = model.predict(data)

            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(data)
                # probability of the predicted class
                confidence = np.max(probs, axis=1)

            result_df = raw_df.copy()
            result_df["Prediction"] = [
                "🚨 Fraud" if p == 1 else "✅ Legitimate" for p in predictions
            ]
            if confidence is not None:
                result_df["Confidence"] = [f"{c*100:.1f}%" for c in confidence]

            n_fraud = int((predictions == 1).sum())
            if n_fraud > 0:
                st.warning(f"⚠️ {n_fraud} potentially fraudulent transaction(s) detected out of {len(predictions)}.")
            else:
                st.success(f"No fraudulent transactions detected out of {len(predictions)}.")

            st.write("### Results")
            # Show flagged transactions first
            st.dataframe(result_df.sort_values("Prediction", ascending=True))

            st.download_button(
                "⬇️ Download Results as CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                "fraud_predictions.csv",
                "text/csv",
            )

        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")
else:
    st.info("Upload a CSV file above to get started.")