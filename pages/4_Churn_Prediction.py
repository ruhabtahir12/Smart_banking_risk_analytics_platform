import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")

st.title("📉 Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave (churn).")

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "churn_scaler.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "models", "churn_features.pkl")
THRESHOLD_PATH = os.path.join(PROJECT_ROOT, "models", "churn_high_charge_threshold.pkl")

DEFAULT_HIGH_CHARGE_THRESHOLD = 89.85  # approximate fallback if you haven't saved the real one

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    try:
        threshold = joblib.load(THRESHOLD_PATH)
        threshold_is_default = False
    except FileNotFoundError:
        threshold = DEFAULT_HIGH_CHARGE_THRESHOLD
        threshold_is_default = True
    return model, scaler, feature_names, threshold, threshold_is_default

try:
    model, scaler, TRAINED_FEATURES, HIGH_CHARGE_THRESHOLD, threshold_is_default = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Could not find a required file: {e}\n\n"
        "This page needs churn_model.pkl, churn_scaler.pkl, and churn_features.pkl "
        "in your models/ folder. See the notebook fix for saving these together."
    )
    st.stop()

if threshold_is_default:
    st.info(
        "ℹ️ Using an approximate 'high charge' threshold since churn_high_charge_threshold.pkl "
        "wasn't found. Predictions may be slightly less accurate — see instructions to save the exact value."
    )

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Replicates the exact feature engineering from Customer_churn_prediction.ipynb"""
    df = raw.copy()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    )

    df["AvgMonthly"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["MonthlyToTotalRatio"] = df["MonthlyCharges"] / (df["AvgMonthly"] + 1e-5)

    df["FiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["NoInternet"] = (df["InternetService"] == "No").astype(int)

    df["TotalServices"] = df[SERVICE_COLS].apply(
        lambda x: (x == "Yes").sum() if x.dtype == "object" else 0, axis=1
    )

    df["ContractNum"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

    df["SeniorPartner"] = ((df["SeniorCitizen"] == 1) & (df["Partner"] == "Yes")).astype(int)
    df["NewCustomer"] = (df["tenure"] <= 6).astype(int)
    df["HighCharge"] = (df["MonthlyCharges"] > HIGH_CHARGE_THRESHOLD).astype(int)

    df = pd.get_dummies(df, drop_first=True)

    # Align to the exact columns/order the model was trained on.
    # Any dummy column not present here (because this batch didn't contain
    # that category) gets filled with 0 — correct, since 0 means "not that category".
    df = df.reindex(columns=TRAINED_FEATURES, fill_value=0)
    return df


def predict_and_show(processed_df: pd.DataFrame, display_df: pd.DataFrame):
    scaled_input = scaler.transform(processed_df)
    predictions = model.predict(scaled_input)

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_input)
        confidence = np.max(probs, axis=1)

    if len(predictions) == 1:
        if predictions[0] == 1:
            st.error("⚠️ This customer is likely to churn")
        else:
            st.success("✅ This customer is likely to stay")
        if confidence is not None:
            st.write(f"**Model confidence:** {confidence[0]*100:.1f}%")
    else:
        result_df = display_df.copy()
        result_df["Prediction"] = ["Will Churn" if p == 1 else "Will Stay" for p in predictions]
        if confidence is not None:
            result_df["Confidence"] = [f"{c*100:.1f}%" for c in confidence]
        st.write("### Prediction Results")
        st.dataframe(result_df)
        n_churn = int((predictions == 1).sum())
        st.warning(f"{n_churn} out of {len(predictions)} customers predicted to churn.")
        st.download_button(
            "⬇️ Download Results as CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            "churn_predictions.csv",
            "text/csv",
        )


# ----------------------------------------------------
# Two modes: manual entry or CSV upload
# ----------------------------------------------------
mode = st.radio("How would you like to check?", ["Enter customer details", "Upload a CSV (batch)"], horizontal=True)

if mode == "Enter customer details":
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=5.0)
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, value=float(70.0 * 12), step=50.0,
                help="Typically Monthly Charges × Tenure, but use the real billed total if known."
            )

        submitted = st.form_submit_button("🔍 Predict Churn")

    if submitted:
        try:
            row = {
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
            raw_df = pd.DataFrame([row])
            processed = engineer_features(raw_df)
            predict_and_show(processed, raw_df)
        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")

else:
    st.caption(
        "Upload data in the same format as the original Telco churn dataset "
        "(customerID and Churn columns are optional and ignored if present)."
    )
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(raw_df.head())

        if st.button("🔍 Predict Churn"):
            try:
                data = raw_df.copy()
                if "Churn" in data.columns:
                    data = data.drop("Churn", axis=1)
                processed = engineer_features(data)
                predict_and_show(processed, raw_df)
            except Exception as e:
                st.error(f"Something went wrong during prediction: {e}")
    else:
        st.info("Upload a CSV file above to get started.")