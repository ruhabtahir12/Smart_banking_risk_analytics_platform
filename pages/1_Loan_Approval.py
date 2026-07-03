# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # ----------------------------------------------------
# # Page config
# # ----------------------------------------------------
# st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")

# st.title("💰 Loan Approval Prediction")
# st.write("Predict whether a loan application should be approved.")

# # ----------------------------------------------------
# # Paths — NOTE: your notebook saved the model as
# # "loan_risk_model.pkl", not "loan_approval_model.pkl".
# # Rename the file in models/ or update this path to match.
# # ----------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(BASE_DIR)
# MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "loan_risk_model.pkl")
# SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

# @st.cache_resource
# def load_artifacts():
#     model = joblib.load(MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     return model, scaler

# try:
#     model, scaler = load_artifacts()
# except FileNotFoundError as e:
#     st.error(f"Could not find model/scaler file: {e}")
#     st.stop()

# # ----------------------------------------------------
# # Exact column order used during training
# # (matches df.columns after dropping Loan_ID, before Loan_Status)
# # ----------------------------------------------------
# FEATURE_ORDER = [
#     "Gender", "Married", "Dependents", "Education", "Self_Employed",
#     "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
#     "Loan_Amount_Term", "Credit_History", "Property_Area"
# ]

# CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
# NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

# # ----------------------------------------------------
# # Hardcoded LabelEncoder mappings (alphabetical order —
# # matches sklearn's default LabelEncoder behavior and your
# # notebook's own new_data test values). If your uploaded CSV
# # contains category values outside these, prediction will fail
# # and you'll need to add them here.
# # ----------------------------------------------------
# ENCODING_MAPS = {
#     "Gender": {"Female": 0, "Male": 1},
#     "Married": {"No": 0, "Yes": 1},
#     "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
#     "Education": {"Graduate": 0, "Not Graduate": 1},
#     "Self_Employed": {"No": 0, "Yes": 1},
#     "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
# }


# def preprocess(df: pd.DataFrame) -> pd.DataFrame:
#     data = df.copy()

#     if "Loan_ID" in data.columns:
#         data = data.drop("Loan_ID", axis=1)
#     if "Loan_Status" in data.columns:
#         data = data.drop("Loan_Status", axis=1)

#     missing_cols = [c for c in FEATURE_ORDER if c not in data.columns]
#     if missing_cols:
#         raise ValueError(f"Uploaded CSV is missing required column(s): {missing_cols}")

#     # Impute missing values the same way training did:
#     # mode for categorical, median for numeric
#     for col in CATEGORICAL_COLS:
#         data[col] = data[col].astype(str)
#         if data[col].isnull().any() or (data[col] == "nan").any():
#             mode_val = data[col].mode()[0]
#             data[col] = data[col].replace("nan", mode_val)
#             data[col] = data[col].fillna(mode_val)

#     for col in NUMERIC_COLS:
#         if data[col].isnull().any():
#             data[col] = data[col].fillna(data[col].median())

#     # Encode categoricals using the fixed mappings
#     unseen_values = {}
#     for col in CATEGORICAL_COLS:
#         mapping = ENCODING_MAPS[col]
#         unseen = set(data[col].unique()) - set(mapping.keys())
#         if unseen:
#             unseen_values[col] = unseen
#         data[col] = data[col].map(mapping)

#     if unseen_values:
#         raise ValueError(
#             f"Found category values not seen during training: {unseen_values}. "
#             f"Cannot encode these reliably."
#         )

#     # Reorder columns to match training order exactly
#     data = data[FEATURE_ORDER]
#     return data


# # ----------------------------------------------------
# # File uploader
# # ----------------------------------------------------
# uploaded_file = st.file_uploader("Upload Loan Application Data (CSV)", type="csv")

# if uploaded_file is not None:
#     raw_df = pd.read_csv(uploaded_file)
#     st.write("### Data Preview")
#     st.dataframe(raw_df.head())

#     if st.button("🔍 Predict Loan Approval"):
#         try:
#             processed = preprocess(raw_df)
#             scaled_input = scaler.transform(processed)

#             predictions = model.predict(scaled_input)
#             # Loan_Status was label-encoded too: N=0, Y=1 (alphabetical)
#             labels = ["✅ Approved" if p == 1 else "❌ Rejected" for p in predictions]

#             confidence = None
#             if hasattr(model, "predict_proba"):
#                 probs = model.predict_proba(scaled_input)
#                 confidence = np.max(probs, axis=1)

#             result_df = raw_df.copy()
#             result_df["Prediction"] = labels
#             if confidence is not None:
#                 result_df["Confidence"] = [f"{c*100:.1f}%" for c in confidence]

#             st.write("### Prediction Results")
#             st.dataframe(result_df)

#             st.download_button(
#                 "⬇️ Download Results as CSV",
#                 result_df.to_csv(index=False).encode("utf-8"),
#                 "loan_predictions.csv",
#                 "text/csv",
#             )

#         except ValueError as ve:
#             st.error(str(ve))
#         except Exception as e:
#             st.error(f"Something went wrong during prediction: {e}")
# else:
#     st.info("Upload a CSV file above to get started. Expected columns: "
#              + ", ".join(FEATURE_ORDER))





import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")

st.title("💰 Loan Approval Prediction")
st.write("Enter your details below to check loan approval likelihood.")

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "loan_risk_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Could not find model/scaler file: {e}")
    st.stop()

# ----------------------------------------------------
# Exact column order used during training
# ----------------------------------------------------
FEATURE_ORDER = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]

ENCODING_MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "Married": {"No": 0, "Yes": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}


def encode_row(row_dict: dict) -> pd.DataFrame:
    """Take a single applicant's raw inputs and return a model-ready 1-row DataFrame."""
    data = {}
    for col in FEATURE_ORDER:
        val = row_dict[col]
        if col in ENCODING_MAPS:
            data[col] = ENCODING_MAPS[col][val]
        else:
            data[col] = val
    return pd.DataFrame([data], columns=FEATURE_ORDER)


def predict_and_show(input_df: pd.DataFrame, display_df: pd.DataFrame):
    scaled_input = scaler.transform(input_df)
    predictions = model.predict(scaled_input)

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_input)
        confidence = np.max(probs, axis=1)

    labels = ["Approved" if p == 1 else "Rejected" for p in predictions]

    if len(predictions) == 1:
        # Single applicant — show a friendly result card instead of a table
        if labels[0] == "Approved":
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")
        if confidence is not None:
            st.write(f"**Model confidence:** {confidence[0]*100:.1f}%")
    else:
        result_df = display_df.copy()
        result_df["Prediction"] = labels
        if confidence is not None:
            result_df["Confidence"] = [f"{c*100:.1f}%" for c in confidence]
        st.write("### Prediction Results")
        st.dataframe(result_df)
        st.download_button(
            "⬇️ Download Results as CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            "loan_predictions.csv",
            "text/csv",
        )


# ----------------------------------------------------
# Two modes: manual entry (for public users) or CSV upload (for batch/testing)
# ----------------------------------------------------
mode = st.radio("How would you like to check?", ["Enter my details", "Upload a CSV (batch)"], horizontal=True)

if mode == "Enter my details":
    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col2:
            applicant_income = st.number_input("Applicant Monthly Income", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Co-applicant Monthly Income", min_value=0, value=0, step=500)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150, step=10)
            loan_term = st.selectbox("Loan Term (months)", [360, 180, 240, 120, 84, 60, 36, 12])
            credit_history = st.selectbox("Credit History", ["Good (has repaid past debts)", "Poor / No history"])

        submitted = st.form_submit_button("🔍 Check Approval")

    if submitted:
        try:
            row = {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": 1 if credit_history.startswith("Good") else 0,
                "Property_Area": property_area,
            }
            input_df = encode_row(row)
            predict_and_show(input_df, input_df)
        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")

else:
    st.caption(
        "Expected columns: " + ", ".join(FEATURE_ORDER)
        + " (Loan_ID and Loan_Status are optional and ignored if present)"
    )
    uploaded_file = st.file_uploader("Upload Loan Application Data (CSV)", type="csv")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(raw_df.head())

        if st.button("🔍 Predict Loan Approval"):
            try:
                data = raw_df.copy()
                if "Loan_ID" in data.columns:
                    data = data.drop("Loan_ID", axis=1)
                if "Loan_Status" in data.columns:
                    data = data.drop("Loan_Status", axis=1)

                missing_cols = [c for c in FEATURE_ORDER if c not in data.columns]
                if missing_cols:
                    st.error(f"Uploaded CSV is missing required column(s): {missing_cols}")
                    st.stop()

                categorical_cols = list(ENCODING_MAPS.keys())
                numeric_cols = [c for c in FEATURE_ORDER if c not in categorical_cols]

                for col in categorical_cols:
                    data[col] = data[col].astype(str)
                    if (data[col] == "nan").any():
                        mode_val = data[col].mode()[0]
                        data[col] = data[col].replace("nan", mode_val)

                for col in numeric_cols:
                    if data[col].isnull().any():
                        data[col] = data[col].fillna(data[col].median())

                unseen_values = {}
                for col in categorical_cols:
                    mapping = ENCODING_MAPS[col]
                    unseen = set(data[col].unique()) - set(mapping.keys())
                    if unseen:
                        unseen_values[col] = unseen
                    data[col] = data[col].map(mapping)

                if unseen_values:
                    st.error(f"Found category values not seen during training: {unseen_values}")
                    st.stop()

                data = data[FEATURE_ORDER]
                predict_and_show(data, raw_df)

            except Exception as e:
                st.error(f"Something went wrong during prediction: {e}")
    else:
        st.info("Upload a CSV file above to get started.")