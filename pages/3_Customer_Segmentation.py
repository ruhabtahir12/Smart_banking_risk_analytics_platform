



import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", page_icon="📊")

st.title("📊 Customer Segmentation")
st.write("Group a customer into one of the discovered spending-behavior segments.")

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "customer_segmentation_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "segmentation_scaler.pkl")

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
# Exact feature order used during training
# ----------------------------------------------------
FEATURE_ORDER = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
    "PAYMENTS", "MINIMUM_PAYMENTS", "TENURE"
]

# ----------------------------------------------------
# Auto-generate human-readable cluster labels from the model itself.
# KMeans stores each cluster's center in SCALED space — we inverse
# transform back to real BALANCE/PURCHASES/etc values, then rank
# clusters against each other on key traits to build a label.
# No manual notebook lookup needed.
# ----------------------------------------------------
@st.cache_data
def build_cluster_labels(_model, _scaler):
    centers_scaled = _model.cluster_centers_
    centers_real = _scaler.inverse_transform(centers_scaled)
    profile = pd.DataFrame(centers_real, columns=FEATURE_ORDER)

    labels = {}
    for cluster_id in profile.index:
        row = profile.loc[cluster_id]

        # Rank this cluster's value for each trait against the other clusters
        balance_rank = profile["BALANCE"].rank(ascending=False)[cluster_id]
        purchases_rank = profile["PURCHASES"].rank(ascending=False)[cluster_id]
        cash_rank = profile["CASH_ADVANCE"].rank(ascending=False)[cluster_id]
        limit_rank = profile["CREDIT_LIMIT"].rank(ascending=False)[cluster_id]
        n = len(profile)

        tags = []
        if purchases_rank <= n / 2 and balance_rank <= n / 2:
            tags.append("High Value")
        if cash_rank == 1:
            tags.append("Cash-Advance Heavy")
        if purchases_rank == n and cash_rank > n / 2:
            tags.append("Low Activity")
        if limit_rank == 1:
            tags.append("Premium Credit Limit")
        if not tags:
            tags.append("Moderate Spender")

        labels[cluster_id] = " / ".join(tags)

    return labels, profile


CLUSTER_LABELS, CLUSTER_PROFILE = build_cluster_labels(model, scaler)


def predict_and_show(input_df: pd.DataFrame, display_df: pd.DataFrame):
    scaled_input = scaler.transform(input_df)
    clusters = model.predict(scaled_input)

    if len(clusters) == 1:
        label = CLUSTER_LABELS.get(clusters[0], f"Segment {clusters[0]}")
        st.success(f"📊 This customer belongs to: **{label}**")
        with st.expander("See what defines this segment"):
            st.write(
                "Average profile of customers in this segment (from the trained model):"
            )
            st.dataframe(CLUSTER_PROFILE.loc[[clusters[0]]])
    else:
        result_df = display_df.copy()
        result_df["Segment"] = [CLUSTER_LABELS.get(c, f"Segment {c}") for c in clusters]
        st.write("### Segmentation Results")
        st.dataframe(result_df)

        st.write("### Segment Distribution")
        st.bar_chart(result_df["Segment"].value_counts())

        st.download_button(
            "⬇️ Download Results as CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            "segmentation_results.csv",
            "text/csv",
        )


st.divider()
with st.expander("📋 View all 4 customer segments"):
    st.write(
        "Here's the average profile of each segment, based on your trained model. "
        "Use this to understand what distinguishes each group."
    )
    overview = CLUSTER_PROFILE.copy()
    overview.insert(0, "Segment Label", [CLUSTER_LABELS[i] for i in overview.index])
    styled_overview = overview.style.format(
        {col: "{:,.0f}" for col in FEATURE_ORDER if col != "TENURE"}
    ).format({"TENURE": "{:,.1f}"})
    st.dataframe(styled_overview)

st.divider()


mode = st.radio("How would you like to check?", ["Enter customer details", "Upload a CSV (batch)"], horizontal=True)

if mode == "Enter customer details":
    with st.form("segmentation_form"):
        st.caption(
            "💡 Enter actual account amounts (e.g. a balance of $5,000 → type 5000). "
            "These are raw currency values, not in thousands."
        )
        col1, col2 = st.columns(2)

        with col1:
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=1500.0, step=100.0, format="%.2f")
            purchases = st.number_input("Total Purchases ($)", min_value=0.0, value=1000.0, step=50.0, format="%.2f")
            cash_advance = st.number_input("Cash Advance Amount ($)", min_value=0.0, value=0.0, step=50.0, format="%.2f")
            credit_limit = st.number_input("Credit Limit ($)", min_value=0.0, value=5000.0, step=500.0, format="%.2f")

        with col2:
            payments = st.number_input("Total Payments Made ($)", min_value=0.0, value=1200.0, step=50.0, format="%.2f")
            minimum_payments = st.number_input("Minimum Payments Due ($)", min_value=0.0, value=300.0, step=25.0, format="%.2f")
            tenure = st.slider("Tenure (months as customer)", min_value=1, max_value=12, value=12)

        submitted = st.form_submit_button("🔍 Find Segment")

    if submitted:
        try:
            row = {
                "BALANCE": balance,
                "PURCHASES": purchases,
                "CASH_ADVANCE": cash_advance,
                "CREDIT_LIMIT": credit_limit,
                "PAYMENTS": payments,
                "MINIMUM_PAYMENTS": minimum_payments,
                "TENURE": tenure,
            }
            input_df = pd.DataFrame([row], columns=FEATURE_ORDER)
            predict_and_show(input_df, input_df)
        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")

else:
    st.caption("Expected columns: " + ", ".join(FEATURE_ORDER))
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(raw_df.head())

        if st.button("🔍 Segment Customers"):
            try:
                data = raw_df.copy()
                if "CUST_ID" in data.columns:
                    data = data.drop("CUST_ID", axis=1)

                missing_cols = [c for c in FEATURE_ORDER if c not in data.columns]
                if missing_cols:
                    st.error(f"Uploaded CSV is missing required column(s): {missing_cols}")
                    st.stop()

                data = data[FEATURE_ORDER].copy()

                # Training filled missing values with column mean
                if data.isnull().any().any():
                    data = data.fillna(data.mean())

                predict_and_show(data, raw_df)

            except Exception as e:
                st.error(f"Something went wrong during prediction: {e}")
    else:
        st.info("Upload a CSV file above to get started.")