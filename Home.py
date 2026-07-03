import streamlit as st

st.set_page_config(
    page_title="Smart Banking Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.hero {
    background: linear-gradient(135deg, #0f2542 0%, #1a3d6d 60%, #2c5aa0 100%);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2.5rem;
    box-shadow: 0 10px 30px rgba(15, 37, 66, 0.25);
}
.hero h1 {
    color: white;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.6rem;
    letter-spacing: -0.5px;
}
.hero p {
    color: #cfe0f5;
    font-size: 1.15rem;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.6rem;
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 1.4rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255,255,255,0.12);
    color: #eaf2fb;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.85rem;
    border: 1px solid rgba(255,255,255,0.25);
}

.module-card {
    background: white;
    border: 1px solid #e8edf3;
    border-radius: 16px;
    padding: 1.6rem;
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border-top: 4px solid var(--accent);
}
.module-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(15, 37, 66, 0.12);
}
.module-icon {
    font-size: 2.4rem;
    margin-bottom: 0.5rem;
}
.module-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f2542;
    margin-bottom: 0.4rem;
}
.module-desc {
    color: #5a6b7d;
    font-size: 0.93rem;
    line-height: 1.45rem;
    margin-bottom: 0.8rem;
}
.module-tag {
    display: inline-block;
    background: #f0f4fa;
    color: #2c5aa0;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 10px;
    margin-bottom: 0.8rem;
}

.footer-note {
    text-align: center;
    color: #8a97a6;
    font-size: 0.85rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #eef1f5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
    <h1>🏦 Smart Banking Analytics</h1>
    <p>An AI-powered suite for smarter lending, fraud prevention, customer insight,
    and retention — built on four trained machine learning models.</p>
    <div class="badge-row">
        <span class="badge">💰 Loan Risk</span>
        <span class="badge">🕵️ Fraud Detection</span>
        <span class="badge">📊 Segmentation</span>
        <span class="badge">📉 Churn</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Modules ----------------
modules = [
    {
        "icon": "💰", "accent": "#2c5aa0", "tag": "CLASSIFICATION MODEL",
        "title": "Loan Approval Prediction",
        "desc": "Instantly check the likelihood of loan approval based on applicant income, credit history, and demographics.",
        "page": "pages/1_Loan_Approval.py",
    },
    {
        "icon": "🕵️", "accent": "#c0392b", "tag": "ANOMALY DETECTION",
        "title": "Fraud Detection",
        "desc": "Scan transaction data in real time to flag potentially fraudulent credit card activity.",
        "page": "pages/2_Fraud_Detection.py",
    },
    {
        "icon": "📊", "accent": "#27808d", "tag": "CLUSTERING MODEL",
        "title": "Customer Segmentation",
        "desc": "Group customers into behavioral segments using spending, balance, and credit usage patterns.",
        "page": "pages/3_Customer_Segmentation.py",
    },
    {
        "icon": "📉", "accent": "#8e44ad", "tag": "PREDICTIVE MODEL",
        "title": "Churn Prediction",
        "desc": "Identify customers likely to leave based on billing, contract type, and service usage.",
        "page": "pages/4_Churn_Prediction.py",
    },
]

col1, col2 = st.columns(2, gap="large")
cols = [col1, col2]

for i, mod in enumerate(modules):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="module-card" style="--accent: {mod['accent']}">
            <div class="module-icon">{mod['icon']}</div>
            <div class="module-tag">{mod['tag']}</div>
            <div class="module-title">{mod['title']}</div>
            <div class="module-desc">{mod['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(mod["page"], label=f"Open {mod['title']}  →")
        st.write("")

st.markdown("""
<div class="footer-note">
    Smart Banking Analytics · Built with Streamlit &amp; scikit-learn
</div>
""", unsafe_allow_html=True)