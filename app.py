from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "customer_churn_ann.keras"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.joblib"


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Churn.AI | Customer Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 3. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(120deg, #172554, #1e40af);
    padding: 32px;
    border-radius: 18px;
    border: 1px solid #3156a5;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.section-title {
    color: #93c5fd;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

div[data-testid="stForm"] {
    background-color: #111c30;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #263449;
}

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 12px;
    min-height: 48px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #1d4ed8;
    color: white;
    border: none;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #1e293b;
    border-color: #334155;
}

label {
    color: #cbd5e1 !important;
}

.result-card {
    background-color: #111c30;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 25px;
    margin-top: 20px;
}

.small-note {
    color: #94a3b8;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 🤖 Churn.AI")

    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.divider()

    st.markdown("### 🧠 Model Information")

    st.write("**Algorithm:** Artificial Neural Network")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Task:** Binary Classification")

    st.divider()

    st.markdown("### 📊 Project Information")

    st.write("Customer Churn Prediction")
    st.write("Educational Deep Learning Project")

    st.divider()

    st.caption(
        "Built with Python, TensorFlow, "
        "Pandas and Streamlit."
    )


# --------------------------------------------------
# 5. LOAD MODEL AND PREPROCESSOR
# --------------------------------------------------

if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():

    st.error(
        "Model files are missing. "
        "Please run python train.py first."
    )

    st.stop()


@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return model, preprocessor


model, preprocessor = load_artifacts()


# --------------------------------------------------
# 6. MAIN HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">

    <h1>Customer Churn Prediction</h1>

    <p>
        Predict customer retention risk using
        an Artificial Neural Network.
    </p>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 7. DASHBOARD METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model",
        value="ANN"
    )

with col2:
    st.metric(
        label="Framework",
        value="TensorFlow"
    )

with col3:
    st.metric(
        label="Problem Type",
        value="Classification"
    )


st.divider()


# --------------------------------------------------
# 8. CUSTOMER INPUT FORM
# --------------------------------------------------

st.markdown(
    '<p class="section-title">👤 Customer Information</p>',
    unsafe_allow_html=True
)

st.write(
    "Enter the customer's information below "
    "to estimate the probability of churn."
)


with st.form("customer_prediction_form"):

    st.markdown("### 📌 Account Information")

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.slider(
            "Tenure (months)",
            min_value=1,
            max_value=72,
            value=24
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=18.0,
            max_value=125.0,
            value=70.0,
            step=1.0
        )

        calls = st.slider(
            "Support Calls",
            min_value=0,
            max_value=8,
            value=1
        )

    with col2:

        contract = st.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        internet = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Predict Customer Churn",
        use_container_width=True
    )


# --------------------------------------------------
# 9. PREDICTION
# --------------------------------------------------

if submitted:

    row = pd.DataFrame([{

        "tenure_months": tenure,

        "monthly_charges": monthly,

        "total_charges": monthly * tenure,

        "contract_type": contract,

        "internet_service": internet,

        "support_calls": calls,

        "paperless_billing": paperless

    }])

    try:

        x = preprocessor.transform(row)

        x = x.astype("float32")

        probability = float(
            model.predict(x, verbose=0)[0][0]
        )

        st.divider()

        st.markdown(
            '<p class="section-title">📊 Prediction Results</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Estimated Churn Probability",
                f"{probability:.1%}"
            )

        with result_col2:

            st.metric(
                "Prediction Confidence",
                f"{(1 - probability if probability < 0.5 else probability):.1%}"
            )

        st.progress(probability)

        if probability >= 0.5:

            st.error(
                "⚠️ High Churn Risk: "
                "The model predicts this customer is likely to churn."
            )

        else:

            st.success(
                "✅ Low Churn Risk: "
                "The model predicts this customer is unlikely to churn."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "This is an educational demonstration using "
            "synthetic customer data. Predictions are not "
            "validated for real-world business decisions."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8;">

        <p>🤖 Churn.AI | Deep Learning Project</p>

        <p>Built using Python · TensorFlow · Keras · Streamlit</p>

    </div>
    """,
    unsafe_allow_html=True
)
