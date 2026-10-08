import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding-bottom: 1rem;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1.8rem;
        border-radius: 12px;
        color: #FFFFFF;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        margin: 1.5rem 0;
    }
    .prediction-title {
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        opacity: 0.9;
        margin-bottom: 0.3rem;
    }
    .prediction-price {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .prediction-inr {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.95;
    }
    .footer-note {
        text-align: center;
        color: #6B7280;
        font-size: 0.88rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    div.stButton > button:first-child {
        background-color: #1E3A8A;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.65rem 1.5rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        box-shadow: 0 2px 6px rgba(30, 58, 138, 0.25);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563EB;
        color: #FFFFFF;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_trained_model():
    model_filename = "car_price_random_forest.pkl"
    # Look in the script directory first, then fallback to current working directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    primary_path = os.path.join(base_dir, model_filename)

    if os.path.exists(primary_path):
        return joblib.load(primary_path)
    elif os.path.exists(model_filename):
        return joblib.load(model_filename)
    else:
        raise FileNotFoundError(f"Model file '{model_filename}' not found.")

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🚗 Car Price Prediction</div>
    <div class="sub-title">Machine Learning based used car valuation using Random Forest Regression</div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
current_year = datetime.now().year

st.subheader("📋 Enter Vehicle Details")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=current_year,
            value=2015,
            step=1,
            help="Year the car was manufactured (e.g., 2015)"
        )

        present_price = st.number_input(
            "Present Price (in Lakhs ₹)",
            min_value=0.0,
            max_value=100.0,
            value=5.50,
            step=0.10,
            format="%.2f",
            help="Current showroom price in Lakhs (e.g., 5.50 for ₹5,50,000)"
        )

        kms_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=27000,
            step=1000,
            help="Total kilometers the car has been driven"
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            options=["Petrol", "Diesel", "CNG"],
            index=0,
            help="Type of fuel the vehicle runs on"
        )

    with col2:
        seller_type = st.selectbox(
            "Seller Type",
            options=["Dealer", "Individual"],
            index=0,
            help="Whether sold by a commercial dealer or individual owner"
        )

        transmission = st.selectbox(
            "Transmission",
            options=["Manual", "Automatic"],
            index=0,
            help="Gear transmission mechanism"
        )

        owner = st.selectbox(
            "Number of Previous Owners",
            options=[0, 1, 2, 3],
            index=0,
            help="Count of previous registered owners (0 for first-hand)"
        )

    predict_btn = st.button("🔍 Predict Selling Price", use_container_width=True)


# --------------------------------------------------
# VALIDATION & PREDICTION
# --------------------------------------------------
if predict_btn:
    # Basic input validation
    validation_passed = True

    if present_price <= 0.0:
        st.error("⚠️ Present Price must be greater than 0.")
        validation_passed = False

    if kms_driven < 0:
        st.error("⚠️ Kilometers Driven cannot be negative.")
        validation_passed = False

    if year > current_year:
        st.error(f"⚠️ Manufacturing Year cannot be in the future (greater than {current_year}).")
        validation_passed = False

    if validation_passed:
        # Create DataFrame with exact feature names expected by pipeline
        input_data = pd.DataFrame([{
            "Year": int(year),
            "Present_Price": float(present_price),
            "Kms_Driven": int(kms_driven),
            "Fuel_Type": fuel_type,
            "Seller_Type": seller_type,
            "Transmission": transmission,
            "Owner": int(owner)
        }])

        try:
            # Predict using existing model pipeline
            raw_prediction = model.predict(input_data)[0]
            # Ensure price is non-negative
            predicted_price = max(0.0, float(raw_prediction))
            predicted_price_inr = predicted_price * 100000

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">Estimated Selling Price</div>
                <div class="prediction-price">₹ {predicted_price:.2f} Lakhs</div>
                <div class="prediction-inr">(Approximately ₹ {predicted_price_inr:,.0f})</div>
            </div>
            """, unsafe_allow_html=True)

            # Feature summary expander for project presentation
            with st.expander("📊 View Input Feature Summary"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")


# --------------------------------------------------
# FOOTER NOTE
# --------------------------------------------------
st.markdown("""
<div class="footer-note">
    ℹ️ <em>Note: This is an estimated selling price generated using a Random Forest regression model trained on historical automotive market data.</em>
</div>
""", unsafe_allow_html=True)
