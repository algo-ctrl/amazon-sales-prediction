# ============================================================
# AMAZON SALES PREDICTION STREAMLIT APP  ✅ FIXED VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# Load model and preprocessors
# --------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('amazon_sales_predictor.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Amazon Sales Predictor", layout="centered")
st.title("🛒 Amazon Sales Prediction App")
st.write("Predict total sales amount for Amazon orders using machine learning.")

st.markdown("### Enter Order Details")

col1, col2 = st.columns(2)

with col1:
    product = st.selectbox("Product", label_encoders['Product'].classes_)
    category = st.selectbox("Category", label_encoders['Category'].classes_)
    customer_location = st.selectbox("Customer Location", label_encoders['Customer Location'].classes_)
    payment_method = st.selectbox("Payment Method", label_encoders['Payment Method'].classes_)
    status = st.selectbox("Order Status", label_encoders['Status'].classes_)

with col2:
    quantity = st.number_input("Quantity Ordered", min_value=1, step=1)
    price_per_unit = st.number_input("Price per Unit ($)", min_value=1.0, step=0.5)
    month = st.selectbox("Month", list(range(1, 13)))
    day = st.slider("Day of Month", 1, 31, 15)
    weekday = st.selectbox("Weekday (0=Mon, 6=Sun)", list(range(0, 7)))

# --------------------------
# Encode categorical inputs
# --------------------------
def encode_input(col, value):
    encoder = label_encoders[col]
    return encoder.transform([value])[0]

# --------------------------
# Predict Button
# --------------------------
if st.button("🔮 Predict Total Sales"):
    try:
        # Encode categorical variables
        encoded_features = [
            encode_input('Product', product),
            encode_input('Category', category),
            encode_input('Customer Location', customer_location),
            encode_input('Payment Method', payment_method),
            encode_input('Status', status),
        ]

        # Numeric inputs (removed discount and shipping cost ✅)
        numeric_features = [
            quantity, price_per_unit, day, month, weekday
        ]

        # Combine all features (total = 10 ✅)
        final_features = np.array(encoded_features + numeric_features).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(final_features)

        # Predict
        prediction = model.predict(scaled_features)[0]

        st.success(f"💰 **Predicted Total Sales: ${prediction:,.2f}**")
        st.balloons()

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Developed by Saksham Kumar Sharma | Amazon Sales Prediction Mini Project")
