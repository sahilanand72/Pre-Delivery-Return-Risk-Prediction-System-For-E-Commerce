import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Product Return Risk Predictor",
    layout="centered"
)

st.title("📦 Product Return Risk Prediction")
st.write(
    "Predict whether an e-commerce order is likely to be **returned** "
    "using only order-time information."
)

# -------------------------------
# Load Model & Encoders
# -------------------------------
MODEL_PATH = "models/return_risk_model.pkl"
ENCODER_PATH = "models/label_encoders.pkl"

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# -------------------------------
# User Inputs
# -------------------------------
st.header("🧾 Enter Order Details")

Product_Category = st.selectbox(
    "Product Category",
    options=label_encoders["Product_Category"].classes_
)

Product_Price = st.number_input(
    "Product Price",
    min_value=0.0,
    step=1.0
)

Order_Quantity = st.number_input(
    "Order Quantity",
    min_value=1,
    step=1
)

Discount_Applied = st.slider(
    "Discount Applied (%)",
    min_value=0,
    max_value=90,
    value=10
)

Payment_Method = st.selectbox(
    "Payment Method",
    options=label_encoders["Payment_Method"].classes_
)

Shipping_Method = st.selectbox(
    "Shipping Method",
    options=label_encoders["Shipping_Method"].classes_
)

User_Age = st.number_input(
    "User Age",
    min_value=10,
    max_value=100,
    value=30
)

User_Gender = st.selectbox(
    "User Gender",
    options=label_encoders["User_Gender"].classes_
)

User_Location = st.selectbox(
    "User Location",
    options=label_encoders["User_Location"].classes_
)

# -------------------------------
# Prepare Input DataFrame
# -------------------------------
input_df = pd.DataFrame([{
    "Product_Category": Product_Category,
    "Product_Price": Product_Price,
    "Order_Quantity": Order_Quantity,
    "Discount_Applied": Discount_Applied,
    "Payment_Method": Payment_Method,
    "Shipping_Method": Shipping_Method,
    "User_Age": User_Age,
    "User_Gender": User_Gender,
    "User_Location": User_Location
}])

# -------------------------------
# Encode Inputs
# -------------------------------
for col, le in label_encoders.items():
    input_df[col] = le.transform(input_df[col])

# -------------------------------
# Predict Probability (Always)
# -------------------------------
probability = model.predict_proba(input_df)[0][1]

# -------------------------------
# Threshold Slider
# -------------------------------
st.subheader("🎚 Decision Threshold")

threshold = st.slider(
    "Return Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

prediction = int(probability >= threshold)

# -------------------------------
# Show Result on Button Click
# -------------------------------
if st.button("🔍 Predict Return Risk"):
    st.subheader("📊 Prediction Result")

    st.write(f"**Return Probability:** {probability:.2f}")
    st.write(f"**Selected Threshold:** {threshold}")

    if prediction == 1:
        st.error("⚠️ High Return Risk — Order likely to be returned")
    else:
        st.success("✅ Low Return Risk — Order likely to be kept")
