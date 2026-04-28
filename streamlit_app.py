import streamlit as st
import pandas as pd
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="Customer Behaviour Analysis", layout="wide")

# --- Load Model and Columns ---
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    return model, columns

model, feature_columns = load_artifacts()

# --- App Title ---
st.title("Customer Behaviour Analysis")
st.markdown("Enter customer details to predict the likelihood of churn.")

# --- Input Form ---
with st.form(key="prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        gender = st.selectbox("Gender", ("Male", "Female"))
        location = st.text_input("Location")
        item_purchased = st.text_input("Item Purchased")
        category = st.selectbox("Category", ["Clothing", "Accessories", "Footwear", "Electronics", "Outerwear"])

    with col2:
        purchase_amount = st.number_input("Purchase Amount (USD)", min_value=0.0, step=0.01)
        shipping_type = st.text_input("Shipping Type")
        size = st.selectbox("Size", ["S", "M", "L", "XL"])
        color = st.text_input("Color")
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])

    with col3:
        review_rating = st.slider("Review Rating (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        discount_applied = st.selectbox("Discount Applied", ("No", "Yes"))
        previous_purchases = st.number_input("Previous Purchases", min_value=0, step=1)
        payment_method = st.text_input("Payment Method")
        frequency = st.number_input("Frequency of Purchases (Annual)", min_value=0, step=1)

    submitted = st.form_submit_button("Predict Customer Behaviour")

# --- Prediction and Display ---
if submitted:
    input_data = {
        'Age': age,
        'Gender': gender,
        'Item Purchased': item_purchased,
        'Category': category,
        'Purchase Amount (USD)': purchase_amount,
        'Location': location,
        'Size': size,
        'Color': color,
        'Season': season,
        'Review Rating': review_rating,
        'Shipping Type': shipping_type,
        'Discount Applied': discount_applied,
        'Previous Purchases': previous_purchases,
        'Payment Method': payment_method,
        'Frequency of Purchases': frequency
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align with training columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    prob = model.predict_proba(input_df)[0][1]
    prob_percent = prob * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    if prob >= 0.7:
        st.error(f"⚠️ There is a **{prob_percent:.1f}%** chance that this customer will NOT return.")
        st.warning("Consider sending a special offer or reaching out to retain them.")
    elif prob >= 0.4:
        st.warning(f"📊 There is a **{prob_percent:.1f}%** chance that this customer will NOT return.")
        st.info("Keep an eye on this customer. A small discount might help secure loyalty.")
    else:
        st.success(f"✅ There is a **{prob_percent:.1f}%** chance that this customer will NOT return.")
        st.success("This customer is likely to continue shopping. No action needed right now.")