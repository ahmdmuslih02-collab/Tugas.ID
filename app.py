import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD MODEL & FILE
# ===============================
dt_model = joblib.load("dt_model.pkl")
rf_model = joblib.load("rf_model.pkl")
encoders = joblib.load("encoders.pkl")
metrics = joblib.load("classification_metrics.pkl")

st.set_page_config(
    page_title="Supermarket Classification",
    page_icon="🛒",
    layout="centered"
)

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("📌 Menu")
page = st.sidebar.radio(
    "Pilih Model:",
    ["Decision Tree", "Random Forest"]
)

st.sidebar.markdown("---")
st.sidebar.write("📊 Akurasi Model")
st.sidebar.write(f"Decision Tree : {metrics['Decision Tree Accuracy']:.2f}")
st.sidebar.write(f"Random Forest : {metrics['Random Forest Accuracy']:.2f}")

# ===============================
# INPUT DATA FUNCTION
# ===============================
def user_input():
    st.subheader("🧾 Input Data Transaksi")

    branch = st.selectbox("Branch", encoders["Branch"].classes_)
    city = st.selectbox("City", encoders["City"].classes_)
    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    product_line = st.selectbox("Product Line", encoders["Product line"].classes_)
    payment = st.selectbox("Payment Method", encoders["Payment"].classes_)

    unit_price = st.number_input("Unit Price", min_value=0.0)
    quantity = st.number_input("Quantity", min_value=1)
    tax = st.number_input("Tax 5%", min_value=0.0)
    rating = st.slider("Rating", 1.0, 10.0)

    data = pd.DataFrame([{
        "Branch": encoders["Branch"].transform([branch])[0],
        "City": encoders["City"].transform([city])[0],
        "Gender": encoders["Gender"].transform([gender])[0],
        "Product line": encoders["Product line"].transform([product_line])[0],
        "Unit price": unit_price,
        "Quantity": quantity,
        "Tax 5%": tax,
        "Payment": encoders["Payment"].transform([payment])[0],
        "Rating": rating
    }])

    return data

# ===============================
# PAGE: DECISION TREE
# ===============================
if page == "Decision Tree":
    st.title("🌳 Decision Tree Classifier")
    st.write(
        "Model ini menggunakan **Decision Tree** "
        "untuk memprediksi **Customer Type**."
    )

    input_data = user_input()

    if st.button("🔮 Prediksi dengan Decision Tree"):
        prediction = dt_model.predict(input_data)[0]
        label = encoders["Customer type"].inverse_transform([prediction])[0]
        st.success(f"✅ Prediksi Customer Type: **{label}**")

# ===============================
# PAGE: RANDOM FOREST
# ===============================
elif page == "Random Forest":
    st.title("🌲 Random Forest Classifier")
    st.write(
        "Model ini menggunakan **Random Forest** "
        "untuk memprediksi **Customer Type**."
    )

    input_data = user_input()

    if st.button("🔮 Prediksi dengan Random Forest"):
        prediction = rf_model.predict(input_data)[0]
        label = encoders["Customer type"].inverse_transform([prediction])[0]
        st.success(f"✅ Prediksi Customer Type: **{label}**")
