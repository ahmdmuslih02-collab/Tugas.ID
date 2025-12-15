import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD MODEL & FILE PENDUKUNG
# ===============================
dt_model = joblib.load("dt_model.pkl")
rf_model = joblib.load("rf_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Supermarket Classification",
    layout="centered"
)

st.title("🛒 Supermarket Customer Classification")

st.markdown("""
Aplikasi ini membandingkan **dua model klasifikasi**:
- 🌳 **Decision Tree**
- 🌲 **Random Forest**

🎯 **Target prediksi:** `Customer type` → **Member / Normal**
""")

# ===============================
# PILIH MODEL (SIDEBAR)
# ===============================
model_choice = st.sidebar.radio(
    "📌 Pilih Model Klasifikasi",
    ("Decision Tree", "Random Forest")
)

# ===============================
# INPUT USER
# ===============================
st.subheader("🧾 Input Data Transaksi")

branch = st.selectbox("Branch", encoders["Branch"].classes_)
city = st.selectbox("City", encoders["City"].classes_)
gender = st.selectbox("Gender", encoders["Gender"].classes_)
product_line = st.selectbox("Product Line", encoders["Product line"].classes_)
payment = st.selectbox("Payment Method", encoders["Payment"].classes_)

unit_price = st.number_input("Unit Price", min_value=0.0, step=0.1)
quantity = st.number_input("Quantity", min_value=1, step=1)
tax = st.number_input("Tax 5%", min_value=0.0, step=0.1)
cogs = st.number_input("COGS", min_value=0.0, step=0.1)
gross_income = st.number_input("Gross Income", min_value=0.0, step=0.1)
rating = st.slider("Rating", 1.0, 10.0, 5.0)

# ===============================
# ENCODE INPUT
# ===============================
input_dict = {
    "Branch": encoders["Branch"].transform([branch])[0],
    "City": encoders["City"].transform([city])[0],
    "Gender": encoders["Gender"].transform([gender])[0],
    "Product line": encoders["Product line"].transform([product_line])[0],
    "Unit price": unit_price,
    "Quantity": quantity,
    "Tax 5%": tax,
    "Payment": encoders["Payment"].transform([payment])[0],
    "cogs": cogs,
    "gross income": gross_income,
    "Rating": rating
}

input_data = pd.DataFrame([input_dict])

# ===============================
# 🔐 FIX UTAMA (ANTI ERROR)
# ===============================
# Pastikan urutan & jumlah fitur SAMA dengan saat training
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# ===============================
# PREDIKSI
# ===============================
if st.button("🔮 Prediksi Customer Type"):
    if model_choice == "Decision Tree":
        prediction = dt_model.predict(input_data)[0]
        st.success(f"🌳 Decision Tree → **{prediction.upper()}**")

    else:
        prediction = rf_model.predict(input_data)[0]
        st.success(f"🌲 Random Forest → **{prediction.upper()}**")

# ===============================
# INFORMASI TAMBAHAN
# ===============================
st.markdown("""
---
📌 **Catatan Teknis**
- Model dilatih menggunakan data supermarket
- Semua fitur telah disamakan urutannya dengan data training
- Random Forest umumnya lebih stabil dibanding Decision Tree
""")
