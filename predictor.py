import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# --- Load XGBoost Model ---
# Ensure your saved model file is named 'XGB.pkl'
model = joblib.load('XGB.pkl')

# Load test data for SHAP (Ensure X_test.csv matches the new feature structure)
X_test = pd.read_csv('X_test.csv')

# Updated Feature Names (Must match the trained XGBoost model exactly)
feature_names = ["opioid", "NRS", "age", "NMR", "ApoB"]

# ------------------- Streamlit UI -------------------
st.title("PRF Efficacy Predictor for Zoster-Associated Pain")
st.markdown("---")

### Input Section
st.subheader("Patient Clinical Data")

# --- UI Layout Changed: All inputs in a single column ---
opioid = st.selectbox("Opioid use:", options=[0, 1], format_func=lambda x: "User" if x == 1 else "Non-user")
NRS = st.slider("Pain Score (NRS):", min_value=0, max_value=10, value=5)
age = st.slider("Age:", min_value=0, max_value=120, value=50)
ApoB = st.number_input("ApoB (g/L):", min_value=0.0, max_value=5.0, value=0.8, format="%.2f")

# Inputs for NMR calculation
neutrophil = st.number_input("Neutrophil Count (10^9/L):", min_value=0.01, max_value=50.0, value=4.0, format="%.2f")
monocyte = st.number_input("Monocyte Count (10^9/L):", min_value=0.01, max_value=10.0, value=0.5, format="%.2f")

# ------------------- Background Logic -------------------
# 1. Binary transformations
nrs_binary = 1 if NRS > 6 else 0
age_binary = 1 if age > 60 else 0

# 2. NMR Calculation
nmr_value = neutrophil / monocyte

# 3. Final Feature Vector [opioid, NRS_bin, age_bin, NMR, ApoB]
feature_values = [opioid, nrs_binary, age_binary, nmr_value, ApoB]
features_df = pd.DataFrame([feature_values], columns=feature_names)

# ------------------- Prediction -------------------
st.markdown("---")
if st.button("Predict Efficacy"):
    # XGBoost Prediction
    predicted_class = model.predict(features_df)[0]
    predicted_proba = model.predict_proba(features_df)[0]

    # Result Display
    result_color = "green" if predicted_class == 1 else "red"
    st.markdown(f"### Predicted Outcome: <span style='color:{result_color}'>{'Positive' if predicted_class == 1 else 'Negative'}</span>", unsafe_allow_html=True)
    
    prob_val = predicted_proba[1] * 100
    st.write(f"**Probability of Positive Response:** {prob_val:.1f}%")

    # Clinical Advice
    if predicted_class == 1:
        st.success(f"Recommendation: Proceed with PRF. The patient has a high likelihood ({prob_val:.1f}%) of pain relief.")
    else:
        st.warning(f"Recommendation: Consider adjusting treatment. Likelihood of PRF efficacy is low ({prob_val:.1f}%).")

    # Display calculated NMR for clinician reference
    #st.info(f"💡 Calculated NMR: {nmr_value:.2f} | Input Logic: NRS > 6: {nrs_binary}, Age > 60: {age_binary}")

    # ------------------- SHAP Analysis -------------------
    st.subheader("Model Explanation (SHAP)")
    
    # Use TreeExplainer for XGBoost
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(features_df)
    
    # Handle SHAP output format
    if isinstance(shap_values, list):
        s_val = shap_values[1]
    else:
        s_val = shap_values

    plt.figure(figsize=(10, 3))
    shap.force_plot(
        explainer_shap.expected_value, 
        s_val, 
        features_df, 
        matplotlib=True,
        show=False
    )
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=200)
    st.image("shap_plot.png")