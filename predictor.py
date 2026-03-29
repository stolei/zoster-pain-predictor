import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier  # 确保安装了 catboost

# --- 1. Load Model & Data ---
@st.cache_resource
def load_model():
    # 加载 CatBoost 模型 (假设文件名已改为 CatBoost.pkl)
    # 如果你是用 model.save_model('model.cbm') 保存的，请改用 model.load_model
    model = joblib.load('CatBoost.pkl')
    return model

model = load_model()

# ------------------- 2. Streamlit UI Configuration -------------------
st.set_page_config(page_title="PRF Efficacy Predictor", layout="wide")
st.title("🛡️ PRF Efficacy Predictor for Zoster-Associated Pain")
st.markdown("""
This clinical decision support tool utilizes a **CatBoost machine learning model** to predict the efficacy of 
**Pulsed Radiofrequency (PRF)** treatment for patients with herpes zoster-related pain.
""")
st.markdown("---")

# Layout: Input on the left, Results/Analytics on the right
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📋 Patient Clinical Inputs")
    
    # Feature 1: Opioid use (Binary)
    opioid = st.selectbox("Opioid Use Status:", 
                          options=[0, 1], 
                          format_func=lambda x: "User" if x == 1 else "Non-user")
    
    # Feature 2: NRS Score (Continuous)
    NRS = st.slider("Pain Intensity (NRS Score):", 
                    min_value=0, max_value=10, value=5, 
                    help="Numeric Rating Scale: 0 = No pain, 10 = Worst pain imaginable")
    
    # Feature 3: Age (Continuous)
    age = st.slider("Patient Age:", min_value=18, max_value=110, value=60)
    
    # Feature 4: ApoB (Continuous)
    ApoB = st.number_input("Apolipoprotein B (ApoB, g/L):", 
                           min_value=0.01, max_value=5.0, value=0.80, format="%.2f")

    # Feature 5: NMR Calculation (Neutrophil-to-Monocyte Ratio)
    st.markdown("**Lab Results (for NMR calculation):**")
    neutrophil = st.number_input("Neutrophil Count (10^9/L):", min_value=0.01, max_value=50.0, value=4.0)
    monocyte = st.number_input("Monocyte Count (10^9/L):", min_value=0.01, max_value=10.0, value=0.5)
    nmr_value = neutrophil / monocyte
    st.caption(f"Calculated NMR: {nmr_value:.2f}")

# ------------------- 3. Prediction Logic -------------------
# 重要：确保特征顺序与训练时完全一致
feature_names = ["opioid", "NRS", "age", "NMR", "ApoB"]
feature_values = [opioid, NRS, age, nmr_value, ApoB]
features_df = pd.DataFrame([feature_values], columns=feature_names)

with col2:
    st.subheader("🚀 Prediction & Analysis")
    
    if st.button("Generate Prediction"):
        # Model Inference
        predicted_class = model.predict(features_df)[0]
        # 处理 CatBoost 可能返回数组的情况
        if isinstance(predicted_class, np.ndarray):
            predicted_class = predicted_class[0]
            
        predicted_proba = model.predict_proba(features_df)[0]
        prob_val = predicted_proba[1]  # Probability of positive outcome

        # Visualization of Result
        if predicted_class == 1:
            st.success("### Outcome: Positive Response Likely")
            st.metric(label="Probability of Efficacy", value=f"{prob_val*100:.1f}%")
            st.info("✅ **Recommendation:** The patient is likely to benefit from PRF treatment.")
        else:
            st.error("### Outcome: Negative Response Likely")
            st.metric(label="Probability of Efficacy", value=f"{prob_val*100:.1f}%")
            st.warning("⚠️ **Recommendation:** Consider alternative therapies, predicted PRF efficacy is low.")

        st.divider()

        # ------------------- 4. SHAP Explanation -------------------
        st.subheader("🔍 Individual Feature Contribution (SHAP)")
        
        try:
            # CatBoost 专用解释器
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df)

            # CatBoost 的 SHAP 输出处理
            # 通常 CatBoost 对二分类返回的是针对 Log-Odds 的单维数组
            if len(shap_values.shape) == 2:
                # 形状为 (1, n_features)
                single_shap_values = shap_values[0, :]
            else:
                # 如果返回了多维 (samples, features, classes)
                single_shap_values = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

            base_value = explainer.expected_value
            # 如果 base_value 是列表，取第二个元素（类别1）
            if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
                base_value = base_value[1]

            # 绘制 Force Plot
            fig = plt.figure(figsize=(12, 3))
            shap.force_plot(
                base_value, 
                single_shap_values, 
                features_df.iloc[0, :], 
                matplotlib=True, 
                show=False,
                link="identity" # CatBoost SHAP 通常在 Log-Odds 空间
            )
            st.pyplot(plt.gcf())
            
            st.markdown("""
            **How to read this plot:**
            * **Red arrows** (to the right) increase the probability of a positive outcome.  
            * **Blue arrows** (to the left) decrease the probability.  
            * The length of the arrow indicates the impact of that specific feature.
            """)
            
        except Exception as e:
            st.error(f"SHAP Analysis Error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Disclaimer: This tool is for research and clinical reference only.")