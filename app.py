import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings

# --- Silence Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# --- Page Configuration ---
st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

# --- Application Title ---
st.title("Heart Disease Prediction App")

# --- User Guidelines ---
with st.expander("ℹ️ User Guidelines & Feature Definitions"):
    st.markdown("""
    ### **How to use this app:**
    1. **Select a Model:** Choose from the sidebar on the left.
    2. **Input Data:** Enter patient data manually or upload a CSV.
    3. **Get Results:** Click the **'🚀 Predict Heart Disease Level'** button.
    """)

# --- Model Loading with Spinner ---
model_dir = 'model' 

@st.cache_resource 
def load_resources():
    # Show a spinner only the first time the models are loaded
    with st.spinner("Initializing AI Models... please wait."):
        models, scalers = {}, {}
        model_files = {
            'Logistic Regression': 'logistic_regression_model.pkl',
            'Decision Tree': 'decision_tree_model.pkl',
            'K-Nearest Neighbor': 'knn_model.pkl',
            'Naive Bayes': 'naive_bayes_model.pkl',
            'Random Forest': 'random_forest_model.pkl',
            'XGBoost': 'xgboost_model.pkl'
        }
        scaler_files = {
            'Logistic Regression': 'logistic_regression_scaler.pkl',
            'K-Nearest Neighbor': 'knn_scaler.pkl',
            'Naive Bayes': 'naive_bayes_scaler.pkl'
        }
        for name, file_name in model_files.items():
            try:
                with open(os.path.join(model_dir, file_name), 'rb') as f:
                    models[name] = pickle.load(f)
            except: pass
        for name, file_name in scaler_files.items():
            try:
                with open(os.path.join(model_dir, file_name), 'rb') as f:
                    scalers[name] = pickle.load(f)
            except: pass
    return models, scalers

models, scalers = load_resources()

# --- SIDEBAR ---
st.sidebar.header("1. Model Configuration")
selected_model_name = st.sidebar.selectbox("Choose a Model:", list(models.keys()) if models else ["No Models Found"])
selected_model = models.get(selected_model_name)

st.sidebar.header("2. Input Patient Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

expected_columns_order = [
    'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca',
    'sex_Male', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'fbs_True', 'restecg_normal', 'restecg_st-t abnormality',
    'exang_True', 'slope_flat', 'slope_upsloping',
    'thal_normal', 'thal_reversable defect'
]

def preprocess_user_input(df_input):
    for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype('object')
    df_input = df_input.drop(columns=['id', 'dataset', 'num', 'target'], errors='ignore')
    num_cols = df_input.select_dtypes(include=np.number).columns
    cat_cols = df_input.select_dtypes(include=['object']).columns
    for c in num_cols: df_input[c] = df_input[c].fillna(df_input[c].mean())
    for c in cat_cols: df_input[c] = df_input[c].fillna(df_input[c].mode()[0])
    df_input = pd.get_dummies(df_input, columns=cat_cols, drop_first=True, dtype=int)
    final_df = pd.DataFrame(0, index=df_input.index, columns=expected_columns_order)
    for col in final_df.columns:
        if col in df_input.columns: final_df[col] = df_input[col]
    return final_df

def get_manual_input():
    st.sidebar.subheader("Manual Data Entry")
    age = st.sidebar.slider('Age', 1, 100, 54); trestbps = st.sidebar.slider('Resting BP', 80, 200, 130)
    chol = st.sidebar.slider('Cholesterol', 100, 600, 240); thalch = st.sidebar.slider('Max Heart Rate', 60, 220, 150)
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0); ca = st.sidebar.selectbox('Major Vessels', (0, 1, 2, 3))
    sex = st.sidebar.radio('Sex', ('Male', 'Female')); cp = st.sidebar.selectbox('Chest Pain Type', ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'))
    fbs = st.sidebar.radio('Fasting Sugar > 120', ('False', 'True')); restecg = st.sidebar.selectbox('ECG Results', ('normal', 'st-t abnormality', 'lv hypertrophy'))
    exang = st.sidebar.radio('Exercise Angina', ('False', 'True')); slope = st.sidebar.selectbox('ST Slope', ('upsloping', 'flat', 'downsloping'))
    thal = st.sidebar.selectbox('Thalassemia', ('normal', 'fixed defect', 'reversable defect'))
    return pd.DataFrame({'age': age, 'trestbps': trestbps, 'chol': chol, 'thalch': thalch, 'oldpeak': oldpeak, 'ca': ca, 'sex': sex, 'fbs': fbs, 'exang': exang, 'cp': cp, 'restecg': restecg, 'slope': slope, 'thal': thal}, index=[0])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    processed_input_df = preprocess_user_input(input_df.copy())
else:
    input_df = get_manual_input()
    processed_input_df = preprocess_user_input(input_df.copy())

# --- MAIN PAGE: Data Summary ---
st.subheader("📋 Patient Data Summary")
if uploaded_file:
    st.dataframe(input_df.head())
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", input_df['age'][0]); c1.metric("Sex", input_df['sex'][0])
    c2.metric("Resting BP", f"{input_df['trestbps'][0]} mmHg"); c2.metric("Cholesterol", f"{input_df['chol'][0]} mg/dl")
    c3.metric("Max HR", input_df['thalch'][0]); c3.metric("ST Depression", input_df['oldpeak'][0])
    c4.write("**Observations:**"); c4.caption(f"Chest Pain: {input_df['cp'][0]}"); c4.caption(f"Thal: {input_df['thal'][0]}")

st.markdown("---")

# --- MAIN PAGE: Prediction with Loader ---
if st.button('🚀 Predict Heart Disease Level'):
    if not selected_model:
        st.error("Model not found.")
    else:
        # Start the Spinner
        with st.spinner(f"AI Model ({selected_model_name}) is analyzing patient health data..."):
            try:
                final_features = processed_input_df
                if selected_model_name in scalers:
                    final_features = scalers[selected_model_name].transform(processed_input_df)

                prediction = selected_model.predict(np.array(final_features))
                probs = selected_model.predict_proba(np.array(final_features))
                
                outcome_map = {0: 'No Disease', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Critical'}
                
                st.toast("Analysis Complete!", icon="✅") # Subtle feedback in the corner
                st.subheader("🎯 Prediction Results")
                
                if len(prediction) == 1:
                    res = outcome_map.get(prediction[0])
                    if prediction[0] == 0: 
                        st.success(f"### Result: {res}")
                    else: 
                        st.warning(f"### Result: {res} Detected")
                    st.bar_chart(pd.DataFrame(probs[0], index=outcome_map.values(), columns=['Probability']))
                else:
                    input_df['Prediction'] = [outcome_map.get(p) for p in prediction]
                    st.dataframe(input_df)
            except Exception as e: 
                st.error(f"Error during calculation: {e}")

st.markdown("---")
st.subheader("📊 Training Performance Metrics")
st.markdown("""
| Model                  | Accuracy | AUC Score | Precision | Recall  | F1 Score | MCC Score |
|:-----------------------|:---------|:----------|:----------|:--------|:---------|:----------|
| Logistic Regression    | 0.9737   | 0.9974    | 0.9722    | 0.9859  | 0.9790   | 0.9439    |
| Decision Tree          | 0.9474   | 0.9440    | 0.9577    | 0.9577  | 0.9577   | 0.8880    |
| K-Nearest Neighbor     | 0.9474   | 0.9820    | 0.9577    | 0.9577  | 0.9577   | 0.8880    |
| Naive Bayes            | 0.9649   | 0.9974    | 0.9589    | 0.9859  | 0.9722   | 0.9253    |
| Random Forest          | 0.9649   | 0.9953    | 0.9589    | 0.9859  | 0.9722   | 0.9253    |
| XGBoost                | 0.9561   | 0.9908    | 0.9583    | 0.9718  | 0.9650   | 0.9064    |
""")
