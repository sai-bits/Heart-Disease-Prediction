import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Configuration ---
st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

# --- Application Title and Description ---
st.title("Heart Disease Prediction App")
st.markdown("An interactive web application to predict heart disease levels using various machine learning models.")

# --- Model and Scaler Loading ---
model_dir = 'model' # Directory where models and scalers are saved

@st.cache_resource # Cache resources to avoid reloading on every rerun
def load_resources():
    """Loads all trained models and scalers from the 'model' directory."""
    models = {}
    scalers = {}

    # Define model filenames
    model_files = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'K-Nearest Neighbor': 'knn_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }

    # Define scaler filenames for models that require scaled input
    scaler_files = {
        'Logistic Regression': 'logistic_regression_scaler.pkl',
        'K-Nearest Neighbor': 'knn_scaler.pkl',
        'Naive Bayes': 'naive_bayes_scaler.pkl'
    }

    # Load models
    for name, file_name in model_files.items():
        file_path = os.path.join(model_dir, file_name)
        try:
            with open(file_path, 'rb') as f:
                models[name] = pickle.load(f)
            # st.success(f"Successfully loaded {name} model.")
        except FileNotFoundError:
            st.error(f"Error: Model file '{file_name}' not found at '{model_dir}/'. Please ensure all models are saved correctly.")
            st.stop() # Stop the app if a critical model is missing
        except Exception as e:
            st.error(f"Error loading {name} model from '{file_name}': {e}")
            st.stop()

    # Load scalers
    for name, file_name in scaler_files.items():
        file_path = os.path.join(model_dir, file_name)
        try:
            with open(file_path, 'rb') as f:
                scalers[name] = pickle.load(f)
            # st.success(f"Successfully loaded {name} scaler.")
        except FileNotFoundError:
            st.warning(f"Scaler file '{file_name}' for {name} not found. This may be expected if the model does not require explicit scaling.")
        except Exception as e:
            st.warning(f"Error loading {name} scaler from '{file_name}': {e}")

    if not models: # Check if any models were loaded
        st.error("No models were loaded. Please check the 'model' directory.")
        st.stop()

    return models, scalers

# Load all resources once
models, scalers = load_resources()

# --- Model Selection Sidebar ---
st.sidebar.header("1. Select ML Model")
selected_model_name = st.sidebar.selectbox("Choose a Model for Prediction:", list(models.keys()))
selected_model = models[selected_model_name]

# --- User Input Features Sidebar ---
st.sidebar.header("2. Input Patient Features or Upload Data")

# Option for user to upload a CSV file
uploaded_file = st.sidebar.file_uploader("Upload your own CSV input file (optional)", type=["csv"])

# Expected columns after preprocessing, used for aligning user input or uploaded data
expected_columns_order = [
    'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca',
    'sex_Male',
    'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'fbs_True',
    'restecg_normal', 'restecg_st-t abnormality',
    'exang_True',
    'slope_flat', 'slope_upsloping',
    'thal_normal', 'thal_reversable defect'
]

def preprocess_user_input(df_input):
    # Ensure all original categorical columns are of 'object' type if they come in as 'str'
    # This helps get_dummies to correctly identify them.
    for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype('object')

    # 1. Drop irrelevant columns if present
    # 'num' is the target, should be dropped if present in input CSV
    df_input = df_input.drop(columns=['id', 'dataset', 'num'], errors='ignore')

    # 2. Impute missing values
    numerical_cols = df_input.select_dtypes(include=np.number).columns
    categorical_cols = df_input.select_dtypes(include=['object']).columns

    # Impute numerical columns with the mean
    for column in numerical_cols:
        if df_input[column].isnull().any():
            df_input[column] = df_input[column].fillna(df_input[column].mean())

    # Impute categorical columns with the mode
    for column in categorical_cols:
        if df_input[column].isnull().any():
            df_input[column] = df_input[column].fillna(df_input[column].mode()[0])

    # 3. One-hot encode categorical features
    df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True, dtype=int)

    # 4. Align columns to the expected order from training data
    final_df = pd.DataFrame(0, index=df_input.index, columns=expected_columns_order)

    for col in final_df.columns:
        if col in df_input.columns:
            final_df[col] = df_input[col]

    return final_df

def user_input_features():
    """Collects user input for all features via Streamlit widgets."""
    st.sidebar.markdown("### Enter Features Manually")
    age = st.sidebar.slider('Age', 29, 77, 54, help="Age of the patient in years")
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 90, 200, 130, help="Resting blood pressure (in mm Hg on admission to the hospital)")
    chol = st.sidebar.slider('Cholesterol (chol)', 126, 564, 246, help="Serum cholestoral in mg/dl")
    thalch = st.sidebar.slider('Maximum Heart Rate Achieved (thalch)', 71, 202, 144, help="Maximum heart rate achieved")
    oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest (oldpeak)', 0.0, 6.2, 1.2, help="ST depression induced by exercise relative to rest")
    ca = st.sidebar.slider('Number of major vessels (0-3) colored by flourosopy (ca)', 0, 3, 0, help="Number of major vessels (0-3) colored by flourosopy")

    sex_Male = st.sidebar.selectbox('Sex', ('Male', 'Female'), help="Patient's sex (Male=True, Female=False)") == 'Male'
    cp_options = ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic']
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', cp_options, help="Chest pain type: typical angina, atypical angina, non-anginal, asymptomatic")
    fbs_True = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ('False', 'True'), help="Fasting blood sugar > 120 mg/dl (True=1, False=0)") == 'True'
    restecg_options = ['normal', 'st-t abnormality', 'lv hypertrophy']
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', restecg_options, help="Resting electrocardiographic results: normal, st-t abnormality, lv hypertrophy")
    exang_True = st.sidebar.selectbox('Exercise Induced Angina (exang)', ('False', 'True'), help="Exercise induced angina (True=1, False=0)") == 'True'
    slope_options = ['upsloping', 'flat', 'downsloping']
    slope = st.sidebar.selectbox('Slope of the peak exercise ST segment (slope)', slope_options, help="The slope of the peak exercise ST segment: upsloping, flat, downsloping")
    thal_options = ['normal', 'fixed defect', 'reversable defect']
    thal = st.sidebar.selectbox('Thalassmeia (thal)', thal_options, help="Thal: normal, fixed defect, reversable defect")

    # Store all inputs in a dictionary
    data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalch': thalch,
        'oldpeak': oldpeak,
        'ca': ca,
        'sex': 'Male' if sex_Male else 'Female', # Pass original category for get_dummies
        'fbs': 'True' if fbs_True else 'False',  # Pass original category for get_dummies
        'exang': 'True' if exang_True else 'False', # Pass original category for get_dummies
        'cp': cp, # Keep original for preprocessing
        'restecg': restecg,
        'slope': slope,
        'thal': thal
    }

    features = pd.DataFrame(data, index=[0])
    return features

if uploaded_file is not None:
    # Read uploaded CSV
    input_df = pd.read_csv(uploaded_file)
    st.subheader('Uploaded Patient Features (First 5 Rows)')
    st.write(input_df.head())
    # Preprocess uploaded data
    processed_input_df = preprocess_user_input(input_df.copy())
else:
    # Get user input from sidebar widgets
    input_df = user_input_features()
    st.subheader('User Input Features')
    st.write(input_df)
    # Preprocess user input from widgets
    processed_input_df = preprocess_user_input(input_df.copy())

# Scale the input features if the selected model requires it
final_input_for_prediction = processed_input_df
if selected_model_name in scalers:
    scaler_for_model = scalers[selected_model_name]
    final_input_for_prediction = scaler_for_model.transform(processed_input_df)
    # Convert back to DataFrame for consistency
    final_input_for_prediction = pd.DataFrame(final_input_for_prediction, columns=processed_input_df.columns)

# Make prediction when a button is clicked or automatically
if st.button('Predict Heart Disease Level'):
    try:
        prediction_raw = selected_model.predict(final_input_for_prediction)
        prediction_proba = selected_model.predict_proba(final_input_for_prediction)

        st.subheader(f'Prediction by {selected_model_name}')

        # Map raw prediction to a descriptive outcome
        outcome_map = {
            0: 'No heart disease (0)',
            1: 'Mild heart disease (1)',
            2: 'Moderate heart disease (2)',
            3: 'Severe heart disease (3)',
            4: 'Critical heart disease (4)'
        }

        # Display results for single prediction or multiple rows from uploaded file
        if len(prediction_raw) == 1:
            predicted_level = prediction_raw[0]
            st.success(f"The predicted heart disease level is: **{outcome_map.get(predicted_level, 'Unknown')}**")

            st.subheader('Prediction Probability (per class)')
            class_labels = [outcome_map.get(i, f'Class {i}') for i in range(prediction_proba.shape[1])]
            proba_series = pd.Series(prediction_proba[0], index=class_labels)
            proba_df_display = proba_series.to_frame(name='Probability').sort_values(by='Probability', ascending=False)
            proba_df_display['Probability'] = proba_df_display['Probability'].apply(lambda x: f"{x:.2%}")
            st.dataframe(proba_df_display)
        else:
            st.write("### Predictions for Uploaded Data")
            # Create a new DataFrame for results with only the predicted level and probabilities
            results_df = pd.DataFrame(index=input_df.index)
            results_df['Predicted_Disease_Level'] = [outcome_map.get(level, 'Unknown') for level in prediction_raw]

            # Add probabilities for each class, formatted as percentage
            for i in range(prediction_proba.shape[1]):
                class_label = outcome_map.get(i, f'Class {i}')
                results_df[f'Probability: {class_label}'] = [f'{p:.2%}' for p in prediction_proba[:, i]]
            st.dataframe(results_df)

        st.markdown("---\n_Note: The model predicts a likelihood of heart disease based on the input features. Consult a medical professional for diagnosis._")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Model Evaluation Metrics Section (Placeholder) ---
st.subheader('Model Evaluation Metrics (from training)')
st.info("Detailed evaluation metrics for the selected model (Accuracy, AUC, Precision, Recall, F1, MCC) would typically be displayed here based on its performance on the test set during training. For this demo, please refer to the `README.md` for a comparison table.")

# --- Optional: Add a section for more info or visualizations ---
st.markdown("\n--- ")
st.markdown("### About the Application")
st.markdown("This application was developed as part of an assignment for Machine Learning. It demonstrates the deployment of various classification models using Streamlit.")