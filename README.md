# Heart Disease Prediction Application

## Problem Statement
This project aims to develop a machine learning application to predict the presence and severity of heart disease based on various patient health indicators. The goal is to implement and compare multiple classification models, and then deploy the best-performing models (or all for demonstration) into an interactive Streamlit web application.

## Dataset Description
**Dataset**: `heart.csv`
**Source**: Kaggle/UCI (Typically from UCI Machine Learning Repository for heart disease datasets)

This dataset contains medical parameters collected from patients, which are used to predict the likelihood and severity of heart disease. The target variable `num` indicates the heart disease status, categorized into 5 levels:
- 0: No heart disease
- 1: Mild heart disease
- 2: Moderate heart disease
- 3: Severe heart disease
- 4: Critical heart disease

**Key Features**:
- `age`: Age of the patient
- `sex`: Sex of the patient (Male/Female)
- `cp`: Chest pain type
- `trestbps`: Resting blood pressure
- `chol`: Serum cholestoral in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting electrocardiographic results
- `thalch`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: The slope of the peak exercise ST segment
- `ca`: Number of major vessels (0-3) colored by flourosopy
- `thal`: Thallium stress test result

**Number of Features**: 12 (after dropping 'id' and 'dataset', before one-hot encoding)
**Number of Instances**: 920

## Models Used and Evaluation Metrics
Six different classification models were implemented and evaluated on the preprocessed `heart.csv` dataset. The evaluation metrics include Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC Score).

| Model                  | Accuracy | AUC Score | Precision | Recall  | F1 Score | MCC Score |
|:-----------------------|:---------|:----------|:----------|:--------|:---------|:----------|
| Logistic Regression    | 0.5815   | 0.7994    | 0.5523    | 0.5815  | 0.5615   | 0.3781    |
| Decision Tree          | 0.5054   | 0.6309    | 0.5174    | 0.5054  | 0.5107   | 0.2923    |
| K-Nearest Neighbor     | 0.5761   | 0.7081    | 0.5218    | 0.5761  | 0.5458   | 0.3612    |
| Naive Bayes            | 0.3043   | 0.7192    | 0.7016    | 0.3043  | 0.3371   | 0.2819    |
| Random Forest          | 0.5652   | 0.8112    | 0.5201    | 0.5652  | 0.5410   | 0.3487    |
| XGBoost                | 0.6087   | 0.7995    | 0.6017    | 0.6087  | 0.6033   | 0.4273    |

### Observations on Model Performance:
- **XGBoost** generally performs best across most metrics, showing the highest Accuracy, Precision, F1 Score, and MCC Score. Its AUC score is also very competitive.
- **Random Forest** has the highest AUC Score, indicating good discrimination ability, but its other metrics are slightly lower than XGBoost.
- **Logistic Regression** shows solid performance, especially in AUC, suggesting it captures linear relationships well even for multi-class problems.
- **K-Nearest Neighbor** and **Decision Tree** have moderate performance, with Decision Tree being the lowest performer in most metrics, possibly due to overfitting or lack of fine-tuning.
- **Naive Bayes** stands out with a surprisingly high Precision but very low Accuracy and Recall, suggesting it might be predicting only one class with high confidence while missing others. This often indicates a poor fit for the data distribution after scaling.

Overall, **XGBoost** appears to be the most robust model for this specific heart disease prediction task among the ones evaluated.

## How to run locally
To run this Streamlit application locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <YOUR_GITHUB_REPO_LINK>
    cd project-folder
    ```

2.  **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

## App Deployment Link
[Link to Deployed Streamlit App](<YOUR_STREAMLIT_APP_LINK_HERE>)

## Screenshot
[Include a screenshot of your assignment execution on BITS Virtual Lab here]

