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
| **Logistic Regression**| **0.9737**| **0.9974**| **0.9722**| **0.9859**| **0.9790**| **0.9439**|
| **Naive Bayes**        | 0.9649   | 0.9974    | 0.9589    | 0.9859  | 0.9722   | 0.9253    |
| **Random Forest**      | 0.9649   | 0.9953    | 0.9589    | 0.9859  | 0.9722   | 0.9253    |
| **XGBoost**            | 0.9561   | 0.9908    | 0.9583    | 0.9718  | 0.9650   | 0.9064    |
| **K-Nearest Neighbor** | 0.9474   | 0.9820    | 0.9577    | 0.9577  | 0.9577   | 0.8880    |
| **Decision Tree**      | 0.9474   | 0.9440    | 0.9577    | 0.9577  | 0.9577   | 0.8880    |

### Observations on Model Performance:

| Index | ML Model Name | Observation about model performance |
|:---:|:---|:---|
| 0 | **Logistic Regression** | Achieved the highest accuracy, F1-score, and MCC among all models. It showed excellent class separation, stable performance, and fast convergence, making it a strong baseline for the dataset. |
| 1 | **Decision Tree** | Showed lower accuracy compared to other models due to overfitting on training data. Performance dropped on test data, indicating sensitivity to data variations and depth settings. |
| 2 | **kNN** | Performed well with high recall and F1-score by capturing local data patterns. However, performance depends strongly on feature scaling and choice of k value. |
| 3 | **Naive Bayes** | Delivered balanced performance with good recall and precision. Assumption of feature independence limits its ability to model complex relationships, but it remains computationally efficient. |
| 4 | **Random Forest** | Improved performance over a single decision tree by reducing overfitting through ensemble learning. Provided strong accuracy and robustness at moderate computational cost. |
| 5 | **XGBoost** | Achieved excellent AUC and F1-score by effectively learning complex feature interactions. As a boosting-based ensemble, it provided high predictive power but required more training time. |


## How to run locally
To run this Streamlit application locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/sai-bits/Heart-Disease-Prediction.git
    cd Heart-Disease-Prediction
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

