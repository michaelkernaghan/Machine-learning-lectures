# README.md

## Overview

This script processes Ethereum transaction data, performs various preprocessing steps, and trains a machine learning model to classify transaction operations. It uses a Random Forest classifier and includes hyperparameter tuning, model evaluation, and feature importance analysis. Additionally, it uses SMOTE to handle imbalanced data and provides functionality for predicting new transaction operations.

## Purpose

The primary purpose of this script is to:
1. Load and preprocess Ethereum transaction data.
2. Engineer features and handle missing values.
3. Train and evaluate a Random Forest classifier for transaction classification.
4. Address class imbalance using SMOTE.
5. Perform hyperparameter tuning and evaluate model performance.
6. Visualize feature importances and confusion matrix.
7. Predict the operation type for new transaction data.

## Methods

### Data Preprocessing

1. **Load the CSV File:**
   - The script loads Ethereum transaction data from `updated_ethereum_transactions.csv` using Pandas.

2. **Replace Commas in Numerical Columns:**
   - Numerical columns ('Received', 'Sent', 'Fee') are processed to replace commas with periods and convert the values to floats.

3. **Feature Engineering:**
   - Extract necessary columns for features and labels.
   - Ensure all required columns are present.

4. **Handle Missing Values and Scale Numerical Data:**
   - Use `SimpleImputer` to handle missing values.
   - Scale numerical features using `StandardScaler`.
   - Encode categorical features using `OneHotEncoder`.

### Model Training

1. **Preprocessing Pipeline:**
   - Define numerical and categorical pipelines for preprocessing.
   - Combine these pipelines using `ColumnTransformer`.

2. **Oversample Minority Class:**
   - Use SMOTE to oversample the minority class in the dataset.

3. **Train-Test Split:**
   - Split the data into training and test sets with an 80-20 ratio.

4. **Random Forest Classifier:**
   - Create a pipeline with a Random Forest classifier.
   - Train the classifier on the training data.

### Model Evaluation

1. **Classification Report:**
   - Evaluate the model using a classification report to display precision, recall, and F1-score.

2. **Feature Importances:**
   - Visualize feature importances using a bar plot.

3. **Cross-Validation:**
   - Perform cross-validation to evaluate the model's performance.

4. **Confusion Matrix:**
   - Visualize the confusion matrix to understand the classification performance.

### Hyperparameter Tuning

1. **Grid Search:**
   - Perform grid search for hyperparameter tuning using `GridSearchCV`.
   - Display the best parameters and cross-validation score.

### Prediction

1. **Predict New Data:**
   - Preprocess new transaction data and predict the operation type.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- SHAP

## Setup

1. **Ensure you have Python 3.x installed on your system.**

2. **Install the required libraries:**
    ```sh
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib shap
    ```

3. **Place the CSV file:**
   - Ensure the `updated_ethereum_transactions.csv` file is in the same directory as the script.

## Running the Script

1. **Run the script:**
    ```sh
    python3 neural-net.py
    ```

2. **Expected Output:**
   - The script will output the classification report, feature importances, cross-validation scores, confusion matrix, and the predicted operation for new data.

## Script Details

### Data Preprocessing

- The script loads the Ethereum transaction data, replaces commas in numerical columns with periods, converts these columns to floats, and handles missing values. It extracts features and labels, and applies scaling and one-hot encoding to the data.

### Model Training

- The script uses a Random Forest classifier within a pipeline. It handles class imbalance using SMOTE and splits the data into training and test sets. The model is trained on the training data and evaluated on the test data.

### Model Evaluation

- The script evaluates the model using a classification report, visualizes feature importances, performs cross-validation, and displays a confusion matrix.

### Hyperparameter Tuning

- The script performs hyperparameter tuning using grid search to find the best parameters for the Random Forest classifier.

### Prediction

- The script preprocesses new transaction data and predicts the operation type using the trained classifier.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.