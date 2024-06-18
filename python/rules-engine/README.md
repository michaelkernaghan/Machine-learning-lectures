# Rules Engine Example

## Overview

This script is part of the "Machine Learning Explorations" series. It processes Ethereum transaction data, labels transactions as credit or debit, trains a Random Forest classifier to predict transaction types, and includes steps for data preprocessing, model evaluation, and visualization of results. Additionally, it demonstrates the integration of a simple rules engine for mapping transactions to accounting entries.

## Files

- `updated_ethereum_transactions.csv`: The CSV file containing the updated Ethereum transaction data.
- `feature_importances.png`: Visualization of the top 10 feature importances.
- `confusion_matrix.png`: Visualization of the confusion matrix.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Setup

1. **Ensure you have Python 3.x installed on your system.**

2. **Install the required libraries:**
    ```sh
    pip install pandas numpy scikit-learn matplotlib
    ```

3. **Ensure the CSV file `updated_ethereum_transactions.csv` is placed in the correct directory:**

## Usage

1. **Run the script:**
    ```sh
    python3 /path/to/script/updated_ethereum_transactions.py
    ```

2. **The script will perform the following steps:**
    - Load the Ethereum transaction data from `updated_ethereum_transactions.csv`.
    - Ensure all entries in 'Received', 'Sent', and 'Fee' columns are strings before converting to float.
    - Verify the presence of the 'Operation' column and other required columns.
    - Label transactions as 'credit' or 'debit'.
    - Preprocess the data (handle missing values, scale numerical data, encode categorical data).
    - Train a Random Forest classifier on the preprocessed data.
    - Evaluate the model and print the classification report.
    - Visualize and save the feature importances and confusion matrix.
    - Predict transaction types for new data.
    - Perform cross-validation and hyperparameter tuning.
    - Integrate a rules engine for mapping transactions to accounting entries.

## Script Details

### Loading and Preprocessing Data

The script starts by loading the Ethereum transaction data from `updated_ethereum_transactions.csv` with the appropriate delimiter and handles quoting. It then ensures the 'Received', 'Sent', and 'Fee' columns are correctly formatted as floats and verifies the presence of necessary columns.

### Labeling Transactions

Transactions are labeled as 'credit' if the 'Received' value is greater than or equal to 0, and 'debit' if the 'Sent' value is less than or equal to 0. Transactions that do not fit these criteria are labeled as 'other'.

### Feature Selection and Preprocessing

The features selected for the model are 'Received', 'Sent', 'Fee', 'From address', and 'To address'. The script includes pipelines for handling missing values, scaling numerical data, and encoding categorical data.

### Splitting Data

The data is split into training and test sets with an 80-20 ratio.

### Training the Model

A Random Forest classifier is trained on the preprocessed data.

### Evaluating the Model

The model's performance is evaluated using a classification report. The script also displays the feature importances and confusion matrix.

### Visualization

The script generates and saves the following visualizations:
- **Feature Importances:** A bar plot showing the top 10 feature importances.
- **Confusion Matrix:** A matrix showing the model's performance in classifying transaction types.

### Predicting on New Data

The script includes an example of predicting transaction types for new data using the trained model and preprocessing pipeline.

### Cross-Validation and Hyperparameter Tuning

Cross-validation scores are calculated, and hyperparameter tuning is performed using GridSearchCV.

### Rules Engine for Accounting Entries

A simple rules engine is integrated to map transactions to accounting entries based on the predicted transaction types.

## License

This project is licensed under the MIT License - see the [LICENSE](../../../LICENSE) file for details.

---

