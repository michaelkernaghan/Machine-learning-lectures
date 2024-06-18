# Build and train a Transformer-based neural network model

## Overview

This script is part of the "Machine Learning Explorations" series. It processes Ethereum transaction data, extracts and engineers features, trains a Transformer-based neural network to classify transaction types, and includes steps for data preprocessing, model evaluation, and visualization of results.

## Files

- `google-ethereum-transactions.csv`: The CSV file containing the Ethereum transaction data.
- `test_data_with_predictions.csv`: Output file containing the test data with actual and predicted labels.
- `transaction_type_confusion_matrix.png`: Visualization of the confusion matrix.
- `test_data_results.png`: Scatter plot of test data showing actual vs. predicted transaction types.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Seaborn
- Matplotlib

## Setup

1. **Ensure you have Python 3.x installed on your system.**

2. **Install the required libraries:**
    ```sh
    pip install pandas numpy scikit-learn tensorflow seaborn matplotlib
    ```

3. **Ensure the CSV file `google-ethereum-transactions.csv` is placed in the correct directory:**
    - The expected directory structure is:
      ```
      project_root/
      └── google-ethereum-transactions.csv
      ```

## Usage

1. **Run the script:**
    ```sh
    python3 /path/to/script/ethereum_transaction_classification.py
    ```

2. **The script will perform the following steps:**
    - Load the Ethereum transaction data from `google-ethereum-transactions.csv`.
    - Fill missing values and ensure numeric columns are correctly typed.
    - Map transaction type numbers to Ethereum transaction types.
    - Extract time-based features and calculate additional features such as gas usage efficiency and address frequency.
    - Define features and target for the model.
    - Encode the target variable and standardize the features.
    - Split the data into training and test sets.
    - Build and train a Transformer-based neural network model.
    - Evaluate the model using a classification report and confusion matrix.
    - Save the test data with predicted and actual labels to a CSV file.
    - Generate and save visualizations of the confusion matrix and test results.

## Script Details

### Loading and Preprocessing Data

The script starts by loading the Ethereum transaction data from `google-ethereum-transactions.csv`. Missing values are filled with 0, and numeric columns are correctly typed. Transaction types are mapped to descriptive labels, and time-based features are extracted. Additional features such as gas usage efficiency and address frequency are calculated.

### Feature Selection and Preprocessing

The features selected for the model are 'gas_price', 'value', 'gas', 'hour', 'day_of_week', 'day_of_month', 'gas_usage_efficiency', 'gas_price_value_ratio', 'from_address_frequency', and 'to_address_frequency'. The target variable is 'transaction_type', which is encoded as a one-hot vector. The features are standardized using `StandardScaler`.

### Splitting Data

The data is split into training and test sets with an 80-20 ratio. The features are reshaped to fit the Transformer model input requirements.

### Building and Training the Model

A Transformer-based neural network model is built using TensorFlow. The model includes Transformer encoder layers followed by a multi-layer perceptron (MLP) for classification. The model is compiled with the Adam optimizer and trained using categorical cross-entropy loss.

### Evaluating the Model

The model's performance is evaluated using a classification report. The script also generates a confusion matrix and saves it as an image.

### Visualization

The script generates and saves the following visualizations:
- **Confusion Matrix:** A heatmap showing the model's performance in classifying transaction types.
- **Test Data Results:** A scatter plot showing the actual vs. predicted transaction types based on gas price and value.

### Saving Results

The test data with predicted and actual labels is saved to `test_data_with_predictions.csv`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
