# Etheruem Transaction Analysis with Machine Learning Tools

## Overview

This script is part of the "Machine Learning Explorations" project. It analyzes Ethereum transaction data to classify transactions as either Credit or Debit using a Random Forest classifier. The script also includes data visualization to understand the distribution and patterns within the transaction data.

## Files

- `google-ethereum-transactions.csv`: The CSV file containing the Ethereum transaction data.
- `labeled_google-ethereum-transactions.csv`: Output file containing the transaction data labeled by the classifier.
- `count_of_credits_and_debits.png`: Visualization of the count of Credit and Debit transactions.
- `distribution_of_transaction_values.png`: Visualization of the distribution of transaction values by type.
- `gas_vs_transaction_value.png`: Scatter plot of gas vs. transaction value.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Setup

1. **Ensure you have Python 3.x installed on your system.**

2. **Install the required libraries:**
    ```sh
    pip install pandas scikit-learn matplotlib seaborn
    ```

3. **Ensure the CSV file `google-ethereum-transactions.csv` is placed in the correct directory:**

## Usage

1. **Run the script:**
    ```sh
    python3 /path/to/script/google-ethereum-transactions.py
    ```

2. **The script will perform the following steps:**
    - Load the Ethereum transaction data from `google-ethereum-transactions.csv`.
    - Fill missing values in the dataset.
    - Create labels for Credit (value > 0) and Debit (value == 0) transactions.
    - Select features (`value`, `gas`, `gas_price`) for the model.
    - Split the data into training and test sets.
    - Train a Random Forest classifier on the training data.
    - Predict on the test set and evaluate the model.
    - Label the transactions in the original dataset using the trained model.
    - Save the labeled dataset to `labeled_google-ethereum-transactions.csv`.
    - Generate and save visualizations:
        - Count of Credits and Debits (`count_of_credits_and_debits.png`)
        - Distribution of Transaction Values by Type (`distribution_of_transaction_values.png`)
        - Gas vs. Transaction Value (`gas_vs_transaction_value.png`)

## Script Details

### Loading and Preprocessing Data

The script starts by loading the Ethereum transaction data from `google-ethereum-transactions.csv` and fills any missing values with 0.

### Creating Labels

Labels are created to classify transactions as Credit (value > 0) or Debit (value == 0).

### Feature Selection

The features selected for the model are `value`, `gas`, and `gas_price`.

### Splitting Data

The data is split into training and test sets with an 80-20 ratio.

### Training the Model

A Random Forest classifier is trained on the training data.

### Evaluating the Model

The model's accuracy and classification report are printed to the console.

### Labeling Transactions

The trained model is used to label the transactions in the original dataset, and the labeled data is saved to `labeled_google-ethereum-transactions.csv`.

### Visualization

The script generates and saves the following visualizations:
- **Count of Credits and Debits:** A count plot showing the number of Credit and Debit transactions.
- **Distribution of Transaction Values by Type:** A histogram showing the distribution of transaction values by type.
- **Gas vs. Transaction Value:** A scatter plot showing the relationship between gas and transaction value.

## License

The whatever you like license.

---

