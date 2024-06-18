# Ethereum Gas Price Anomaly Detection with Machine Learning

## Overview

This script is designed to detect anomalies in Ethereum transaction gas prices using an autoencoder neural network. The script includes steps for data preprocessing, model training, anomaly detection, and visualization of results.

## Files

- `google-ethereum-transactions.csv`: The CSV file containing Ethereum transaction data.
- `anomalous_transactions.csv`: Output file containing the transactions identified as anomalies.
- `reconstruction_error_distribution.png`: Visualization of the reconstruction error distribution.
- `gas_price_anomalies.png`: Scatter plot of gas prices with anomalies highlighted.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- TensorFlow
- NumPy
- Matplotlib
- Seaborn

## Setup

1. **Ensure you have Python 3.x installed on your system.**

2. **Install the required libraries:**
    ```sh
    pip install pandas scikit-learn tensorflow numpy matplotlib seaborn
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
    python3 /path/to/script/ethereum_gas_price_anomaly_detection.py
    ```

2. **The script will perform the following steps:**
    - Load the Ethereum transaction data from `google-ethereum-transactions.csv`.
    - Fill missing values in the dataset.
    - Focus on the 'gas_price' feature and standardize it.
    - Split the data into training and test sets.
    - Build and train an autoencoder neural network.
    - Predict the reconstruction of gas prices and calculate reconstruction errors.
    - Determine a threshold for anomaly detection based on the reconstruction error.
    - Identify and mark anomalies in the original dataset.
    - Save the anomalous transactions to `anomalous_transactions.csv`.
    - Generate and save visualizations:
        - Reconstruction error distribution (`reconstruction_error_distribution.png`)
        - Scatter plot of gas prices with anomalies (`gas_price_anomalies.png`)

## Script Details

### Loading and Preprocessing Data

The script starts by loading the Ethereum transaction data from `google-ethereum-transactions.csv`. Missing values are filled with 0, and the 'gas_price' feature is standardized using `StandardScaler`.

### Splitting Data

The data is split into training and test sets with an 80-20 ratio.

### Building and Training the Model

An autoencoder neural network is built using TensorFlow. The model includes an input layer, multiple dense layers with ReLU activation, and an output layer. The model is compiled with the Adam optimizer and trained using mean squared error loss.

### Predicting and Evaluating

The autoencoder predicts the reconstruction of gas prices for both the training and test sets. The reconstruction error is calculated, and a threshold for anomaly detection is determined based on the training set errors.

### Identifying Anomalies

Transactions with reconstruction errors above the threshold are marked as anomalies. These anomalies are saved to `anomalous_transactions.csv`.

### Visualization

The script generates and saves the following visualizations:
- **Reconstruction Error Distribution:** A histogram showing the distribution of reconstruction errors for the training and test sets, with the anomaly detection threshold indicated.
- **Gas Price Anomalies:** A scatter plot showing gas prices with anomalies highlighted in red.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
