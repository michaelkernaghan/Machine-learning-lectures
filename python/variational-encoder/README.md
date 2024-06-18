# Variational Encoder to generate synthetic Ethereum transaction data

## Overview

This script demonstrates the use of a Variational Autoencoder (VAE) to generate synthetic Ethereum transaction data. It performs several steps including data preprocessing, feature engineering, model training, and generating new transaction data.

## Purpose

The purpose of this script is to:
1. Load and preprocess Ethereum transaction data.
2. Engineer features from the transaction data.
3. Train a Variational Autoencoder (VAE) to learn the distribution of the transaction data.
4. Generate new synthetic transactions using the trained VAE model.
5. Output the generated transactions in a human-readable format.

## Functionality

### Data Preprocessing

1. **Load the CSV File:**
   - The script loads Ethereum transaction data from `google-ethereum-transactions.csv`.

2. **Fill Missing Values:**
   - Missing values in the dataset are filled with zeros to ensure smooth processing.

3. **Convert Columns to Numeric Types:**
   - Necessary columns are converted to numeric types, with errors coerced to ensure proper data format.

4. **Extract Time-Based Features:**
   - The script extracts time-based features such as the hour, day of the week, and day of the month from the `block_timestamp` column.

5. **Calculate Additional Features:**
   - Gas usage efficiency and gas price to value ratio are calculated.
   - Address frequency features are computed for both 'from_address' and 'to_address'.

### Feature Engineering

- The script defines a set of features to be used for training the VAE, including `gas_price`, `value`, `gas`, `transaction_type`, and various time and efficiency features.

### Data Standardization

- The features are standardized using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1.

### Train-Test Split

- The data is split into training and test sets with an 80-20 ratio for model evaluation.

### Variational Autoencoder (VAE) Model

1. **Define the VAE Model:**
   - The VAE model is defined with an encoder that reduces the dimensionality of the input data to a latent space and a decoder that reconstructs the data from the latent space.

2. **Compile the Model:**
   - The VAE model is compiled with a custom loss function that includes both reconstruction loss and KL divergence loss.

3. **Train the Model:**
   - The model is trained on the training data for 50 epochs.

### Generate New Transactions

- After training, the script generates new synthetic transactions by sampling from the latent space and using the decoder to generate new data points.

### Output

- The generated data is inverse-transformed to its original scale.
- The synthetic transactions are formatted and printed in a human-readable table using the `tabulate` library.

## Prerequisites

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Matplotlib
- Seaborn
- Tabulate

## Setup

1. **Ensure you have Python 3.x installed on your system.**

2. **Install the required libraries:**
    ```sh
    pip install numpy pandas scikit-learn tensorflow matplotlib seaborn tabulate
    ```

3. **Place the CSV file:**
   - Ensure the `google-ethereum-transactions.csv` file is in the same directory as the script.

## Running the Script

1. **Run the script:**
    ```sh
    python3 your_script_name.py
    ```

2. **Expected Output:**
   - The script will print a human-readable table of generated synthetic transactions.

## Script Details

### Data Preprocessing

The script loads and preprocesses Ethereum transaction data, filling missing values and converting necessary columns to numeric types. It also extracts time-based features and calculates additional features such as gas usage efficiency and gas price to value ratio.

### VAE Model

The VAE model is defined and trained to learn the distribution of the transaction data. The encoder reduces the data to a latent space, and the decoder reconstructs the data from this latent space.

### Generating Synthetic Data

After training, the script generates new synthetic transactions by sampling from the latent space and decoding these samples back to the original feature space.

### Output

The generated synthetic transactions are formatted and printed in a readable table format, showcasing the capabilities of the VAE model in generating realistic transaction data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.