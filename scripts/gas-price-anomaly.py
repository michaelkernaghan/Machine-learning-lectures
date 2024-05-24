import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'google-ethereum-transactions.csv'
data = pd.read_csv(file_path)

# Fill missing values (for simplicity, we'll fill numeric columns with 0)
data.fillna(0, inplace=True)

# Focus on the 'gas_price' feature
gas_price = data[['gas_price']].astype(float)

# Standardize the gas_price feature
scaler = StandardScaler()
gas_price_scaled = scaler.fit_transform(gas_price)

# Split the data into training and test sets
X_train, X_test, X_train_index, X_test_index = train_test_split(gas_price_scaled, data.index, test_size=0.2, random_state=42)

# Build the autoencoder model
input_dim = X_train.shape[1]
autoencoder = Sequential()
autoencoder.add(Input(shape=(input_dim,)))
autoencoder.add(Dense(16, activation='relu'))
autoencoder.add(Dense(8, activation='relu'))
autoencoder.add(Dense(16, activation='relu'))
autoencoder.add(Dense(input_dim, activation='linear'))

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Predict the reconstruction
X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)

# Calculate the reconstruction error
train_mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
test_mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Determine the threshold for anomaly detection
threshold = np.mean(train_mse) + 3 * np.std(train_mse)

# Identify anomalies
train_anomalies = train_mse > threshold
test_anomalies = test_mse > threshold

# Add the anomaly information to the original data
data['is_anomaly'] = False
data.loc[X_test_index[test_anomalies], 'is_anomaly'] = True

# Save the original transactions that are marked as anomalies
anomalous_transactions = data[data['is_anomaly']]
anomalous_transactions.to_csv('anomalous_transactions.csv', index=False)

# Plot and save the reconstruction error distribution
plt.figure(figsize=(12, 8))
sns.histplot(train_mse, kde=True, label='Train')
sns.histplot(test_mse, kde=True, label='Test', color='r')
plt.axvline(threshold, color='k', linestyle='--', label='Threshold')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.legend()
plt.savefig('reconstruction_error_distribution.png')
plt.close()

# Plot and save the anomalies in gas_price
plt.figure(figsize=(12, 8))
sns.scatterplot(x=data.index, y=data['gas_price'], hue=data['is_anomaly'], palette=['blue', 'red'])
plt.title('Gas Price Anomalies')
plt.xlabel('Index')
plt.ylabel('Gas Price')
plt.savefig('gas_price_anomalies.png')
plt.close()

print("Plots saved as image files:")
print("reconstruction_error_distribution.png")
print("gas_price_anomalies.png")
print("Anomalous transactions saved to: anomalous_transactions.csv")
