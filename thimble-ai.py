import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the CSV file
file_path = '/home/mike/Machine-learning-lectures/data/google-ethereum-transactions.csv'
data = pd.read_csv(file_path)

# Fill missing values
data.fillna(0, inplace=True)

# Convert necessary columns to numeric types, coercing errors
data['value'] = pd.to_numeric(data['value'], errors='coerce').fillna(0)
data['gas'] = pd.to_numeric(data['gas'], errors='coerce').fillna(0)
data['gas_price'] = pd.to_numeric(data['gas_price'], errors='coerce').fillna(0)
data['transaction_type'] = pd.to_numeric(data['transaction_type'], errors='coerce').fillna(0)

# Convert block_timestamp to datetime
data['block_timestamp'] = pd.to_datetime(data['block_timestamp'])

# Extract time-based features
data['hour'] = data['block_timestamp'].dt.hour
data['day_of_week'] = data['block_timestamp'].dt.dayofweek
data['day_of_month'] = data['block_timestamp'].dt.day

# Calculate gas usage efficiency
data['gas_usage_efficiency'] = np.where(data['gas'] != 0, data['value'] / data['gas'], 0)

# Calculate gas price to value ratio
data['gas_price_value_ratio'] = np.where(data['value'] != 0, data['gas_price'] / data['value'], 0)

# Calculate address frequency features
data['from_address_frequency'] = data.groupby('from_address')['from_address'].transform('count')
data['to_address_frequency'] = data.groupby('to_address')['to_address'].transform('count')

# Define features and target
features = ['gas_price', 'value', 'gas', 'transaction_type', 'hour', 'day_of_week', 'day_of_month', 
            'gas_usage_efficiency', 'gas_price_value_ratio', 'from_address_frequency', 'to_address_frequency']
target = 'transaction_type'  # Assuming this is the target for classification

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(data[features])
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Dummy new transaction data for demonstration
new_transaction_data = {
    'gas_price': [8.5e+09, 9.0e+09],
    'value': [3.0e+18, 3.5e+18],
    'gas': [200000, 150000],
    'transaction_type': [1, 2],
    'hour': [10, 14],
    'day_of_week': [3, 4],
    'day_of_month': [15, 18],
    'gas_usage_efficiency': [1.5e+13, 2.0e+13],
    'gas_price_value_ratio': [2.5e+06, 2.7e+06],
    'from_address_frequency': [50, 70],
    'to_address_frequency': [30, 40]
}

new_data = pd.DataFrame(new_transaction_data)
new_data_scaled = scaler.transform(new_data[features])
predicted_labels = clf.predict(new_data_scaled)

# Create a DataFrame for the output
output_df = new_data.copy()
output_df['Predicted Label'] = predicted_labels
output_df['Entry Type'] = output_df['Predicted Label'].apply(lambda x: "Credit" if x == 1 else "Debit")

# Print the table
print("\nGenerated Accounting Entries (Random Forest):")
print(tabulate(output_df, headers='keys', tablefmt='pretty'))

# Optional: Using Deep Learning (Example with a Simple Neural Network)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nLoss: {loss}, Accuracy: {accuracy}')

# Generate new transactions using the neural network
predicted_labels_nn = (model.predict(new_data_scaled) > 0.5).astype("int32")

# Create a DataFrame for the neural network output
output_df_nn = new_data.copy()
output_df_nn['Predicted Label'] = predicted_labels_nn
output_df_nn['Entry Type'] = output_df_nn['Predicted Label'].apply(lambda x: "Credit" if x == 1 else "Debit")

# Print the table for neural network predictions
print("\nGenerated Accounting Entries (Neural Network):")
print(tabulate(output_df_nn, headers='keys', tablefmt='pretty'))
