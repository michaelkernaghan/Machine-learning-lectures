import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Add, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path =  "google-ethereum-transactions.csv"
data = pd.read_csv(file_path)

# Fill missing values
data.fillna(0, inplace=True)

# Ensure numeric columns are of correct type
data['value'] = pd.to_numeric(data['value'], errors='coerce').fillna(0)
data['gas'] = pd.to_numeric(data['gas'], errors='coerce').fillna(0)
data['gas_price'] = pd.to_numeric(data['gas_price'], errors='coerce').fillna(0)
data['transaction_type'] = pd.to_numeric(data['transaction_type'], errors='coerce').fillna(0)

# Map transaction type numbers to Ethereum transaction types
transaction_type_mapping = {
    1: 'Standard Transaction',
    2: 'Contract Creation',
    3: 'Contract Execution',
    4: 'Token Transfer'
}
data['transaction_type'] = data['transaction_type'].map(transaction_type_mapping)

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
features = ['gas_price', 'value', 'gas', 'hour', 'day_of_week', 'day_of_month', 
            'gas_usage_efficiency', 'gas_price_value_ratio', 'from_address_frequency', 'to_address_frequency']
target = 'transaction_type'

# Prepare the data
X = data[features]
y = data[target]

# Encode the target
y_encoded = pd.get_dummies(y).values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check class distribution
print("Class distribution in training data:")
print(y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X_scaled, y_encoded, data.index, test_size=0.2, random_state=42)

# Reshape the data for the transformer input
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the Transformer model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = Flatten()(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    
    outputs = Dense(y_train.shape[1], activation="softmax")(x)
    return Model(inputs, outputs)

input_shape = X_train.shape[1:]
model = build_model(
    input_shape,
    head_size=64,
    num_heads=2,
    ff_dim=64,
    num_transformer_blocks=2,
    mlp_units=[128],
    dropout=0.1,
    mlp_dropout=0.1,
)

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Handle undefined metrics in classification report
labels = list(transaction_type_mapping.keys())  # Explicitly specify the labels
print(classification_report(y_test_classes, y_pred_classes, target_names=list(transaction_type_mapping.values()), labels=labels, zero_division=0))

# Save the test data with predicted and actual labels
test_data_with_predictions = data.iloc[test_indices].copy()
test_data_with_predictions['actual'] = y_test_classes
test_data_with_predictions['predicted'] = y_pred_classes
test_data_with_predictions.to_csv('test_data_with_predictions.csv', index=False)

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes, labels=labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(transaction_type_mapping.values()), yticklabels=list(transaction_type_mapping.values()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('transaction_type_confusion_matrix.png')
plt.close()

# Plot test results
plt.figure(figsize=(10, 7))
sns.scatterplot(data=test_data_with_predictions, x='gas_price', y='value', hue='predicted', palette='viridis', style='actual')
plt.xlabel('Gas Price')
plt.ylabel('Value')
plt.title('Test Data: Actual vs. Predicted Transaction Types')
plt.legend(title='Predicted Type')
plt.savefig('test_data_results.png')
plt.close()

print("Confusion matrix plot saved as image file: transaction_type_confusion_matrix.png")
print("Test data with predictions saved as CSV file: test_data_with_predictions.csv")
print("Test data results plot saved as image file: test_data_results.png")
