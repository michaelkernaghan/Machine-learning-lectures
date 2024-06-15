import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path =  os.path.join('..', '..', 'data', 'transactions', 'google-ethereum-transactions.csv') 
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

# Cyclic transformation of time features
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

# Calculate gas usage efficiency
data['gas_usage_efficiency'] = np.where(data['gas'] != 0, data['value'] / data['gas'], 0)

# Calculate gas price to value ratio
data['gas_price_value_ratio'] = np.where(data['value'] != 0, data['gas_price'] / data['value'], 0)

# Calculate address frequency features
data['from_address_frequency'] = data.groupby('from_address')['from_address'].transform('count')
data['to_address_frequency'] = data.groupby('to_address')['to_address'].transform('count')

# Define features and target
features = ['gas_price', 'value', 'gas', 'transaction_type', 'hour', 'day_of_week', 'day_of_month', 
            'gas_usage_efficiency', 'gas_price_value_ratio', 'from_address_frequency', 'to_address_frequency',
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
target = 'is_anomaly'

# Ensure 'is_anomaly' column exists; otherwise, create it for demonstration purposes
if 'is_anomaly' not in data.columns:
    data['is_anomaly'] = np.random.randint(0, 2, data.shape[0])  # Randomly assigning anomalies for demonstration

# Split the data
X = data[features]
y = data[target]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Additional Analysis: Hourly Transaction Volume
plt.figure(figsize=(14, 7))
sns.countplot(x='hour', data=data)
plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')
plt.title('Hourly Transaction Volume')
plt.savefig('hourly_transaction_volume.png')
plt.close()

# Additional Analysis: Day of Week Transaction Volume
plt.figure(figsize=(14, 7))
sns.countplot(x='day_of_week', data=data)
plt.xlabel('Day of Week')
plt.ylabel('Number of Transactions')
plt.title('Day of Week Transaction Volume')
plt.savefig('day_of_week_transaction_volume.png')
plt.close()

print("Confusion matrix plot saved as image file: confusion_matrix.png")
print("Hourly transaction volume plot saved as image file: hourly_transaction_volume.png")
print("Day of week transaction volume plot saved as image file: day_of_week_transaction_volume.png")
