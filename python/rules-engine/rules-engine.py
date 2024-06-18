import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt



# There's a dataset in Google itself (BigQuery): https://console.cloud.google.com/marketplace/details/ethereum/crypto-ethereum-blockchain

# Currently, https://github.com/blockchain-etl/ethereum-etl is still active.

# Documentation is at http://ethereum-etl.readthedocs.io

# A 2018 article by its creator: https://evgemedvedev.medium.com/ethereum-blockchain-on-google-bigquery-283fb300f579

# Other project contacts in case the Github disappears:

#     https://t.me/BlockchainETL
#     https://discord.gg/tRKG7zGKtF


# Load the CSV file with the appropriate delimiter and handle quoting
file_path =  "updated_ethereum_transactions.csv"
data = pd.read_csv(file_path, delimiter=',', quotechar='"', on_bad_lines='skip')

# Ensure all entries in 'Received', 'Sent', and 'Fee' columns are strings before replacing
data['Received'] = data['Received'].astype(str).str.replace(',', '.').astype(float)
data['Sent'] = data['Sent'].astype(str).str.replace(',', '.').astype(float)
data['Fee'] = data['Fee'].astype(str).str.replace(',', '.').astype(float)

# Ensure 'Operation' column exists
if 'Operation' not in data.columns:
    raise ValueError("'Operation' column is missing. Available columns are:", data.columns)

# Ensure all necessary columns are present for features and labels
required_columns = ['Block level', 'Datetime', 'Operation', 'Received', 'From address', 'Sent', 'Fee', 'To address', 'Explorer link']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Test cases to verify labeling logic
test_data = pd.DataFrame({
    'Received': [0.01, 0.0, np.nan, np.nan],
    'Sent': [np.nan, np.nan, 0.03, 5],
    'Fee': [0.0001, 0.0001, 0.0001, 0.0001]
})

def label_transaction(row):
    if pd.notnull(row['Received']) and row['Received'] >= 0:
        return 'credit'
    elif pd.notnull(row['Sent']) and row['Sent'] <= 0:
        return 'debit'
    else:
        return 'other'

# Apply the labeling function to test data
test_data['Transaction_Type'] = test_data.apply(label_transaction, axis=1)

# Display the test data with the new labels
test_data

data['Transaction_Type'] = data.apply(label_transaction, axis=1)

# Verify the updated distribution of the labels
label_distribution_updated = data['Transaction_Type'].value_counts()
print(label_distribution_updated)

# Features and Labels
features = ['Received', 'Sent', 'Fee', 'From address', 'To address']
labels = data['Transaction_Type']

# Preprocessing: Handling missing values and scaling numerical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, ['Received', 'Sent', 'Fee']),
        ('cat', categorical_pipeline, ['From address', 'To address'])
    ])

# Apply preprocessing to the entire dataset
X_preprocessed = preprocessor.fit_transform(data[features])
labels = labels.reset_index(drop=True)

# Get column names after transformation
num_columns = ['Received', 'Sent', 'Fee']
cat_columns = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['From address', 'To address'])
all_columns = np.hstack([num_columns, cat_columns])

X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_columns)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, labels, test_size=0.2, random_state=42)

# Creating a pipeline that includes preprocessing
full_pipeline = Pipeline(steps=[
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# Training the model
full_pipeline.fit(X_train, y_train)

# Making predictions
y_pred = full_pipeline.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred, zero_division=0))

# Feature Importances
feature_importances = full_pipeline.named_steps['classifier'].feature_importances_

# Plot top 10 feature importances
top_n = 10
top_features = np.argsort(feature_importances)[-top_n:]
plt.figure(figsize=(10, 6))
plt.barh(np.array(all_columns)[top_features], feature_importances[top_features])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances from Random Forest')
plt.savefig('feature_importances.png')  # Save the plot as an image

# Predict on New Data
new_data = pd.DataFrame({
    'Received': [0.01, np.nan],
    'Sent': [np.nan, 0.02],
    'Fee': [0.0001, 0.0001],
    'From address': ['some_address', 'another_address'],
    'To address': ['another_address', 'some_address']
})

# Preprocess the new data using the same preprocessing pipeline
new_data_preprocessed = preprocessor.transform(new_data)
new_data_preprocessed_df = pd.DataFrame(new_data_preprocessed, columns=all_columns)
prediction = full_pipeline.predict(new_data_preprocessed_df)
print("Predicted Transaction Types:", prediction)

# Cross-Validation
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the confusion matrix as an image

# Hyperparameter Tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Rules Engine for Mapping Transactions to Accounting Entries
def create_accounting_entry(transaction, entry_type, account, amount):
    print(f"Creating accounting entry: {entry_type} {amount} to {account} for transaction {transaction['hash']}")

# Example new transactions
new_transactions = [
    {'hash': '0x1', 'value': 0.01, 'Received': 0.01, 'Sent': np.nan, 'Fee': 0.0001, 'From address': 'some_address', 'To address': 'another_address'},
    {'hash': '0x2', 'value': 0.02, 'Received': np.nan, 'Sent': 0.02, 'Fee': 0.0001, 'From address': 'another_address', 'To address': 'some_address'}
    # Add more transactions as needed
]

# Preprocess new transactions
new_transactions_df = pd.DataFrame(new_transactions)
new_transactions_features = preprocessor.transform(new_transactions_df[features])
new_transactions_features_df = pd.DataFrame(new_transactions_features, columns=all_columns)

# Predict transaction types
predictions = full_pipeline.predict(new_transactions_features_df)

# Apply rules engine to generate accounting entries
for tx, pred in zip(new_transactions, predictions):
    if pred == 'debit':
        create_accounting_entry(tx, 'debit', 'asset', tx['value'])
    elif pred == 'credit':
        create_accounting_entry(tx, 'credit', 'cash', tx['value'])
    else:
        print(f"Transaction {tx['hash']} is categorized as 'other'")

