import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path =  os.path.join('..', '..', 'data', 'transactions', 'google-ethereum-transactions.csv')  
data = pd.read_csv(file_path)

# Fill missing values (for simplicity, we'll fill numeric columns with 0)
data.fillna(0, inplace=True)

# Create labels: 1 for Credit (value > 0), 0 for Debit (value == 0)
data['label'] = data['value'].apply(lambda x: 1 if float(x) > 0 else 0)

# Select features for modeling
features = ['value', 'gas', 'gas_price']
X = data[features].astype(float)
y = data['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report_output)

# Label the transactions in the original dataset
data['predicted_label'] = rf_model.predict(X)

# Save the labeled dataset to a new CSV file
labeled_file_path = 'labeled_google-ethereum-transactions.csv'
data.to_csv(labeled_file_path, index=False)

# Visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='predicted_label', data=data)
plt.title('Count of Credits and Debits')
plt.xlabel('Transaction Type (0 = Debit, 1 = Credit)')
plt.ylabel('Count')
plt.savefig('count_of_credits_and_debits.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data, x='value', hue='predicted_label', multiple='stack', bins=50)
plt.title('Distribution of Transaction Values by Type')
plt.xlabel('Transaction Value')
plt.ylabel('Count')
plt.legend(title='Transaction Type', labels=['Debit', 'Credit'])
plt.savefig('distribution_of_transaction_values.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='gas', y='value', hue='predicted_label', data=data)
plt.title('Gas vs. Transaction Value')
plt.xlabel('Gas')
plt.ylabel('Transaction Value')
plt.legend(title='Transaction Type', labels=['Debit', 'Credit'])
plt.savefig('gas_vs_transaction_value.png')
plt.show()
