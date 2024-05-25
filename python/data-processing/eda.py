import pandas as pd
import os

# Load the CSV file
file_path =  os.path.join('..', '..', 'data', 'transactions', 'google-ethereum-transactions.csv')
data = pd.read_csv(file_path)

# Load the anomalous transactions
anomalous_transactions = pd.read_csv('anomalous_transactions.csv')

# Print details of the anomalous transactions
print("Anomalous Transactions:")
print(anomalous_transactions)

# Compare with normal transactions
normal_transactions = data[~data.index.isin(anomalous_transactions.index)].head(len(anomalous_transactions))

print("Normal Transactions:")
print(normal_transactions)

# Save detailed analysis for further inspection
anomalous_transactions.to_csv('anomalous_transactions_detailed.csv', index=False)
normal_transactions.to_csv('normal_transactions_detailed.csv', index=False)

print("Detailed transaction data saved as CSV files:")
print("anomalous_transactions_detailed.csv")
print("normal_transactions_detailed.csv")
