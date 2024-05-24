import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the CSV file
file_path = 'google-ethereum-transactions.csv'
data = pd.read_csv(file_path)

# Fill missing values (for simplicity, we'll fill numeric columns with 0)
data.fillna(0, inplace=True)

# Define the specific account address to identify credits and debits
specific_address = '0xe79b0195ff39ba93915f9d78b740d571cfe52214'

# Create labels: 'Credit' if to_address is the specific address, 'Debit' if from_address is the specific address
data['label'] = data.apply(lambda row: 'Credit' if row['to_address'] == specific_address else ('Debit' if row['from_address'] == specific_address else 'Other'), axis=1)

# Filter out rows labeled as 'Other'
data = data[data['label'] != 'Other']

# Select features for modeling (add more features as needed)
features = ['value', 'gas', 'gas_price']
X = data[features].astype(float)
y = data['label']

# Encode labels: 'Credit' -> 1, 'Debit' -> 0
y = y.map({'Debit': 0, 'Credit': 1})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=1)

# Predict on the test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Decode labels: 1 -> 'Credit', 0 -> 'Debit'
y_test_decoded = y_test.map({0: 'Debit', 1: 'Credit'})
y_pred_decoded = pd.Series(y_pred).map({0: 'Debit', 1: 'Credit'})

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test_decoded, y_pred_decoded)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report_output)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('learning_curves.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curves.png')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

# Plot feature distributions for different classes
plt.figure(figsize=(12, 8))
sns.histplot(data, x='value', hue='label', multiple='stack')
plt.title('Transaction Value Distribution')
plt.xlabel('Transaction Value')
plt.ylabel('Count')
plt.savefig('transaction_value_distribution.png')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(data, x='gas', hue='label', multiple='stack')
plt.title('Gas Distribution')
plt.xlabel('Gas')
plt.ylabel('Count')
plt.savefig('gas_distribution.png')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(data, x='gas_price', hue='label', multiple='stack')
plt.title('Gas Price Distribution')
plt.xlabel('Gas Price')
plt.ylabel('Count')
plt.savefig('gas_price_distribution.png')
plt.show()

# Explain the model's predictions using SHAP with sampling
background = shap.sample(X_train, 100)  # Sample 100 data points for the background
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X_test, nsamples=100)

# Plot SHAP values
shap.summary_plot(shap_values, X_test, feature_names=features)

# Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train SVM
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate models
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f'Logistic Regression Accuracy: {accuracy_lr}')
print(f'SVM Accuracy: {accuracy_svm}')
