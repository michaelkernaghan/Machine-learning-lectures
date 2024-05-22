import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Load the CSV file with the appropriate delimiter and handle quoting
data = pd.read_csv('ethereum_transactions.csv', delimiter=',', quotechar='"', on_bad_lines='skip')

# Print column names to verify
print("Columns in the dataframe:", data.columns)

# Replace commas with periods in numerical columns and convert to float
data['Received'] = data['Received'].str.replace(',', '.').astype(float)
data['Sent'] = data['Sent'].str.replace(',', '.').astype(float)
data['Fee'] = data['Fee'].str.replace(',', '.').astype(float)

# Ensure 'Operation' column exists
if 'Operation' not in data.columns:
    raise ValueError("'Operation' column is missing. Available columns are:", data.columns)

# Ensure all necessary columns are present for features and labels
required_columns = ['Block level', 'Datetime', 'Operation', 'Received', 'From address', 'Sent', 'Fee', 'To address', 'Explorer link']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Features and Labels
features = ['Received', 'Sent', 'Fee', 'From address', 'To address']
labels = data['Operation']

# Preprocessing: Handling missing values, categorical data, and scaling numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), ['Received', 'Sent', 'Fee']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['From address', 'To address'])
    ])

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(data[features], labels, test_size=0.2, random_state=42)

# Creating a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Training the model
pipeline.fit(X_train, y_train)

# Making predictions
y_pred = pipeline.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
