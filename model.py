import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import shap

# Load the CSV file with the appropriate delimiter and handle quoting
data = pd.read_csv('updated_ethereum_transactions.csv', delimiter=',', quotechar='"', on_bad_lines='skip')

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

# Convert the processed features back into a DataFrame
num_columns = ['Received', 'Sent', 'Fee']
cat_columns = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(['From address', 'To address'])
all_columns = np.hstack([num_columns, cat_columns])

X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_columns)

# Use SMOTE with adjusted k_neighbors to oversample the minority class
smote = SMOTE(random_state=42, k_neighbors=2)
X_res, y_res = smote.fit_resample(X_preprocessed_df, labels)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Creating a pipeline
pipeline = Pipeline(steps=[('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Training the model
pipeline.fit(X_train, y_train)

# Making predictions
y_pred = pipeline.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Feature Importances
# Fit the full pipeline including preprocessing for feature importance extraction
full_pipeline = Pipeline(steps=[('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

full_pipeline.fit(X_train, y_train)

feature_importances = full_pipeline.named_steps['classifier'].feature_importances_
feature_names = all_columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest')
plt.show()

# Predict on New Data
new_data = pd.DataFrame({
    'Received': [0.01],
    'Sent': [0.02],
    'Fee': [0.0001],
    'From address': ['some_address'],
    'To address': ['another_address']
})

# Preprocess the new data
new_data_preprocessed = preprocessor.transform(new_data)
new_data_preprocessed_df = pd.DataFrame(new_data_preprocessed, columns=all_columns)
prediction = pipeline.named_steps['classifier'].predict(new_data_preprocessed_df)
print("Predicted Operation:", prediction)

# Cross-Validation
cv_scores = cross_val_score(full_pipeline, X_res, y_res, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Confusion Matrix')
plt.show()

# Hyperparameter Tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_res, y_res)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# # Model Interpretability with SHAP
# full_pipeline.fit(X_preprocessed_df, labels)
# explainer = shap.TreeExplainer(full_pipeline.named_steps['classifier'])
# shap_values = explainer.shap_values(preprocessor.transform(data[features]))

# # Plot SHAP summary plot
# shap.summary_plot(shap_values, feature_names=feature_names)
