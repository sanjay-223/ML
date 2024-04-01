import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore", message="The least populated class in y has only 1 members, which is less than n_splits=5.")

# Function to extract features
def extract_features(url):
    return [len(url), url.count('http'), url.count('https')]

# Load dataset
dataset = pd.read_csv('malicious_phish.csv', encoding='ISO-8859-1').sample(frac=0.1, random_state=42)
X = np.array(dataset['url'].apply(extract_features).tolist())
y = dataset['type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test

scaler = StandardScaler()

# Create an instance of SimpleImputer to impute missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on X_train and transform X_train and X_test
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Now, you can scale the data using StandardScaler
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize CatBoostClassifier model
catboost_model = CatBoostClassifier(iterations=100, random_state=42, verbose=0)

# Train CatBoostClassifier model
catboost_model.fit(X_train_scaled, y_train)

# Model evaluation for CatBoostClassifier
y_pred_catboost = catboost_model.predict(X_test_scaled)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
precision_catboost = precision_score(y_test, y_pred_catboost, average='macro', zero_division=1)
recall_catboost = recall_score(y_test, y_pred_catboost, average='macro', zero_division=1)
f1_catboost = f1_score(y_test, y_pred_catboost, average='macro', zero_division=1)
conf_matrix_catboost = confusion_matrix(y_test, y_pred_catboost)

# Print evaluation metrics for CatBoostClassifier
print("CatBoost Metrics:")
print("Testing Accuracy:", accuracy_catboost)
print("Precision:", precision_catboost)
print("Recall:", recall_catboost)
print("F1 Score:", f1_catboost)
print("Confusion Matrix:")
print(conf_matrix_catboost)
