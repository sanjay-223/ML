import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb

# Read the dataset
df = pd.read_csv("../URL_data.csv")

# Split dataset into features (X) and target variable (y)
X = df.drop(columns=['target'])
y = df['target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize LightGBM classifier
lgb_classifier = lgb.LGBMClassifier(n_estimators=100, random_state=42)

# Train LightGBM classifier
lgb_classifier.fit(X_train, y_train)

# Predictions
y_pred = lgb_classifier.predict(X_test)

# Model evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
