import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

df = pd.read_csv("../URL_data.csv")

# print(df.shape)
# features_to_include = np.array([1, 0, 1 ,0, 1, 1, 0, 0, 0, 1 ,1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0,0])
# selected_features = df.loc[:,features_to_include.astype(bool)]

# print(selected_features.shape)

X = df.drop(columns=['target'])#selected_features
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

from sklearn.metrics import confusion_matrix

# Assuming y_test and y_pred are your actual and predicted labels, respectively
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)