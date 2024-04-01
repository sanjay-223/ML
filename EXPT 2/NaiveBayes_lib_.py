import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')
df = df.drop(df.columns[df.columns.str.contains('Unnamed')], axis=1)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

def encode_boolean(value):
    return 1 if str(value).lower() == 'true' else 0

label_encoders = [LabelEncoder() for _ in range(X.shape[1])]
X_encoded = [le.fit_transform(X[:, i]) if i != X.shape[1] - 1 else np.array([encode_boolean(val) for val in X[:, i]]) for i, le in enumerate(label_encoders)]
X_encoded = np.array(list(zip(*X_encoded)))

clf = CategoricalNB()
clf.fit(X_encoded, y)

def predict_outcome():
    outlook = input("Enter Outlook (e.g., 'Sunny', 'Overcast', 'Rainy'): ")
    temperature = input("Enter Temperature (e.g., 'Hot', 'Mild', 'Cool'): ")
    humidity = input("Enter Humidity (e.g., 'High', 'Normal'): ")
    windy = input("Enter Windy (True/False): ")

    user_input_encoded = [le.transform([val])[0] if i != X.shape[1] - 1 else encode_boolean(val) for i, (le, val) in enumerate(zip(label_encoders, [outlook, temperature, humidity, windy]))]
    user_input_encoded = np.array(user_input_encoded).reshape(1, -1)

    prediction = clf.predict(user_input_encoded)[0]
    print(f'Predicted Outcome: {prediction}')

y_pred = clf.predict(X_encoded)
accuracy = accuracy_score(y, y_pred)
print(f'Accuracy: {accuracy}')
predict_outcome()