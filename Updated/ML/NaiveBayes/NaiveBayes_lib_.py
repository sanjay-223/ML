import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')
print(df)
print(df.columns)
df['Windy'] = df['Windy'].astype(str)

# df = df.drop(df.columns[df.columns.str.contains('Unnamed')], axis=1)

# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# def encode_boolean(value):
#     return 1 if str(value).lower() == 'true' else 0

# label_encoders = [LabelEncoder() for _ in range(X.shape[1])]
# X_encoded = [le.fit_transform(X[:, i]) if i != X.shape[1] - 1 else np.array([encode_boolean(val) for val in X[:, i]]) for i, le in enumerate(label_encoders)]
# X_encoded = np.array(list(zip(*X_encoded)))
# X_encoded = 

label_encoders = {}

# Iterate over columns except 'Play Golf'
for col in df.columns:  # Exclude the last column ('Play Golf')
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])


X = df.drop(columns=['Play Golf']).values
y = df['Play Golf']

clf = CategoricalNB()
clf.fit(X, y)

def predict_outcome(label_encoders, clf):
    user_input = {
    'Outlook' : 'Sunny',#input("Enter Outlook (e.g., 'Sunny', 'Overcast', 'Rainy'): "),
    'Temp' :'Hot',# input("Enter Temperature (e.g., 'Hot', 'Mild', 'Cool'): "),
    'Humidity' :'High', #input("Enter Humidity (e.g., 'High', 'Normal'): "),
    'Windy' : 'True', #input("Enter Windy (True/False): ")
    }

    user_input_encoded = []
    for col in df.columns[:-1]:  # Exclude the last column ('Play Golf')
        user_input_encoded.append(label_encoders[col].transform([user_input[col]])[0])

    prediction = clf.predict([user_input_encoded])[0]
    return prediction

y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f'Accuracy: {accuracy}')
print(predict_outcome(label_encoders,clf))