import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data.csv')

# Separate boolean columns for preprocessing
boolean_cols = df.select_dtypes(include=bool).columns
for col in boolean_cols:
    df[col] = df[col].astype(str)

# Prepare data
X = df.iloc[:, :-1].apply(lambda col: LabelEncoder().fit_transform(col) if col.name not in boolean_cols else col.apply(lambda x: 1 if x.lower() == 'true' else 0)).values
y = np.vectorize(lambda x: 1 if str(x).lower() == 'true' else 0)(df.iloc[:, -1].values)

# Train classifier
clf = CategoricalNB().fit(X, y)
print(X)
# Prediction function
def predict_outcome():
    labels = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    user_input = [input(f"Enter {label} ({', '.join(df[label].unique())}): ") for label in labels]
    user_input_encoded = np.array([LabelEncoder().fit_transform(user_input)]).reshape(1, -1)
    prediction = clf.predict(user_input_encoded)[0]
    print(f'Predicted Outcome: {prediction}')

# Make prediction
predict_outcome()
