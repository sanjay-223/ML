import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Salary_dataset.csv')
data.drop(columns=['Sno'],inplace=True)
X = data['Exp'].values.reshape(-1,1)
y = data['Sal'].values //1000
y = y.reshape(-1,1)

model = LinearRegression()
model.fit(X, y)

predicted_y = model.predict(X)

mse = mean_squared_error(y, predicted_y)
r2 = r2_score(y, predicted_y)

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, predicted_y, color='red', label='Linear Regression')
plt.title('Simple Linear Regression (sklearn)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)
