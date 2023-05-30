import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("lenovo.csv")

X = df[['High']]
y = df['Open']

regression_model = LinearRegression()

regression_model.fit(X, y)

y_pred = regression_model.predict(X)

print("Predict", y_pred)

RSS = np.sum((y - y_pred) ** 2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y, y_pred)

plt.scatter(X, y, color='b', label='Data Points')
plt.plot(X, y_pred, color='r', label='Regression Line')
plt.xlabel('High')
plt.ylabel('Open')
plt.title('Linear Regression')
plt.legend()
plt.show()
print("Resultados:")
print("RSS:", RSS)
print("MSE:", MSE)
print("RMSE:", RMSE)
print("R_squared:", R_squared)

