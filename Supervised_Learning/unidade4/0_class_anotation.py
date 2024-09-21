from src.utils import load_diabetes_clean_dataset
from sklearn.linear_model import LinearRegression

diabetes_df = load_diabetes_clean_dataset()
print(diabetes_df)
print(diabetes_df.head())
X = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df['glucose'].values
print(type(X), type(y))
X_bmi = X[:,3]
print(y.shape, X_bmi.shape)
X_bmi = X_bmi.reshape(-1, 1)
print(X_bmi.shape)

import matplotlib.pyplot as plt
plt.scatter(X_bmi, y)
plt.ylabel('Blood Glucose (mg/dl)')
plt.xlabel('body mass index')
plt.show()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel('Blood Glucose (mg/dl)')
plt.xlabel('Body Mass Index')
plt.show()


