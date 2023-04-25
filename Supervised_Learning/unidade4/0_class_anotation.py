from src.utils import load_diabetes_clean_dataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


diabetes_df = load_diabetes_clean_dataset()

print(diabetes_df.head(n=5))

x = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df['glucose'].values
print(type(x), type(y))

X_bmi = x[:,3]
print(y.shape, X_bmi.shape)

X_bmi = X_bmi.reshape(-1, 1)
print(X_bmi.shape)

'''plt.scatter(X_bmi, y)
plt.ylabel("blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
#plt.show()'''


reg = LinearRegression()
reg.fit(X_bmi, y)
predicitions = reg.predict(X_bmi)
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predicitions)
plt.ylabel("Blood Glucose (mg/dl")
plt.xlabel("Body Mass Index")
plt.show()
