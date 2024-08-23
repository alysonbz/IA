from src.utils import load_diabetes_clean_dataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Predição da intensidade de glicose no corpo
diabetes_df = load_diabetes_clean_dataset()
print(diabetes_df.head())

# Separação do objeto target
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values


# Realizar uma regressão utilizando 1 único atributo
X_bmi = X[:, 3]
print(y.shape, X_bmi.shape)

X_bmi = X_bmi.reshape(-1, 1)
print(X_bmi.shape)


# Plot do gráfico da glicose em função do IMC
plt.scatter(X_bmi, y)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()


# Treinando um modelo de regressão
reg = LinearRegression()
reg.fit(X_bmi, y)

predictions = reg.predict(X_bmi)

plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()