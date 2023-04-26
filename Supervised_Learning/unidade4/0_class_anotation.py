import matplotlib.pyplot as plt
from src.utils import load_diabetes_clean_dataset
from sklearn.linear_model import LinearRegression

diabetes_df = load_diabetes_clean_dataset()

# Tratando valores nulos
# diabetes_df = diabetes_df.loc[diabetes_df['bmi'] != 0]
# diabetes_df.drop(["bmi"] == 0, inplace=True)

print(diabetes_df.head(5))

# Separando o objeto target
X = diabetes_df.drop("glucose", axis = 1).values
y = diabetes_df["glucose"].values
print(type(X), type(y))

# Realizando uma regressão utilizando 1 único atributo
X_bmi = X[:, 3]
print(y.shape, X_bmi.shape)

X_bmi = X_bmi.reshape(-1, 1)
print(X_bmi.shape)

# Plotando o gráfico da glicose em função do IMC
plt.scatter(X_bmi, y)
plt.ylabel("Glicose no sangue (mg/dl)")
plt.xlabel("Índice de massa corporal")
plt.show()

# Treinando um modelo de regressão
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel("Glicose no sangue (mg/dl)")
plt.xlabel("Índice de massa corporal")
plt.show()