import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


clean_dataset = pd.read_csv('./dataset/Clean_Dataset.csv')
clean_dataset = clean_dataset.drop(clean_dataset.columns[0], axis=1)

clean_dataset["class"] = clean_dataset["class"].replace("Economy", 0)
clean_dataset["class"] = clean_dataset["class"].replace("Business", 1)

X = clean_dataset[['class']] # Atributo mais relevante
y = clean_dataset['price']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Visualizar a reta de regressão com um gráfico de dispersão
plt.scatter(X_test, y_test, color='green', label='Dados reais')
plt.plot(X_test, y_pred, color='red', label='Regressão linear')
plt.title('Relação entre Class e Price')
plt.xlabel('class')
plt.ylabel('Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

# Calcular RSS
rss = sum((y_test - y_pred) ** 2)

# Calcular MSE e RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Calcular R²
r2 = r2_score(y_test, y_pred)

# Exibir os resultados
print(f"RSS: {rss}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")