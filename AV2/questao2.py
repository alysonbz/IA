### Questão 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV2\Sample - Superstore.csv', encoding='latin-1')

#Utilizando o atributo mais relevante calculado na questão 1, implemente uma regressão linear utilizando
#somente este atributo mais relevante, para predição do atributo alvo determinado na questão 1 também.
X = df['Sales'].values.reshape(-1, 1)
y = df['Profit'].values
reg = LinearRegression()
reg.fit(X, y)
pred = reg.predict(X)

#Mostre o gráfico da reta de regressão em conjunto com a nuvem de atributo.
plt.scatter(X, y, color='#4682b4', label='Dados')
plt.plot(X, pred, color='#ff1493', linewidth=2)
plt.title('Regressão Linear', fontweight='bold')
plt.xlabel('Sales', fontweight='bold')
plt.ylabel('Profit', fontweight='bold')
plt.show()

#Determine também os valores:
#RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante.
rss = np.sum((pred - X) ** 2)
print("RSS:", rss)
mse = mean_squared_error(X, pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(X, pred)
print("R-squared:", r2)

#Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.


