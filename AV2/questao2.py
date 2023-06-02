from questao1 import database
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print('DataFrame Atualizada: \n',database)

#Utilizando o atributo mais relevante calculado na questão 1, implemente uma regressão linear utilizando somente este
#atributo mais relevante, para predição do atributo alvo determinado na questão  1 também.
print(database["class"].value_counts(), '\n')

y = database['price'].values
X = database['duration'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
reg.score(X_test, y_test)

# Mostre o gráfico da reta de regressão em conjunto com a nuvem de atributo.
plt.scatter(X_test, y_test, color='#440958', edgecolor = "#440958", label='Nuvem de atributos')
plt.plot(X_test, y_pred, color= '#43C59E', linewidth=2, label='Reta de regressão')
plt.xlabel('Variável Duration')
plt.ylabel('Variável Price')
plt.legend()
plt.show()

# Determine também os valores: RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante.
rss = np.sum((y_pred - y_test) ** 2)
print("Soma dos quadrados dos erros:",rss)
mse = mean_squared_error(y_test, y_pred)
print("Erro Quadrático Médio:", mse)
rmse = np.sqrt(mse)
print("Raiz do Erro Quadrático médio:", rmse)
r_squared = r2_score(y_test, y_pred)
print("Coeficiente de Determinação:", r_squared)
# Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados
