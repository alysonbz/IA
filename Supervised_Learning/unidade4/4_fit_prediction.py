from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Carregar o dataset
sales_df = load_sales_clean_dataset()

# Excluir colunas 'sales' e 'influencer' e armazenar o resultado em X
X = sales_df.drop(["sales", "influencer"], axis=1)

# Armazenar os valores da coluna 'sales' em y
y = sales_df["sales"]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instanciar o modelo de regressão
reg = LinearRegression()

# Ajustar o modelo aos dados de treino
reg.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Calcular o coeficiente de determinação R^2
r_squared = reg.score(X_test, y_test)

# Calcular o valor da raiz do erro quadrático médio (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Imprimir as métricas
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))
