from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

sales_df = load_sales_clean_dataset()

# Cria matrizes X e y.
X = sales_df.drop(["sales", "social_media", "influencer"], axis=1)
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instanciar o modelo.
reg = LinearRegression()

# Ajustar o modelo aos dados.
reg.fit(X_train, y_train)

# Faz previsões.
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Calcula o R-squared.
r_squared = reg.score(X_test, y_test)

# Calcula o RMSE.
mse = mean_squared_error(y_test, y_pred)

# Calcula o RMSE extraindo a raiz quadrada do MSE.
rmse = np.sqrt(mse)

# Imprime as métricas.
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))
