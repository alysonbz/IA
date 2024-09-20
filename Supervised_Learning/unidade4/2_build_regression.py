from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Importando a regressão linear.
from sklearn.linear_model import LinearRegression

y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Cria o modelo.
reg = LinearRegression()

# Ajustar o modelo aos dados.
reg.fit(X, y)

# Faz as previsões.
predictions = reg.predict(X)

print(predictions)
