import matplotlib.pyplot as plt
from src.utils import load_sales_clean_dataset

# Importa Lasso.
from sklearn.linear_model import Lasso

sales_df = load_sales_clean_dataset()

# Crie matrizes X e y.
X = sales_df.drop(["sales", "influencer"], axis=1)
y = sales_df["sales"].values
sales_columns = X.columns

# Instanciar um modelo de regressão Lasso.
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)

# Calcule e imprima os coeficientes.
lasso_coef = lasso.coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
