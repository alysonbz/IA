from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Crie X a partir dos valores da coluna de rádio.
X = sales_df['radio'].values

# Crie y a partir dos valores da coluna de vendas.
y = sales_df['sales'].values

# Remodelar.
X = X.reshape(-1, 1)

# Verifique o formato dos recursos e alvos.
print(X.shape, y.shape)
