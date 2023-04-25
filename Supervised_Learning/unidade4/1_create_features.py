import numpy as np
from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()
print(sales_df)

# Create X from the radio column's values
X = sales_df["radio"].values

# Create y from the sales column's values
y = sales_df["sales"].values

# Reshape X
X_sales = X.reshape(-1, 1)

# Check the shape of the features and targets
print(X_sales.shape, y.shape)