from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Import LinearRegression
from Sklearn.linear_model import LineaRegression


y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = LineaRegression()

# Fit the model to the data
reg=LineaRegression()

# Make predictions
predictions = ____

print(__)