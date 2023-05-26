import matplotlib.pyplot as plt
import pandas as pd

# Import Lasso
from sklearn.linear_model import Lasso

emission = pd.read_csv('CO2 Emissions_Canada.csv')

# Create X and y arrays
X = emission.drop(["CO2 Emissions(g/km)"], axis=1).select_dtypes(exclude=["object"])

y = emission["CO2 Emissions(g/km)"].values
sales_columns = X.columns

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Compute and print the coefficients
lasso_coef = lasso.fit(X,y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)

plt.xticks(rotation=45)
plt.show()

