import numpy as np

from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
# Import the necessary modules
from sklearn.model_selection import cross_val_score, KFold

sales_df = load_sales_clean_dataset()
# Create X and y arrays
X = sales_df["radio"].values.reshape(-1, 1)
y = sales_df["sales"].values


#Create a KFold object
kf = KFold(n_splits=5, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print cv_scores
print("\nScore: ", cv_scores)

# Print the mean
print("\nMédia: ", np.mean(cv_scores))

# Print the standard deviation
print("\nDesvio padrão: ", np.std(cv_scores))