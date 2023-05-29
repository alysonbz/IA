from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

emissao_new = pd.read_csv("emissao_new.csv")
# Create X and y arrays
X = emissao_new["year"].values.reshape(-1, 1)
y = emissao_new["emissions_tons"].values

#Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print cv_scores
print(cv_scores)

# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print(np.std(cv_scores))
