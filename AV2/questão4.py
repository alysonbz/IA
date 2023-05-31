from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
db = pd.read_csv("db_final.csv")
# Create X and y arrays
X = db["Deep sleep percentage"].values.reshape(-1, 1)
y = db["Sleep efficiency"].values


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