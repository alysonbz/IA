from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
# Import the necessary modules
from sklearn.model_selection import cross_val_score, KFold

df = pd.read_csv("df.csv")
# Create X and y arrays
X = df["Weight"].values.reshape(-1, 1)
y = df["BodyFat"].values


#Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg =LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print cv_scores
print(cv_scores)
# Print the mean
print(np.mean(cv_scores))
# Print the standard deviation
print(np.std(cv_scores))