import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
db = pd.read_csv("db_final.csv")
X = db.drop(['ID', 'Sleep efficiency'],axis=1)
y = db['Sleep efficiency'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#inicialize Lasso
lasso = Lasso()

#inicialize kfold
kf = KFold(n_splits=6, shuffle=True, random_state=42)

#Set up the parameter grid
param_grid = {"alpha": np.arange(0.00001, 1, 20)}

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv = kf)

# Fit to the training data
lasso_cv.fit(X_train, y_train)

print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))

print("\nRIDGE\n")

param_grid2 = {"alpha": np.arange(0.00001, 1, 20),
              "solver":["sag","lsqr"]}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid2, cv=kf)
ridge_cv.fit(X_train,y_train)
print("Tuned Ridge paramaters: {}".format(ridge_cv.best_params_))
print("Tuned Ridge score: {}".format(ridge_cv.best_score_))

