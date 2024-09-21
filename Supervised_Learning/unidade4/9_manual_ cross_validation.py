import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression


class KFold:

   def __init__(self,n_splits):

       self.n_splits = n_splits

   def _compute_score(self,y_true,y_pred):
       ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
       ss_residual = np.sum((y_true - y_pred) ** 2)
       r2 = 1 - (ss_residual / ss_total)
       return r2


   def cross_val_score(self,obj,X, y):
        scores = []
        for fold in range(self.n_splits):
            # parte 1: dividir o dataset X em n_splits vezes
            n_samples = len(X)
            fold_size = n_samples // self.n_splits
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            start = fold * fold_size
            end = (fold + 1) * fold_size if fold != self.n_splits - 1 else n_samples

            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            obj.fit(X_train, y_train)

            y_pred = obj.predict(X_test)

            # parte 2: Calcular a métrica score para subset dividida na parte 1. Chamar a função _compute_score para cada subset

            score = self._compute_score(y_test, y_pred)

            # appendar na lista scores cada valor obtido na parte 2

            scores.append(score)
        # parte 3 - retornar a lista de scores
        return scores


sales_df = load_sales_clean_dataset()

X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

kf = KFold(n_splits=6)

reg = LinearRegression()

cv_scores = kf.cross_val_score(reg,X, y)

print(cv_scores)

print(np.mean(cv_scores))

print(np.std(cv_scores))