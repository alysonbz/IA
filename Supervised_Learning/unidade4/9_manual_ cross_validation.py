import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class KFold:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _compute_score(self, obj, X_train, y_train, X_test, y_test):
        obj.fit(X_train, y_train)
        y_pred = obj.predict(X_test)
        score = mean_squared_error(y_test, y_pred)  # Usa o erro quadrático médio como métrica de score
        return score

    def cross_val_score(self, obj, X, y):
        scores = []
        fold_size = len(X) // self.n_splits

        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size

            X_test = X[start:end]
            y_test = y[start:end]

            X_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])

            score = self._compute_score(obj, X_train, y_train, X_test, y_test)
            scores.append(score)

        return scores


# Carrega o dataset
sales_df = load_sales_clean_dataset()

# Cria os arrays X e y
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Cria um objeto KFold
kf = KFold(n_splits=6)

# Cria o modelo de regressão linear
reg = LinearRegression()

# Calcula os scores de validação cruzada com 6 folds
cv_scores = kf.cross_val_score(reg, X, y)

# Imprime os scores
print(cv_scores)

# Imprime a média dos scores
print(np.mean(cv_scores))

# Imprime o desvio padrão dos scores
print(np.std(cv_scores))
