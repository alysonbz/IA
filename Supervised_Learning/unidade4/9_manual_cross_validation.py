import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression


class KFold:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _compute_score(self, obj, X_train, y_train, X_test, y_test):
        # Treina o modelo nos dados de treino.
        obj.fit(X_train, y_train)
        # Calcula e retorna a métrica R² no conjunto de teste.
        return obj.score(X_test, y_test)

    def cross_val_score(self, obj, X, y):
        scores = []

        n_samples = len(X)
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Define os índices para o conjunto de teste.
            test_indices = range(i * fold_size, (i + 1) * fold_size)
            # Define os índices para o conjunto de treino (complementar ao de teste).
            train_indices = np.setdiff1d(range(n_samples), test_indices)

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Computa o score para esta divisão.
            score = self._compute_score(obj, X_train, y_train, X_test, y_test)
            scores.append(score)

        return scores


sales_df = load_sales_clean_dataset()

# Cria matrizes X e y.
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Cria um objeto KFold.
kf = KFold(n_splits=6)

reg = LinearRegression()

# Calcula pontuações de validação cruzada de 6 vezes.
cv_scores = kf.cross_val_score(reg, X, y)

# Printa os scores.
print(cv_scores)

# Printa a média.
print(np.mean(cv_scores))

# Printa o desvio padrão.
print(np.std(cv_scores))
