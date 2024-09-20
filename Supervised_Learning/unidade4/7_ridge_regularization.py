from src.utils import load_sales_clean_dataset
from sklearn.model_selection import train_test_split

# Importa o Ridge
from sklearn.linear_model import Ridge

sales_df = load_sales_clean_dataset()

# Cria matrizes X e y.
X = sales_df.drop(["sales", "influencer"], axis=1)
y = sales_df["sales"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
    # Crie um modelo de regressão Ridge.
    ridge = Ridge(alpha=alpha)

    # Ajusta os dados.
    ridge.fit(X_train, y_train)

    # Obtém o R-squared.
    score = ridge.score(X_test, y_test)
    ridge_scores.append(score)
print(ridge_scores)
