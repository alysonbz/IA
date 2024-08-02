import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
file_path = 'dataset/flavors_of_cacao_ajustado.csv'
df = pd.read_csv(file_path)

# Separar o dataset em características e rótulo
X = df.drop("Rating", axis=1)
X = pd.get_dummies(X)
X = X.fillna(X.mean())
y = df['Rating']
y = y.astype('category').cat.codes

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização de Média Zero e Variância Unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir o modelo KNN
knn = KNeighborsClassifier()

# Definir o range de valores de k para testar
param_grid = {'n_neighbors': list(range(1, 21))}

# Definir o StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Realizar a busca em grade
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Melhor valor de k e acurácia correspondente
best_k = grid_search.best_params_['n_neighbors']
best_accuracy = grid_search.best_score_

print(f'Melhor valor de k: {best_k}')
print(f'Acurácia com o melhor k: {best_accuracy:.4f}')

# Obter resultados da busca em grade
results = grid_search.cv_results_

# Plotar gráfico de acurácia versus k
plt.figure(figsize=(10, 6))
plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia Média')
plt.title('Acurácia Média versus Número de Vizinhos')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Melhor k = {best_k}')
plt.legend()
plt.grid(True)
plt.show()
