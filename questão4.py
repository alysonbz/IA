from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from src.utils import load_new_customer_dataset

# Carregue o dataset definido para você.
healthcare_data = pd.read_csv('dataset/healthcare-dataset-stroke-data-new.csv')
print(healthcare_data.head())

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir e otimizar o modelo KNN
param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Obter os melhores parâmetros e acurácia
best_k = grid_search.best_params_['n_neighbors']
best_accuracy = grid_search.best_score_

print(f'Melhor valor de k: {best_k}')
print(f'Acurácia com o melhor k: {best_accuracy:.4f}')

# Visualizar os resultados
results = grid_search.cv_results_
plt.figure(figsize=(10, 6))
plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia Média')
plt.title('Acurácia Média versus Número de Vizinhos')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Melhor k = {best_k}')
plt.legend()
plt.grid(True)
plt.show()