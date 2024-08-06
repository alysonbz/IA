# Importe as bibliiotecas necessárias
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
drug200_new = pd.read_csv('dataset/drug200_new.csv')
print(drug200_new.head())

# Normalize com a melhor normalização o conjunto de dados se houver melhoria.
X = drug200_new.drop(columns=['Drug'])
y = drug200_new['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

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