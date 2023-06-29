import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Carregue o conjunto de dados em um DataFrame do Pandas
df = pd.read_csv('oil_spill.csv')

# Separar os atributos (X) e o target (y)
X = df.drop('target', axis=1)
y = df['target']

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar uma instância do PCA
pca = PCA()

# Ajustar o PCA aos dados de treino
pca.fit(X_train)

# Calcular as variâncias das componentes principais
variances = pca.explained_variance_ratio_

# Ordenar as variâncias em ordem decrescente
sorted_variances = sorted(variances, reverse=True)

# Definir o número de componentes principais a serem selecionadas (por exemplo, as top 10)
num_components = 10

# Selecionar as top N componentes principais com as maiores variâncias
top_components = sorted_variances[:num_components]

# Obter os índices das top N componentes principais
top_component_indices = [variances.tolist().index(component) for component in top_components]

# Reduzir a dimensionalidade dos conjuntos de treino e teste para as top N componentes principais selecionadas
X_train_pca = pca.transform(X_train)[:, top_component_indices]
X_test_pca = pca.transform(X_test)[:, top_component_indices]

# Criar e treinar um classificador KNN usando os dados reduzidos pelo PCA
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

# Fazer previsões no conjunto de teste reduzido pelo PCA
y_pred_pca = knn_pca.predict(X_test_pca)

# Calcular a acurácia do classificador com os dados reduzidos pelo PCA
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("Acurácia usando PCA:", accuracy_pca)
