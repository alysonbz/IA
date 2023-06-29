import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Carregue o conjunto de dados em um DataFrame do Pandas
df = pd.read_csv('oil_spill.csv')

# Separar os atributos (X) e o target (y)
X = df.drop('target', axis=1)
y = df['target']

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar uma instância do PCA
pca = PCA(n_components=2)

# Reduzir a dimensionalidade dos conjuntos de treino e teste usando PCA
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Criar uma instância do t-SNE
tsne = TSNE(n_components=2, random_state=42)

# Reduzir a dimensionalidade dos conjuntos de treino e teste usando t-SNE
X_train_tsne = tsne.fit_transform(X_train)
X_test_tsne = tsne.fit_transform(X_test)

# Criar e treinar um classificador KNN usando os dados reduzidos pelo PCA
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

# Fazer previsões no conjunto de teste reduzido pelo PCA
y_pred_pca = knn_pca.predict(X_test_pca)

# Calcular a acurácia do classificador com os dados reduzidos pelo PCA
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("Acurácia usando PCA:", accuracy_pca)

# Criar e treinar um classificador KNN usando os dados reduzidos pelo t-SNE
knn_tsne = KNeighborsClassifier(n_neighbors=5)
knn_tsne.fit(X_train_tsne, y_train)

# Fazer previsões no conjunto de teste reduzido pelo t-SNE
y_pred_tsne = knn_tsne.predict(X_test_tsne)

# Calcular a acurácia do classificador com os dados reduzidos pelo t-SNE
accuracy_tsne = accuracy_score(y_test, y_pred_tsne)
print("Acurácia usando t-SNE:", accuracy_tsne)
