'''
Utilizando os dados da questão 2, aplique algum método de classificação e gere números que quantificam
o desempenho deste. Compare os números classificando o dataset reduzido pelo PCA e pelo T-SNE.
'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Carregando o conjunto de dados
financial_distress = pd.read_csv("Financial Distress Atualizado.csv")

# Separando X e y
sem_fd = financial_distress.drop(['Financial Distress'], axis=1)    # X
somente_fd = financial_distress['Financial Distress'].values        # y

# Reduzindo a dimensionalidade usando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(sem_fd)

# Dividindo os dados reduzidos em treino e teste
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, somente_fd, test_size=0.2, random_state=42)

# Criando classificadores KNN para PCA
knn_pca = KNeighborsClassifier(n_neighbors=5)

# Treinando os classificadores usando os dados de treino
knn_pca.fit(X_pca_train, y_train)

# Fazendo previsões nos dados de teste
y_pred_pca = knn_pca.predict(X_pca_test)

# Calculando a acurácia para PCA
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# Métricas PCA
print("\nMétricas de avaliação para o método de PCA:")
print(classification_report(y_test, y_pred_pca))
print("\nMatriz de confusão para o método de PCA:")
print(confusion_matrix(y_test, y_pred_pca))
print("\nAcurácia PCA:", accuracy_pca)


# Reduzindo a dimensionalidade usando TSNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(sem_fd)

# Dividindo os dados reduzidos em treino e teste
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(X_tsne, somente_fd, test_size=0.2, random_state=42)

# Criando classificadores KNN para TSNE
knn_tsne = KNeighborsClassifier(n_neighbors=5)

# Treinando os classificadores usando os dados de treino
knn_tsne.fit(X_tsne_train, y_train)

# Fazendo previsões nos dados de teste
y_pred_tsne = knn_tsne.predict(X_tsne_test)

# Calculando a acurácia para TSNE
accuracy_tsne = accuracy_score(y_test, y_pred_tsne)

# Métricas TSNE
print("\nMétricas de avaliação para o método de TSNE:")
print(classification_report(y_test, y_pred_tsne))
print("\nMatriz de confusão para o método de TSNE:")
print(confusion_matrix(y_test, y_pred_tsne))
print("\nAcurácia TSNE:", accuracy_tsne)

