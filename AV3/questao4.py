'''Utilizando análise de variância do PCA. Reduza a dimensão para realizar uma classificação utilizando somente as
colunas de maior variância. Aplique o mesmo método de classificação testado na questão 3. Gere os mesmos números
que analisam o desempenho do classificador e verifique se houve melhoria no resultado.'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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


# Variância do PCA
scaler = StandardScaler()
pca = PCA()

# Criando pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Ajustando o pipeline X: 'sem_fd'
pipeline.fit(sem_fd)

# Plotando de gráfico para descobrir as colunas com maior variância
features = range(pca.n_features_)
plt.bar(features, pca.explained_variance_, color='purple')
plt.title("PCA: escolha das colunas com maior variância")
plt.xlabel('Colunas')
plt.ylabel('Variância')
plt.xticks(features)
plt.show()


# Colunas com maior variância
col_var = financial_distress[['Company', 'Time', 'Financial Distress', 'x1']]


# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(col_var)

# Dividir os dados reduzidos em treinamento e teste
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, somente_fd, test_size=0.2, random_state=42)

# Criar classificadores KNN para PCA
knn_pca = KNeighborsClassifier(n_neighbors=3)

# Treinar o classificador usando os dados de treino
knn_pca.fit(X_pca_train, y_train)

# Fazer previsões sobre os dados de teste
y_pred_pca = knn_pca.predict(X_pca_test)

# Calculando a acurácia para PCA
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# Calcular métricas de avaliação para PCA
print("\nMétricas de avaliação para PCA:")
print(classification_report(y_test, y_pred_pca))
print("\nMatriz de confusão para PCA:")
print(confusion_matrix(y_test, y_pred_pca))
print("\nAcurácia PCA:", accuracy_pca)


# TSNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(col_var)

# Dividir os dados reduzidos em treinamento e teste
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(X_tsne, somente_fd, test_size=0.2, random_state=42)

# Criar classificadores KNN para TSNE
knn_tsne = KNeighborsClassifier(n_neighbors=3)

# Treinar o classificador usando os dados de treino
knn_tsne.fit(X_tsne_train, y_train)

# Fazer previsões sobre os dados de teste
y_pred_tsne = knn_tsne.predict(X_tsne_test)

# Calculando a acurácia para TSNE
accuracy_tsne = accuracy_score(y_test, y_pred_tsne)

# Calcular métricas de avaliação para TSNE
print("\nMétricas de avaliação para TSNE:")
print(classification_report(y_test, y_pred_tsne))
print("\nMatriz de confusão para TSNE:")
print(confusion_matrix(y_test, y_pred_tsne))
print("\nAcurácia TSNE:", accuracy_tsne)

