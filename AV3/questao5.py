'''Você descobriu qual a melhor forma de pré-processar os dados. Assim, utilizando a metodologia que
proporcionou o melhor acerto do classficador faça agora uma comparação entre classicadores para que você
também possa descobrir qual classificador mais adequado. Utilize outra técnica de classificação com os mesmos
dados, gere os números que quantificam o desempenho e faça uma comparação entre estes. Conclua o relatório
com auxílio de um fluxogragrama mostrando qual a metodologia completa para classificação dos dados do seu dataset.'''


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

# Colunas com maior variância
col_var = financial_distress[['Company', 'Time', 'Financial Distress', 'x1']]


# PCA
print('\nClassificador: PCA')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(col_var)

# Dividir os dados reduzidos em treinamento e teste
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, somente_fd, test_size=0.2, random_state=42)

# Criar classificadores KNN para PCA
knn_pca = KNeighborsClassifier(n_neighbors=3)

# Treinar os classificadores usando os dados de treinamento
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
print('\nClassificador: TSNE')
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(col_var)

# Dividir os dados reduzidos em treinamento e teste
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(X_tsne, somente_fd, test_size=0.2, random_state=42)

# Criar classificadores KNN para TSNE
knn_tsne = KNeighborsClassifier(n_neighbors=3)

# Treinar os classificadores usando os dados de treinamento
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



# Outro classificador: KNN
print('\nClassificador: KNN')

# Dividir os dados em treinamento e teste
X_knn_train, X_knn_test, y_train, y_test = train_test_split(col_var, somente_fd, test_size=0.2, random_state=42)

# Criar uma instância do classificador KNN com n_neighbors=3
knn = KNeighborsClassifier(n_neighbors=3)

# Treinar o classificador com os dados de treinamento
knn.fit(X_knn_train, y_train)

# Fazer previsões usando o conjunto de teste
y_pred_knn = knn.predict(X_knn_test)

# Calcular a acurácia para KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Calcular métricas de avaliação para KNN
print("\nMétricas de avaliação para KNN:")
print(classification_report(y_test, y_pred_knn))
print("\nMatriz de confusão para KNN:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nAcurácia KNN:", accuracy_knn)
