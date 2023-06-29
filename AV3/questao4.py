import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregar o conjunto de dados
trojan_df = pd.read_csv('Trojan_Detection.csv')

# Transformando df_reduzido
map_df_reduzido = {
    'Trojan': 0,
    'Benign': 1
}
trojan_df['Class'] = trojan_df['Class'].map(map_df_reduzido)
trojan = trojan_df.drop(['Flow ID', ' Source IP', ' Destination IP', ' Timestamp'], axis=1)


df = trojan.drop(['Class'],axis=1)
trojan_valor= trojan['Class'].values

#PCA_VARIANCE
scaler = StandardScaler()
pca = PCA()
# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

pipeline.fit(df)

# Plot the explained variances
features = range(pca.n_features_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

#COLUNAS COM MAIOR VARIANCIA
colunas_v= trojan[[' Source Port',' Destination Port',' Protocol',' Flow Duration',' Total Fwd Packets', ' Total Backward Packets']]

#DESEMPENHO
pca = PCA(n_components=2)
X_pca = pca.fit_transform(colunas_v)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(colunas_v)

# Dividir os dados reduzidos em treinamento e teste
X_pca_train, X_pca_test, y_train1, y_test1 = train_test_split(X_pca, trojan_valor, test_size=0.2, random_state=42)
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(X_tsne, trojan_valor, test_size=0.2, random_state=42)

# Criar classificadores k-NN para PCA e t-SNE
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_tsne = KNeighborsClassifier(n_neighbors=3)

# Treinar os classificadores usando os dados de treinamento
knn_pca.fit(X_pca_train, y_train1)
knn_tsne.fit(X_tsne_train, y_train)
# Fazer previsões sobre os dados de teste
y_pred_pca = knn_pca.predict(X_pca_test)
y_pred_tsne = knn_tsne.predict(X_tsne_test)
# Calcular métricas de avaliação para PCA
print("Métricas de avaliação para PCA:")
print(classification_report(y_test1, y_pred_pca))
print("Matriz de confusão para PCA:")
print(confusion_matrix(y_test1, y_pred_pca))
print(" \nAcuracia PCA: 0.")

print('\n_____________________________')

# Calcular métricas de avaliação para t-SNE
print(" Métricas de avaliação para t-SNE:")
print(classification_report(y_test, y_pred_tsne))
print("Matriz de confusão para t-SNE:")
print(confusion_matrix(y_test, y_pred_tsne))
print(" \nAcuracia T-SNE: 0.")