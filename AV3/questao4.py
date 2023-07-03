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
oleo_df = pd.read_csv(r"C:\Users\UFC\Downloads\savio\IA\AV3\oil_spill.csv")
test = oleo_df.drop(['target'],axis=1)
Area = oleo_df['target'].values

#PCA_VARIANCE
scaler = StandardScaler()
pca = PCA()
# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

pipeline.fit(test)

# Plot the explained variances
features = range(pca.n_features_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

#COLUNAS COM MAIOR VARIANCIA
colunas_v= test[['f_1', 'f_2', 'f_3', 'f_4']]

#DESEMPENHO
pca = PCA(n_components=2)
X_pca = pca.fit_transform(colunas_v)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(colunas_v)

# Dividir os dados reduzidos em treinamento e teste
pca_train, pca_test, y_train1, y_test1 = train_test_split(X_pca, Area, test_size=0.2, random_state=42)
tsne_train, tsne_test, y_train, y_test = train_test_split(X_tsne, Area, test_size=0.2, random_state=42)

# Criar classificadores k-NN para PCA e t-SNE
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_tsne = KNeighborsClassifier(n_neighbors=3)

# Treinar os classificadores usando os dados de treinamento
knn_pca.fit(pca_train, y_train1)
knn_tsne.fit(tsne_train, y_train)

# Fazer previsões sobre os dados de teste
y_pred_pca = knn_pca.predict(pca_test)
y_pred_tsne = knn_tsne.predict(tsne_test)


# Calcular métricas de avaliação para PCA
print("classification_report para PCA:")
print(classification_report(y_test1, y_pred_pca))
print("confusion Matrix para PCA:")
print(confusion_matrix(y_test1, y_pred_pca))


# Calcular métricas de avaliação para t-SNE
print("classification_report para t-SNE:")
print(classification_report(y_test, y_pred_tsne))
print("confusion Matrix para t-SNE:")
print(confusion_matrix(y_test, y_pred_tsne))