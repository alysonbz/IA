import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Carregar o conjunto de dados
cogu_df = pd.read_csv('mushrooms.csv')

# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Percorrer as colunas do dataset
for coluna in cogu_df.columns:
    # Verificar se a coluna contém valores de string
    if cogu_df[coluna].dtype == 'object':
        # Aplicar o LabelEncoder na coluna
        cogu_df[coluna] = label_encoder.fit_transform(cogu_df[coluna])


cogu = cogu_df.drop(['class'],axis=1)
clas = cogu_df['class'].values

#PCA_VARIANCE
scaler = StandardScaler()
pca = PCA()
# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

pipeline.fit(cogu)

# Plot the explained variances
features = range(pca.n_features_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

#COLUNAS COM MAIOR VARIANCIA
colunas_v= cogu[['odor','gill-color','stalk-shape','spore-print-color']]

#DESEMPENHO
pca = PCA(n_components=2)
X_pca = pca.fit_transform(colunas_v)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(colunas_v)

# Dividir os dados reduzidos em treinamento e teste
pca_train, pca_test, y_train1, y_test1 = train_test_split(X_pca, clas, test_size=0.2, random_state=42)
tsne_train, tsne_test, y_train, y_test = train_test_split(X_tsne, clas, test_size=0.2, random_state=42)

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