import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# INICIALIZANDO
label_encoder = LabelEncoder()

# PROCESSANDO
train_df = pd.read_csv('train_dataset.csv')

train = train_df.drop(['Class'], axis=1)
Class = label_encoder.fit_transform(train_df['Class'])

# Criar um PCA
pca = PCA(n_components=2)
scaled_train = pca.fit_transform(train)

# Reduzir a dimensionalidade usando t-SNE
model = TSNE(n_components=2)
normalized_train = model.fit_transform(train)

# Dividir os dados reduzidos em treinamento e teste
pca_train, pca_test, y_train1, y_test1 = train_test_split(scaled_train, Class, test_size=0.2, random_state=42)
tsne_train, tsne_test, y_train, y_test = train_test_split(normalized_train, Class, test_size=0.2, random_state=42)

# Criar classificadores k-NN para PCA e t-SNE
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_tsne = KNeighborsClassifier(n_neighbors=3)

# Treinar os classificadores usando os dados de treinamento
knn_pca.fit(pca_train, y_train1)
knn_tsne.fit(tsne_train, y_train)

# Fazer previsões sobre os dados de teste
y_pred_pca = knn_pca.predict(pca_test)
y_pred_tsne = knn_tsne.predict(tsne_test)

# Calcular métricas para o PCA
print("classification_report para PCA:")
print(classification_report(y_test1, y_pred_pca))
print("confusion_Matrix para PCA:")
print(confusion_matrix(y_test1, y_pred_pca))

# Calcular métricas para t-SNE
print("classification_report para t-SNE:")
print(classification_report(y_test, y_pred_tsne))
print("confusion_Matrix para t-SNE:")
print(confusion_matrix(y_test, y_pred_tsne))
