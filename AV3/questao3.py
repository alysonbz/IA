from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Definir semente para geração de números aleatórios
random_seed = 42

# Carregar o Dataset
cogu_df = pd.read_csv('mushrooms.csv')

# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Percorrer as colunas do dataset
for coluna in cogu_df.columns:
    # Verificar se a coluna contém valores de string
    if cogu_df[coluna].dtype == 'object':
        # Aplicar o LabelEncoder na coluna
        cogu_df[coluna] = label_encoder.fit_transform(cogu_df[coluna])

cogu = cogu_df.drop(['class'], axis=1)
clas = cogu_df['class'].values

# Dividir os dados em treinamento e teste
cogu_train, cogu_test, y_train, y_test = train_test_split(cogu, clas, test_size=0.2, random_state=random_seed)

# Criar um PCA
pca = PCA(n_components=2, random_state=random_seed)
pca_train = pca.fit_transform(cogu_train)
pca_test = pca.transform(cogu_test)

# Reduzir a dimensionalidade usando t-SNE
model = TSNE(n_components=2, random_state=random_seed)
tsne_train = model.fit_transform(cogu_train)

# Transformar os dados de teste usando as mesmas transformações do t-SNE nos dados de treinamento
tsne_test = model.fit_transform(cogu_test)

# Criar classificadores k-NN para PCA e t-SNE
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_tsne = KNeighborsClassifier(n_neighbors=3)

# Treinar os classificadores usando os dados de treinamento
knn_pca.fit(pca_train, y_train)
knn_tsne.fit(tsne_train, y_train)

# Fazer previsões sobre os dados de teste
y_pred_pca = knn_pca.predict(pca_test)
y_pred_tsne = knn_tsne.predict(tsne_test)

# Calcular métricas para o PCA
print("classification_report para PCA:")
print(classification_report(y_test, y_pred_pca))
print("confusion_Matrix para PCA:")
print(confusion_matrix(y_test, y_pred_pca))

# Calcular métricas para t-SNE
print("classification_report para t-SNE:")
print(classification_report(y_test, y_pred_tsne))
print("confusion_Matrix para t-SNE:")
print(confusion_matrix(y_test, y_pred_tsne))
