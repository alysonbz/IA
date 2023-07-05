import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# INICIALIZANDO
label_encoder = LabelEncoder()

# PROCESSANDO
train_df = pd.read_csv('train_dataset.csv')

train = train_df.drop(['Class'], axis=1)
Class = label_encoder.fit_transform(train_df['Class'])
colunas_V = train[['Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation']]

# PCA e TSNE
pca = PCA(n_components=2)
Column_pca = pca.fit_transform(colunas_V)
tsne = TSNE(n_components=2)
Column_tsne = tsne.fit_transform(colunas_V)

# Dividir os dados reduzidos em treinamento e teste
X_pca_train, X_pca_test, y_train1, y_test1 = train_test_split(Column_pca, Class, test_size=0.2, random_state=42)
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(Column_tsne, Class, test_size=0.2, random_state=42)

# K-NN para PCA e TSNE
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_tsne = KNeighborsClassifier(n_neighbors=3)

# SVM
svm = SVC()

# Treinar os modelos
knn_pca.fit(X_pca_train, y_train1)
knn_tsne.fit(X_tsne_train, y_train)
svm.fit(colunas_V, Class)

# Fazer previsões sobre os dados de teste
y_pred_pca = knn_pca.predict(X_pca_test)
y_pred_tsne = knn_tsne.predict(X_tsne_test)
y_pred_svm = svm.predict(colunas_V)

# PCA
print("Classification Report para PCA (k-NN):")
print(classification_report(y_test1, y_pred_pca))
print("Confusion Matrix para PCA (k-NN):")
print(confusion_matrix(y_test1, y_pred_pca))

# TSNE
print("Classification Report para t-SNE (k-NN):")
print(classification_report(y_test, y_pred_tsne))
print("Confusion Matrix para t-SNE (k-NN):")
print(confusion_matrix(y_test, y_pred_tsne))

# SVM
print("Classification Report para SVM:")
print(classification_report(Class, y_pred_svm))
print("Confusion Matrix para SVM:")
print(confusion_matrix(Class, y_pred_svm))
