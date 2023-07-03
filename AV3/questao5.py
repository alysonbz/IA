#Bibliotecas
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Carregar o Dataset
test_df = pd.read_csv(r"C:\Users\UFC\Downloads\savio\IA\AV3\oil_spill.csv")

#DIvidindo
test = test_df.drop(['target'],axis=1)
Area = test_df['target'].values
colunas_R = test[['f_1', 'f_2', 'f_3', 'f_4']]

#PCA e TSNE
pca = PCA(n_components=2)
Column_pca = pca.fit_transform(colunas_R)
tsne = TSNE(n_components=2)
Column_tsne = tsne.fit_transform(colunas_R)


# Dividir os dados reduzidos em treinamento e teste
X_pca_train, X_pca_test, y_train1, y_test1 = train_test_split(Column_pca, Area, test_size=0.2, random_state=42)
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(Column_tsne, Area, test_size=0.2, random_state=42)


#k-NN para PCA e tSNE
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_tsne = KNeighborsClassifier(n_neighbors=3)


# Treinar os modelos
knn_pca.fit(X_pca_train, y_train1)
knn_tsne.fit(X_tsne_train, y_train)


# Fazer previsões sobre os dados de teste
y_pred_pca = knn_pca.predict(X_pca_test)
y_pred_tsne = knn_tsne.predict(X_tsne_test)


#PCA
print("classification_report para PCA:")
print(classification_report(y_test1, y_pred_pca))
print("confusion_matrix para PCA:")
print(confusion_matrix(y_test1, y_pred_pca))

#TSNE
print("classification_report para t-SNE:")
print(classification_report(y_test, y_pred_tsne))
print("confusion_matrix para t-SNE:")
print(confusion_matrix(y_test, y_pred_tsne))

#(Regressão Logistica)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#usando o Train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(colunas_R, Area, test_size=0.2, random_state=42)

# Criar uma instância do classificador de Regressão Logística
logreg = LogisticRegression()

# Treinar o classificador com os dados de treinamento
logreg.fit(X_train2, y_train2)

# Fazer previsões usando o conjunto de teste
y_pred2 = logreg.predict(X_test2)

# Criar a matriz de confusão
confusion_matrix = confusion_matrix(y_test2, y_pred2)
print('confusion_matrix:')
print(confusion_matrix)

# Gerar o relatório de classificação
report = classification_report(y_test, y_pred2)
print('\nClassification Report:')
print(report)
# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred2)
print("\nAcurácia da Regressão Logística:", accuracy)