
import pandas as pd
from sklearn.preprocessing import  Normalizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#lendo o dataset
estado_do_olho0 = pd.read_csv(r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\AV3\archive (2)\EEG_Eye_State_Classification.csv")
estado_do_olho = estado_do_olho0.sample(n= 1498, replace=False) #com 10% do dataset

# Separando os atributos das classes
X = estado_do_olho.drop('eyeDetection', axis=1)
y = estado_do_olho['eyeDetection']

# Aplicando a normalização
scaler = Normalizer()
normal = scaler.fit_transform(X)

# Criando um novo DataFrame com os dados normalizados
estado_do_olho2 = pd.DataFrame(normal, columns=X.columns)


# Redução de dimensionalidade usando T-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_result = tsne.fit_transform(estado_do_olho2)

# Redução de dimensionalidade usando PCA
pca = PCA(n_components=2, random_state=0)
pca_result = pca.fit_transform(estado_do_olho2)

estado_do_olho2['eyeDetection'] = y

# Divisão dos dados em treino e teste
X_train_tsne, X_test_tsne, y_train, y_test = train_test_split(tsne_result, y, test_size=0.2, random_state=0)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(pca_result, y, test_size=0.2, random_state=0)

# Treinamento do classificador usando T-SNE
classifier_tsne = LogisticRegression()
classifier_tsne.fit(X_train_tsne, y_train)

# Previsão dos rótulos usando T-SNE
y_pred_tsne = classifier_tsne.predict(X_test_tsne)

# Avaliação do desempenho usando métricas
accuracy_tsne = accuracy_score(y_test, y_pred_tsne)
precision_tsne = precision_score(y_test, y_pred_tsne)
recall_tsne = recall_score(y_test, y_pred_tsne)
f1_score_tsne = f1_score(y_test, y_pred_tsne)

# Treinamento do classificador usando PCA
classifier_pca = LogisticRegression()
classifier_pca.fit(X_train_pca, y_train)

# Previsão dos rótulos usando PCA
y_pred_pca = classifier_pca.predict(X_test_pca)

# Avaliação do desempenho usando métricas
accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca)
recall_pca = recall_score(y_test, y_pred_pca)
f1_score_pca = f1_score(y_test, y_pred_pca)

# Impressão dos resultados
print("Resultados usando T-SNE:")
print("Accuracy:", accuracy_tsne)
print("Precision:", precision_tsne)
print("Recall:", recall_tsne)
print("F1-score:", f1_score_tsne)
print()
print("Resultados usando PCA:")
print("Accuracy:", accuracy_pca)
print("Precision", precision_pca)
print("Recall:", recall_pca)
print("F1-score:", f1_score_pca)
