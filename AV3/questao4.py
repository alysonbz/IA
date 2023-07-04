import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Lendo o dataset
estado_do_olho0 = pd.read_csv(r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\AV3\archive (2)\EEG_Eye_State_Classification.csv")
estado_do_olho = estado_do_olho0.sample(n=1498, replace=False)  # Com 10% do dataset

# Separando os atributos das classes
X = estado_do_olho.drop('eyeDetection', axis=1)
y = estado_do_olho['eyeDetection']

# Redução de dimensionalidade usando PCA e análise de variância
pca = PCA(n_components=0.95, random_state=0)  # Manter 95% da variância
X_pca = pca.fit_transform(X)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)

# Treinamento do classificador
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

# Previsão dos rótulos
y_pred = classifier.predict(X_test)

# Avaliação do desempenho usando métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

# Impressão dos resultados
print("Resultados usando PCA com análise de variância:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
