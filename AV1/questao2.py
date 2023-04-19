#realizar uma classificação utilizando KNN.

#Importe as bibliotecas necessárias.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
diabetes = pd.read_csv(r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\AV1\dataset\diabetes_ajustado.csv")
print(diabetes.shape)
#Sem normalizar o conjunto de dados divida o dataset em treino e teste.
X = diabetes.drop(["Diabetes"], axis=1)
#y = diabetes["Diabetes"].value_counts()
y = diabetes[["Diabetes"]]
X_train, y_train, X_test, y_test = train_test_split(X , y, stratify= y, random_state=42)

#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print("acerto", knn.score(X_teste, y_teste))