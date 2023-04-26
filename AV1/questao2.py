#Importe as bibliotecas necessárias.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 

#Carregue o dataset. Se houver o dataset atualizado,
#hda = pd.read_csv('dataset/heart_desease.csv')
from questao1 import hda
# carregue o atualizado.
#print(hda.to_string(),'\n')
print('DataFrame Atualizada: \n',hda.to_string())

#Sem normalizar o conjunto de dados divida o dataset
# em treino e teste.
print(hda["HeartDisease"].value_counts(), '\n')

X= hda.drop(['HeartDisease'], axis=1)
y=hda[["HeartDisease"]]

X_train, X_test, y_train, y_test = train_test_split\
    (X, y, stratify=y, random_state=42, test_size=0.3)
print(y_train['HeartDisease'].value_counts())

#Implemente o Knn exbindo sua acurácia nos dados de
# teste e mantenha sua parametrização default.
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
y_pred= knn.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(accuracy)
