#importe as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
cancer_relevancia = pd.read_csv('dados_preprocessados.csv')


#Sem normalizar o conjunto de dados divida o dataset em treino e teste.

X = cancer_relevancia.drop('diagnosis', axis=1) #pega todas as colunas exceto diagnosis e atribui a X
y = cancer_relevancia['diagnosis']  #pega somente a coluna diagnosis e atribui a y

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #serve para dividir o conjunto em treino e teste


#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)  #treina o modelo com os dados de treino
y_pred = knn.predict(X_test)   #realiza as predições com os dados de teste

print('acurácia:', knn.score(X_test, y_test))  #calcula a acuracia
