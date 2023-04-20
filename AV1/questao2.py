import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
Drug_and_Features = pd.read_csv('Drugs_and_Features')
print(Drug_and_Features)

#Sem normalizar o conjunto de dados divida o dataset em treino e teste.

X = Drug_and_Features.drop(['Drug'], axis=1) # cria um dataframe com todas as colunas, com exceção de "target"
y = Drug_and_Features['Drug'].values # cria um dataframe de labels com a coluna "target"

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42) # divide o dataset em treino e teste com amostragem estratificada

#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier() # inicializa o algoritmo KNN
knn.fit(X_train, y_train) # aplica a função fit do KNN
print("score:\n", knn.score(X_test, y_test)) # verifica o acerto do classificador

