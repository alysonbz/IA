#Importe as bibliotecas necessárias.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
cancer = pd.read_csv('dados_preprocessados.csv')
print(cancer)

#Sem normalizar o conjunto de dados divida o dataset em treino e teste.
# Separando os dados em features e target
X = cancer.drop('diagnosis', axis=1)
y = cancer['diagnosis']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_test, y_test)

print('score', knn.score(X_test, y_test))

X.to_csv('dados_preprocessados2.csv', index=False)
y.to_csv('dados_preprocessados3.csv', index=False)