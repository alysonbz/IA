
# Importe as bibliotecas necessárias.
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
bt_atualizado = pd.read_csv('bt_novo.csv')
print("\n Dataset: Bt Atualizado")
print(bt_atualizado)

#Sem normalizar o conjunto de dados divida o dataset em treino e teste.
X = bt_atualizado.drop(['classe'], axis=1)
y = bt_atualizado["classe"].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, stratify=y, random_state=5)

#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Acurácia sem normalizar:')
print(knn.score(X_test, y_test))