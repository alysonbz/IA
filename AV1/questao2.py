#Importe as bibliotecas necessárias.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
db = pd.read_csv("db_ajustado.csv")

#Sem normalizar o conjunto de dados divida o dataset em treino e teste.
X = db[['Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y = db['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.2, random_state=11)

#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier()

knn.fit(X_train,y_train)
print("Acurácia: ",knn.score(X_test, y_test))