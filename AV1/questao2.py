#Importe as bibliotecas necessárias.
from sklearn.model_selection import train_test_split

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
import df from questao1

#Sem normalizar o conjunto de dados divida o dataset em treino e teste.
X = #Crie um DataFrame com todas as colunas
y = # Crie um dataframe de labels

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier()
print(knn.score(X_test, y_test))

