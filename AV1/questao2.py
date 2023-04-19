#Importe as bibliotecas necessárias.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from questao1 import load_binary_dataset

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df = load_binary_dataset()

#Sem normalizar o conjunto de dados divida o dataset em treino e teste.
X = df.drop(['target'], axis=1) #Cria um DataFrame com todas as colunas, com exceção de "target"
y = df['target'] # Cria um dataframe de labels com a coluna "target"

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("score:\n", knn.score(X_test, y_test))

