# Implementação KNN

# Importando o dataset e os módulos: pandas e numpy
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Retornando os dados
iris = load_iris()
iris

# Transformando em um DataFrame
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df['target'] = iris.target

# Visualizando a base de dados iris
iris_df

# Selecionando apenas as colunas de pétala
iris1 = iris_df.loc[iris_df.target.isin([1,2]),['petal length (cm)','petal width (cm)','target']]
iris1

# Separando em X e y
X = iris1[['petal length (cm)','petal width (cm)']]
y = iris1.target

# Importando a função que separa os dados em treino e teste
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

# Importando o módulo que visualiza os gráficos
import matplotlib.pyplot as plt

# Visualizando os dados de treino
fig, ax = plt.subplots()
ax.scatter(x=X_train['petal width (cm)'],
           y=X_train['petal length (cm)'],
           c=y_train,
           cmap='viridis')
ax.set(xlim=(0.9, 2.6), xticks=[1,1.5,2,2.5],
       ylim=(3, 7), yticks=[3,4,5,6,7])
plt.show()

# Importando o método utilizado: KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Estabelecendo quantos vizinhos mais próximos serão utilizados
clf = KNeighborsClassifier(n_neighbors=3)

# Fazendo o fit com os dados de treino
clf = clf.fit(X_train,y_train)

# Fazendo a previsão para os dados de teste
y_pred = clf.predict(X_test)

# Importando o módulo que verifica a matriz de confusão
from sklearn.metrics import confusion_matrix

# Verificando a matriz de confusão
confusion_matrix(y_test,y_pred)

# Podemos agora visualizar os dados de treino e teste
fig, ax = plt.subplots()
ax.scatter(x=X_train['petal width (cm)'],
           y=X_train['petal length (cm)'],
           c=y_train, alpha=0.7,
           cmap='viridis')
ax.scatter(x=X_test['petal width (cm)'],
           y=X_test['petal length (cm)'],
           c=y_pred,alpha=0.2,
           cmap='RdYlGn')
ax.scatter(x=X_test['petal width (cm)'],
           y=X_test['petal length (cm)'],
           c=y_test,alpha=0.2,
           cmap='RdYlGn')
ax.set(xlim=(0.9, 2.6), xticks=[1,1.5,2,2.5],
       ylim=(3, 7), yticks=[3,4,5,6,7])
plt.show()

X_test[y_test != y_pred]


"""acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))"""

print("\nCALCULANDO AS DISTÂNCIAS")

# Distância Euclidiana
a = np.array(iris1['petal length (cm)'])
b = np.array(iris1['petal width (cm)'])
print("\nDistância Euclidiana: ")
dist_euc = np.sqrt(np.sum(np.square(a-b)))
print(dist_euc)

# Distância Manhattan
# Criando uma função para calcular
def manhattan_distance(point1, point2):
    return sum(abs(value1 - value2) for value1, value2 in zip(point1, point2))
# Definindo dois pontos
a = np.array(iris1['petal length (cm)'])
b = np.array(iris1['petal width (cm)'])
# Mostrando o resultado
print("\nDistância Manhattan: ")
print(manhattan_distance(a, b))

# Distância de Minkowski
# Definindo dois pontos
a = np.array(iris1['petal length (cm)'])
b = np.array(iris1['petal width (cm)'])
# Calculando a distância Minkowski com ordem p=3
p = 3
dist_minkowski = np.linalg.norm(a - b, ord=p)
# Mostrando o resultado
print("\nDistância de Minkowski: ")
print(dist_minkowski)

# Distância de Chebyshev
# Definindo dois pontos
a = np.array(iris1['petal length (cm)'])
b = np.array(iris1['petal width (cm)'])
# Mostrando na tela a frase p/ melhor entendimento e organização
print("\nDistância de Chebyshev: ")
# Calculando a distância Chebyshev entre os dois pontos
dist_chebyshev = np.amax(np.abs(a - b))
print(dist_chebyshev)

'''
Ao comparar os resultados das quatro distâncias calculadas para o conjunto de dados iris1, é possível observar 
que as distâncias variam entre si, pois cada uma delas utiliza uma fórmula diferente para calcular a distância 
entre dois pontos 'a' e 'b'.

A distância Euclidiana é a mais comum e simples, calculando a distância "em linha reta" entre dois pontos. 

A distância Manhattan, também conhecida como distância da cidade, considera apenas as distâncias horizontais 
e verticais, ignorando a distância diagonal. 

A distância de Minkowski é uma generalização da distância Euclidiana e da distância Manhattan, pois permite usar 
diferentes ordens p (como p = 3 no meu exemplo) para ajustar a fórmula à situação. 

A distância de Chebyshev, calcula a maior diferença entre as coordenadas dos dois pontos, sendo mais útil quando 
se quer avaliar a discrepância máxima entre dois conjuntos de dados.
'''