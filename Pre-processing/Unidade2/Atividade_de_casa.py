
# Importando as bibliotecas que serão usadas
from sklearn.datasets import load_iris # dataset utilizado
import pandas as pd # para ler base de dados
import numpy as np # para lidar com números
from sklearn.model_selection import train_test_split # Importando a função que separa os dados em treino e teste
import matplotlib.pyplot as plt # Importando o módulo que visualiza os gráficos
from sklearn.neighbors import KNeighborsClassifier # Importando o método utilizado: KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

#leitura do arquivo
#o comando with open abre o arquivo iris_data em modo leitura por conta do 'r'
lista=[]
with open('iris_data.csv', 'r') as f:
    for linha in f.readlines(): # o loop for lê cada linha usando o método readlines()
        a=linha.replace('\n','').split(',')
        lista.append(a) #adicionando cada sub lista em uma lista
data = pd.read_csv("iris_data.csv",
                   sep = ",",
                   names = ["sepal.length", "sepal.width","petal.length", "petal.width", "species"]
                   )
X = data.iloc[:, : -1].values
y = data.iloc[:, -1].values
#Transformando as strings
l_encode = LabelEncoder()
y = l_encode.fit_transform(y)

#função countclasses para contar o número de sublistas

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)): #percorrendo cada elemento da lista
        if lista[i][-1] == 0:
            setosa += 1
        if lista[i][-1] == 1:
            versicolor += 1
        if lista[i][-1] == 2:
            virginica += 1
    return [setosa, versicolor, virginica] #retorna uma lista contendo a contagem de cada classe

# Divisão do conjunto em teste e treino
p=0.6 #60% separado para o conjunto de treinamento e 40% para teste
setosa,versicolor, virginica = countclasses(lista) #contagem do número de ocorrências em cada classe
treinamento, teste= [], []

# cálculo do número máximo de instâncias de cada classe que devem ser usadas para treinamento.
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)

# As variáveis á seguir servirão para contar quantas instâncias de cada classe
# já foram adicionadas ao conjunto de treinamento.
total1 =0
total2 =0
total3 =0
for lis in lista:
    if lis[-1]== 0 and total1< max_setosa:
        treinamento.append(lis)
        total1 +=1
    elif lis[-1]== 1 and total2<max_versicolor:
        treinamento.append(lis)
        total2 +=1
    elif lis[-1]== 2 and total3<max_virginica:
        treinamento.append(lis)
        total3 +=1
    else:
        teste.append(lis) #Se a linha não for adicionada ao conjunto de treinamento, ela será adicionada à lista "teste".

# Função que calcula distância entre vetores
import math
def dist_euclidiana(v1,v2): #aplicação do teorema de pitágoras
    dim, soma = len(v1), 0 # dim determina o número de elementos de cada vetor
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)

#Aplicação do Knn
def knn(treinamento, nova_amostra, K): #aqui são passados os parâmetros
    # o K delimita a quant de vizinhos mais próximos
    # treinamento, que é uma lista contendo as amostras de treinamento
    # nova_amostra, que é a nova amostra a ser classificada
    # é criado o dicionário "dists" que armazena as distâncias euclidianas
    # entre a nova amostra e todas as amostras de treinamento.
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_euclidiana(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]
    # os K vizinhos mais próximos são selecionados com base nas menores distâncias calculadas,
    # usando o método "sorted" para ordenar o dicionário por valores de distância
    # e depois selecionando as chaves correspondentes aos K menores valores.
    #Depois de selecionar os K vizinhos mais próximos,
    # a função conta quantos exemplos de cada classe estão presentes entre esses vizinhos.
    # A classe mais frequente entre os K vizinhos é atribuída à nova amostra
    # e depois retornada pela função.

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 1:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 0

#Quantidade de acertos na predição
# Aqui o código está verificando se a classe atribuída pelo algoritmo K-NN para cada amostra de teste
# é a mesma que a classe real dessa amostra.
# Se a classe prevista for igual à classe real, isso significa que o modelo classificou
# corretamente a amostra de teste.
# A cada vez que o modelo classifica corretamente uma amostra de teste,
# a variável "acertos" é incrementada.
acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))

# Implementação KNN

# Fazendo o retorno dos dados
iris = load_iris()
iris

# Transformando em um DataFrame
data = pd.DataFrame(iris.data, columns = iris.feature_names)
data['target'] = iris.target

# Printando para visuzalizar a base de dados iris
print(data)

# Fazendo a seleção das colunas de pétala
petala = data.loc[data.target.isin([1,2]),['petal length (cm)','petal width (cm)','target']]
print(petala)

# Separando em X e y
X = petala[['petal length (cm)','petal width (cm)']]
y = petala.target

# Separando os dados em treino e teste
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)

# Criando vizinhos
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    # Configurando um classificador KNN
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Ajuste do modelo
    knn.fit(X_train, y_train)

    # Precisão de cálculo
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Test Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()

print("\nAcurácia (precisão) do treino: ",train_accuracies,"\nAcurácia (precisão) do teste: ", test_accuracies)

# Visualizando os dados de treino
fig, ax = plt.subplots()
ax.scatter(x=X_train['petal width (cm)'],
           y=X_train['petal length (cm)'],
           c=y_train,
           cmap='viridis')
ax.set(xlim=(0.9, 2.6), xticks=[1,1.5,2,2.5],
       ylim=(3, 7), yticks=[3,4,5,6,7])
plt.title('Dados de treino')
plt.show()

# Estabelecendo quantos vizinhos mais próximos serão utilizados
clf = KNeighborsClassifier(n_neighbors=3)

# Fazendo o fit com os dados de treino
clf = clf.fit(X_train,y_train)

# Fazendo a previsão para os dados de teste
y_pred = clf.predict(X_test)

# Visualização dos dados de treino e teste
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
plt.title('Dados de treino e teste')
plt.show()

print(X_test[y_test != y_pred])

print("\nCÁLCULO DAS DISTÂNCIAS")

# Distância Euclidiana
a = np.array(petala['petal length (cm)'])
b = np.array(petala['petal width (cm)'])
print("\nDistância Euclidiana: ")
dist_euc = np.sqrt(np.sum(np.square(a-b)))
print(dist_euc)

# Distância Manhattan
# Criando uma função para calcular
def manhattan_distance(point1, point2):
    return sum(abs(value1 - value2) for value1, value2 in zip(point1, point2))
# Definindo dois pontos
a = np.array(petala['petal length (cm)'])
b = np.array(petala['petal width (cm)'])
# Mostrando o resultado
print("\nDistância Manhattan: ")
print(manhattan_distance(a, b))

# Distância de Minkowski
# Definindo dois pontos
a = np.array(petala['petal length (cm)'])
b = np.array(petala['petal width (cm)'])
# Calculando a distância Minkowski com ordem p=3
p = 3
dist_minkowski = np.linalg.norm(a - b, ord=p)
# Mostrando o resultado
print("\nDistância de Minkowski: ")
print(dist_minkowski)

# Distância de Chebyshev
# Definindo dois pontos
a = np.array(petala['petal length (cm)'])
b = np.array(petala['petal width (cm)'])
# Printando na tela
print("\nDistância de Chebyshev: ")
# Calculando a distância Chebyshev entre os dois pontos
dist_chebyshev = np.amax(np.abs(a - b))
print(dist_chebyshev)
