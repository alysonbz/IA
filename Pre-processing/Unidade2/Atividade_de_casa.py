
#Visualização de o que estar sendo processado
with open('iris.data', 'r') as f:
    for linha in f:
        print(linha)

lista=[]
# Processamento
with open('iris.data', 'r') as f:
    for linha in f.readlines():
        a=linha.replace('\n','').split(',')
        lista.append(a)

#Visualização dos dados ja processados
print(lista)

# Esse algoritmo vai contar o número de ocorrencias das 3 classes. Ele vai ultilizar o for para percorrer
#a lista e verificar se o ultimo o quarto indice corresponde a 1, 2 ou 3. E contabilizar nas variáveis locais.

classes_dict = {
    "Iris-setosa": 1.0,
    "Iris-versicolor": 2.0,
    "Iris-virginica": 3.0
}

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)-1):
        classe = classes_dict[lista[i][4]]
        if classe == 1.0:
            setosa += 1
        if classe == 2.0:
            versicolor += 1
        if classe == 3.0:
            virginica += 1


    return [setosa, versicolor, virginica]

print(countclasses(lista))

p=0.6
setosa,versicolor, virginica = countclasses(lista)
treinamento, teste= [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1 =0
total2 =0
total3 =0
for lis in lista:
    if lis[-1]==1.0 and total1< max_setosa:
        treinamento.append(lis)
        total1 +=1
    elif lis[-1]==2.0 and total2<max_versicolor:
        treinamento.append(lis)
        total2 +=1
    elif lis[-1]==3.0 and total3<max_virginica:
        treinamento.append(lis)
        total3 +=1
    else:
        teste.append(lis)


import math
def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)


def knn(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_euclidiana(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
iris = load_iris()

# Divide o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Cria uma lista de valores de K para avaliar
k_values = list(range(1, 50))

# Armazena as precisões em uma lista
accuracies = []

for k in k_values:
    # Treina o modelo com o valor de K atual
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Faz as previsões no conjunto de teste
    y_pred = knn.predict(X_test)

    # Calcula a precisão do modelo e armazena na lista
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plota a curva de validação
plt.plot(k_values, accuracies)
plt.xlabel('Valor de K')
plt.ylabel('Precisão')
plt.show()


acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))

