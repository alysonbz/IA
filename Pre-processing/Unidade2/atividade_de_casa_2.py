import math

# Carregar os dados
lista = []
with open('iris.data', 'r') as f:
    for linha in f.readlines():
        a = linha.replace('\n', '').split(',')
        # Ignorar linhas incompletas
        if len(a) < 5:
            continue
        # Mapeando classes para números
        if a[-1] == 'Iris-setosa':
            a[-1] = 1.0
        elif a[-1] == 'Iris-versicolor':
            a[-1] = 2.0
        elif a[-1] == 'Iris-virginica':
            a[-1] = 3.0
        try:
            # Convertendo os atributos para float, exceto a classe
            a[:-1] = list(map(float, a[:-1]))
        except ValueError:
            continue  # Ignorar linhas que não podem ser convertidas para float
        lista.append(a)

# Contar as classes
def countclasses(lista):
    setosa, versicolor, virginica = 0, 0, 0
    for i in range(len(lista)):
        if len(lista[i]) >= 5 and lista[i][4] == 1.0:
            setosa += 1
        elif len(lista[i]) >= 5 and lista[i][4] == 2.0:
            versicolor += 1
        elif len(lista[i]) >= 5 and lista[i][4] == 3.0:
            virginica += 1
    return setosa, versicolor, virginica

p = 0.6
setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
max_setosa = int(p * setosa)
max_versicolor = int(p * versicolor)
max_virginica = int(p * virginica)
total1, total2, total3 = 0, 0, 0

for lis in lista:
    if lis[-1] == 1.0 and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif lis[-1] == 2.0 and total2 < max_versicolor:
        treinamento.append(lis)
        total2 += 1
    elif lis[-1] == 3.0 and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)

# Função para calcular a distância euclidiana
def dist_euclidiana(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(soma)

# Função k-NN
def knn(treinamento, nova_amostra, K):
    dists = {}
    len_treino = len(treinamento)

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

# Avaliação do modelo
acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:
        acertos += 1

print("Porcentagem de acertos:", 100 * acertos / len(teste))