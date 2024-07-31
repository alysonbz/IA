import math

# Leitura dos dados e conversão para o formato adequado
lista = []
with open('iris.data', 'r') as f:
    for linha in f.readlines():
        if linha.strip():  # Ignora linhas vazias
            a = linha.strip().split(',')
            a[:4] = list(map(float, a[:4]))  # Converte atributos para float
            lista.append(a)

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for item in lista:
        if item[4] == "Iris-setosa":
            setosa += 1
        elif item[4] == "Iris-versicolor":
            versicolor += 1
        elif item[4] == "Iris-virginica":
            virginica += 1
    return setosa, versicolor, virginica

# Divisão dos dados em treinamento e teste
p = 0.6
setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = int(p * setosa), int(p * versicolor), int(p * virginica)
total1, total2, total3 = 0, 0, 0

for lis in lista:
    if lis[4] == "Iris-setosa" and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif lis[4] == "Iris-versicolor" and total2 < max_versicolor:
        treinamento.append(lis)
        total2 += 1
    elif lis[4] == "Iris-virginica" and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)

# Função para calcular a distância euclidiana
def dist_euclidiana(v1, v2):
    dim, soma = len(v1) - 1, 0  # Ignora a última coluna (classe)
    for i in range(dim):
        soma += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(soma)

# Função KNN
def knn(treinamento, nova_amostra, K):
    dists = []
    for i in range(len(treinamento)):
        d = dist_euclidiana(treinamento[i], nova_amostra)
        dists.append((d, treinamento[i][4]))  # Armazena a distância e a classe

    dists.sort(key=lambda x: x[0])  # Ordena pela distância
    k_vizinhos = dists[:K]  # Seleciona os K vizinhos mais próximos

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for d in k_vizinhos:
        if d[1] == "Iris-setosa":
            qtd_setosa += 1
        elif d[1] == "Iris-versicolor":
            qtd_versicolor += 1
        elif d[1] == "Iris-virginica":
            qtd_virginica += 1

    # Retorna a classe com maior frequência
    if qtd_setosa > qtd_versicolor and qtd_setosa > qtd_virginica:
        return "Iris-setosa"
    elif qtd_versicolor > qtd_setosa and qtd_versicolor > qtd_virginica:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"

# Teste e cálculo da precisão
acertos, K = 0, 3  # K pode ser ajustado conforme necessário
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[4] == classe:
        acertos += 1

print("Porcentagem de acertos:", 100 * acertos / len(teste))
