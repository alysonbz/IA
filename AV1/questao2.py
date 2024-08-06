import math
import string

val = []
lista = []
CleanWaterQuality = []


def isString(args):
    letters = string.ascii_letters
    for arg in args:
        if arg in letters:
            return True
    return False

#CleanWaterQuality = pd.read_csv('dataset/CleanWaterQuality1.csv')
with open('dataset/CleanWaterQuality1.csv', 'r') as f:
    for linha in f.readlines():
        if not isString(linha):
            a = linha.replace('\n', '').split(',')
            val = [float(component) for component in a]
            CleanWaterQuality.append(val)
def countclasses(coluna):
    seguro, nao_seguro = 0, 0
    for elemento in coluna:
        if elemento == 0:
            nao_seguro += 1
        else:
            seguro += 1

    return seguro, nao_seguro

seguro, nao_seguro = countclasses([linha[-1] for linha in CleanWaterQuality])


p = 0.6
#setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
#max_setosa, max_versicolor, max_virginica = int(p * setosa), int(p * versicolor), int(p * virginica)
max_seguro, max_nao_seguro = int(p*seguro), int(p*nao_seguro)
#print(max_seguro, max_nao_seguro)

total1 = 0
total2 = 0

for row in CleanWaterQuality:
    if row[-1] == 1 and total1 < max_seguro:
        treinamento.append(row)
        total1 += 1
    elif row[-1] == 0 and total2 < max_nao_seguro:
        treinamento.append(row)
        total2 += 1
    else:
        teste.append(row)
    """elif lis[-1] == 3.0 and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1"""
def dist_euclidiana(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(soma)


'''def calcular_media(dados):

    num_amostras = len(dados)
    num_caracteristicas = len(dados[0])

    # Inicializa uma lista para armazenar a soma das características
    soma = [0] * num_caracteristicas

    # Somar os valores de cada característica
    for amostra in dados:
        for i in range(num_caracteristicas):
            soma[i] += amostra[i]

    # Calcular a média dividindo a soma pelo número de amostras
    media = [s / num_amostras for s in soma]

    return media
def calcular_covariancia(data):
    num_linhas, num_colunas = len(data), len(data[0])
    media = [sum(col) / num_linhas for col in zip(*data)]

    covariancia = [[0] * num_colunas for _ in range(num_colunas)]

    for i in range(num_colunas):
        for j in range(num_colunas):
            covariancia[i][j] = sum((data[k][i] - media[i]) * (data[k][j] - media[j]) for k in range(num_linhas)) / (
                        num_linhas - 1)

    return covariancia


def inversa_matriz(matriz):
    n = len(matriz)
    A = [row[:] + [0] * n for row in matriz]  # Cria uma matriz aumentada
    for i in range(n):
        A[i][i + n] = 1  # Adiciona a matriz identidade

    for i in range(n):
        pivot = A[i][i]
        if pivot == 0:
            raise ValueError("Matriz não é invertível")

        for j in range(n * 2):
            A[i][j] /= pivot

        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(n * 2):
                    A[k][j] -= factor * A[i][j]

    return [row[n:] for row in A]

def distancia_mahalanobis(ponto, media, matriz_covariancia):
    inv_cov = inversa_matriz(matriz_covariancia)

    diferenca = [ponto[i] - media[i] for i in range(len(ponto))]
    distancia = sum(
        diferenca[i] * sum(inv_cov[i][j] * diferenca[j] for j in range(len(ponto)))
        for i in range(len(ponto))
    )

    return math.sqrt(distancia)'''
import numpy as np
X_train = [linha[:-1] for linha in treinamento]

cov_matrix = np.cov(X_train, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

def dist_mahalanobis(v1, v2, inv_cov):
    diff = np.array(v1) - np.array(v2)
    return math.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
def distancia_de_chebyshev(p, q):
        if len(p) != len(q):
            raise ValueError("Os pontos devem ter a mesma dimensão")

        max_diferenca = 0

        for i in range(len(p)):
            # Calcula a diferença absoluta entre as coordenadas correspondentes
            diferenca = abs(p[i] - q[i])

            # Atualiza o valor máximo da diferença
            if diferenca > max_diferenca:
                max_diferenca = diferenca

        return max_diferenca


def distance_de_manhattan(ponto1, ponto2):

    return sum(abs(a - b) for a, b in zip(ponto1, ponto2))

def knn(treinamento, nova_amostra, K, distancia):
    dists, len_treino = {}, len(treinamento)
    for i in range(len_treino):
        if distancia == 'manhattan':
            d = distance_de_manhattan(CleanWaterQuality[i], nova_amostra)
        elif distancia == 'chebyshev':
            d = distancia_de_chebyshev(CleanWaterQuality[i], nova_amostra)
        elif distancia == 'euclidiana':
            d = dist_euclidiana(treinamento[i], nova_amostra)
        elif distancia == 'mahalanobis':
            '''media = calcular_media(CleanWaterQuality)
            covariancia = calcular_covariancia(CleanWaterQuality)
            d = distancia_mahalanobis(CleanWaterQuality[i], media, covariancia)'''
            d = dist_mahalanobis(treinamento[i][:-1], nova_amostra[:-1], inv_cov_matrix)


        dists[i] = d
    k_vizinhos = sorted(dists, key=dists.get)[:K]
    qtd_seguro, qtd_nao_seguro = 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1:
            qtd_seguro += 1
        else:
            qtd_nao_seguro += 1
    a = [qtd_seguro, qtd_nao_seguro]
    return a.index(max(a)) + 1.0
print(f'Train: {len(treinamento)}, Test: {len(teste)}')
acertos_manhattan, acertos_chebyshev, acertos_euclidiana, acertos_mahalanobis, K = 0, 0, 0, 0, 5

for amostra in teste:
    classe_manhattan = knn(treinamento, amostra, K, 'manhattan')
    if int(amostra[-1]) == classe_manhattan:
        acertos_manhattan += 1

for amostra in teste:
    classe_chebyshev = knn(treinamento, amostra, K, 'chebyshev')
    if int(amostra[-1]) == classe_chebyshev:
        acertos_chebyshev += 1

for amostra in teste:
    classe_euclidiana = knn(treinamento, amostra, K, "euclidiana")
    if int(amostra[-1]) == classe_euclidiana:
        acertos_euclidiana += 1

for amostra in teste:
    classe_mahalanobis = knn(treinamento, amostra, K, 'mahalanobis')
    if int(amostra[-1]) == classe_mahalanobis:
        acertos_mahalanobis += 1




def acuracia(acertos, teste):
    return 100 * acertos / len(teste)

print("Acuracia distancia de chebyshev:", acuracia(acertos_chebyshev, teste))
print("Acuracia distancia de manhattan:", acuracia(acertos_manhattan, teste))
print("Acuracia distancia de euclidiana:", acuracia(acertos_euclidiana, teste))
print("Acuracia distancia de mahalanobis:", acuracia(acertos_mahalanobis, teste))
