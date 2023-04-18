"""
O k-NN é um método que ajuda a classificar novas amostras de dados com base em exemplos já conhecidos. Ele mede a
distância entre esses exemplos e uma nova amostra para determinar quão semelhantes elas são. Quanto mais próximos
elas estiveram, maior é a probabilidade de serem classificadas da mesma forma.
"""

"""
Para usar o k-NN:
1.Você começa com uma amostra que precisa ser classificada, mas que ainda não sabe a que classe pertence.
2.Em seguida, você calcula a distância entre essa amostra desconhecida e todas as outras amostras do conjunto de treinamento, 
usando uma medida de distância como a distância Euclidiana.
3.Depois, você identifica os K vizinhos mais próximos - ou seja, como K amostra que teve as menores distâncias em relação à 
amostra desconhecida.
4.Em seguida, você conta quantas vezes cada classe aparece entre os K vizinhos mais próximos.
5.Finalmente, você atribui a classe mais frequente aos vizinhos mais próximos como a classe da amostra desconhecida.
"""

"""
Para usar o k-NN, você precisa seguir estas etapas:
1. Obtenha um conjunto de dados.
2. Pré-processar os dados, como normalizá-los ou lidar com valores faltantes.
3. Divida o conjunto em um conjunto de treinamento e um conjunto de teste para avaliar a precisão do modelo.
4. Crie uma função que calcula a distância entre como exceção de dados.
5. Aplique o k-NN com base no conjunto de treinamento e use o modelo resultante para classificar novas amostras.
"""

"""
Dataset:
O repositório de aprendizado de máquina UCI foi usado para obter os dados usados ​​neste estudo. Existem 
três classes de flores: iris-setosa, iris-versicolor e iris-virginica, cada uma contendo 50 instâncias. O conjunto 
de dados tem 150 instâncias e 5 colunas. Cada coluna corresponde a um atributo, incluindo o comprimento e largura 
de sépalas, o comprimento e largura de pétalas, e a classe das flores.
"""

# INICIANDO O CÓDIGO - SEGUINDO AS ETAPAS
# Data set obtido do repositório de machine learning UCI

"Etapa 2 | Pré-Processamento"

lista = []
with open('iris_data.csv', 'r') as f:
    for linha in f.readlines():
        a = linha.replace('\n', '').split(',')
        lista.append(a)

# Lendo um arquivo e verifique a distribuição dos dados que estão dentro dele.
# Transformando cada conjunto de dados em uma pequena lista.
# Adicionando cada uma dessas listas em uma lista maior, que contém todas as instâncias do conjunto de dados.


def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        if lista[i][-1] == 0:
            setosa += 1
        if lista[i][-1] == 1:
            versicolor += 1
        if lista[i][-1] == 2:
            virginica += 1

    return [setosa, versicolor, virginica]

# Esta seção de código cria uma função chamada "countclasses", que recebe uma lista como entrada.
# A função é responsável por contar quantas flores pertencem a cada uma das três espécies - Setosa, Versicolor e Virginica.
# Em seguida, é aplicado o 'for' para percorrer cada elemento da lista.
# Ao final, a função retorna a lista contendo o número de flores para cada espécie.

"Etapa 3 | Dividindo o conjunto em teste e treino"

p = 0.6
setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = int(p * setosa), int(p * versicolor), int(p * virginica)
total1 = 0
total2 = 0
total3 = 0
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

# As instâncias de cada classe de flores foram divididas em conjuntos de treinamento e teste usando
# uma proporção fixa de 60% para treinamento e 40% para teste. Por exemplo, das 50 instâncias da classe
# iris-setosa, 60% foram selecionadas aleatoriamente para o conjunto de treinamento e os outros 40% foram
# para o conjunto de teste. O mesmo processo foi repetido para as outras classes.


# Ele usa o valor de 0,6 para a variável "p". Ele também chama a função "countclasses" que conta o número
# de instâncias de cada classe na lista e uma lista retorna com esses contagens.

# O código verifica cada elemento da lista de dados e, com base no valor da última posição (índice -1),
# que corresponde às classes, ele adiciona o elemento ao conjunto de treinamento se o número máximo de
# instâncias para aquele tipo de classe não existir sido atingido. Caso contrário, ele adiciona o
# elemento à lista de teste por meio do uso do "else".

"Etapa 4 | Definindo a função que calcula distância entre vetore"

import math

def dist_euclidiana(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(soma)

# Neste codigo usamos a Distancia Euclidiana:
# A distância euclidiana é um tipo de medida de distância utilizada para calcular a distância entre dois pontos em um
# plano. Essa distância pode ser encontrada aplicando-se o teorema de Pitágoras, que é amplamente conhecido na geometria.

# O código define uma função chamada "euclidean_distance" que calcula a distância euclidiana.
# A variável "dim" é definida como a dimensão do vetor v1 e a soma inicia-se no zero.
# Um "for" é usado para iterar sobre cada elemento dos vetores, exceto o último elemento.
# Dentro do loop, a diferença entre os elementos dos vetores é elevada ao quadrado e adicionada à variável soma.
# Após o loop, a função retorna a raiz quadrada da variável soma, resultando a distância

"Etapa 5 | Aplicando KNN"

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

# O código a cima classifica uma nova amostra desconhecida usando o algoritmo k-NN (k-Nearest Neighbors). Para isso, a função
# knn é definida com três parâmetros: o conjunto de treinamento, a nova amostra e o valor de K, que determina a quantidade de
# vizinhos mais próximos a serem considerados na classificação.
#A função começa contando quantos elementos de cada classe estão presentes nos K vizinhos mais próximos da nova amostra desconhecida.
# A classe mais frequente é então escolhida como a classe de destino da nova amostra.

# Dessa forma, o código classifica a nova amostra retirada do conjunto de teste.

acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:
        acertos += 1
print("Porcentagem de acertos:", 100 * acertos / len(teste))

# O código realiza a avaliação da precisão do algoritmo de classificação KNN utilizando um conjunto de teste. A primeira etapa é
# definir o valor de K que será usado no algoritmo: neste caso, K=1.

# Em seguida, é realizado um loop sobre todas as Sample do conjunto de teste. Para cada amostra, o algoritmo KNN é utilizado para
# classificá-la com base nos K vizinhos mais próximos encontrados no conjunto de treinamento.

# Após a classificação, é feita uma comparação entre a classe atribuída pelo KNN e a classe real da amostra de teste (que está mantida
# na última posição da lista da amostra). Se as duas classes forem iguais, é incrementada a contagem de acertos. Ao final do loop, a
# precisão do algoritmo é continuada como o número de acertos dividido pelo número total de amostra de teste.
