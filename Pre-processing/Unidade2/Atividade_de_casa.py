import pandas as pd
import csv


#código começa criando uma lista vazia chamada "lista". Em seguida, ele abre o arquivo "iris.data" no modo de
#leitura usando a função "open" com o parâmetro "r". A cláusula "with" é usada para garantir que o arquivo seja
#fechado corretamente após o uso.
#o código usa um loop "for" para iterar sobre as linhas do arquivo. A função "readlines" é usada para ler todas
#as linhas do arquivo e retorná-las como uma lista. O loop "for" então itera sobre cada uma dessas linhas.
#Dentro do loop, o código remove o caractere de quebra de linha da string "linha" usando o método "replace" e,
#em seguida, divide a string em substrings usando a vírgula como separador. Essas substrings são armazenadas em
#uma lista chamada "a".
#Finalmente, a lista "a" é adicionada à lista "lista" usando o método "append". Após a execução desse código, a
#lista "lista" conterá uma lista de listas, onde cada sublista contém as informações de uma flor iris presente
#no arquivo "iris.data".
lista=[]
with open('iris_data.data', 'r') as f:
    for linha in f.readlines():
        a=linha.replace('\n','').split(',')
        lista.append(a)

#Esse código define uma função chamada "countclasses" que recebe uma lista como argumento.
#Dentro da função, três variáveis (setosa, versicolor e virginica) são inicializadas com o valor zero.
#Em seguida, um loop "for" é usado para iterar sobre um intervalo de índices com base no comprimento da lista de
#entrada.
#Dentro do loop, o código usa as condições "if" para verificar se a linha atual da lista tem pelo menos cinco
#elementos e, em seguida, verifica se o valor do quinto elemento é igual a 1.0 (para setosa), 2.0 (para
#versicolor) ou 3.0 (para virginica).
#Se a condição for verdadeira, o contador apropriado (setosa, versicolor ou virginica) é incrementado em 1
#Depois que o loop termina, a função retorna os valores dos contadores setosa, versicolor e virginica como uma
#tupla (conjunto de valores separados por vírgula dentro de parênteses).
#Em resumo, essa função conta quantas flores são da classe setosa, versicolor e virginica, com base no quinto
#elemento de cada linha na lista de entrada


def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        if len(lista[i]) >= 5 and lista[i][4] == 1.0:
            setosa += 1
        if len(lista[i]) >= 5 and lista[i][4] == 2.0:
            versicolor += 1
        if len(lista[i]) >= 5 and lista[i][4] == 3.0:
            virginica += 1

    return [setosa, versicolor, virginica]

#Este código divide uma lista de dados de flores iris em duas sub-listas, uma para treinamento e outra
# para teste, com base em uma proporção p (neste caso, 0.6).
#Primeiro, a função "countclasses" é chamada com a lista de dados como argumento, e os valores de retorno são
#atribuídos às variáveis "setosa", "versicolor" e "virginica".
#Em seguida, três variáveis são inicializadas com a proporção de cada classe (max_setosa, max_versicolor e
#max_virginica), multiplicando a contagem de cada classe pelo valor de p.
#Em seguida, o código inicializa três contadores (total1, total2 e total3) com o valor zero para contar quantas
#flores de cada classe já foram adicionadas à lista de treinamento.
#Então, um loop "for" é usado para iterar sobre cada linha da lista de entrada. Para cada linha, o código
#verifica o valor do último elemento na lista (que representa a classe da flor), e adiciona a linha à lista de
#treinamento correspondente (setosa, versicolor ou virginica) se o número de linhas adicionadas até o momento
#não exceder o valor máximo permitido para a classe correspondente. Caso contrário, a linha é adicionada à lista
#de teste.
#Finalmente, as sub-listas de treinamento e teste são retornadas como saída.
#Em resumo, este código realiza a divisão dos dados em subconjuntos de treinamento e teste com base em uma
#proporção p, usando as contagens de cada classe de flor para garantir que a proporção de cada classe seja
#mantida em ambas as sub-listas.
p=0.6
setosa, versicolor, virginica = countclasses(lista)
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

#Este código define uma função chamada "dist_euclidiana" que calcula a distância euclidiana entre dois vetores,
# representados como listas de números.
#A distância euclidiana é uma medida de distância entre dois pontos em um espaço euclidiano, que corresponde à
# distância linear mais curta entre esses pontos. É calculada pela raiz quadrada da soma dos quadrados das
# diferenças entre as coordenadas dos dois pontos.
#Na implementação desta função, a dimensão dos vetores (o número de elementos em cada lista) é determinada
# usando a função "len", e uma variável "soma" é inicializada com o valor zero.
#Então, um loop "for" é usado para iterar sobre o índice dos elementos em uma das listas (neste caso, v1), e o
# quadrado da diferença entre o elemento atual de v1 e o elemento correspondente em v2 é adicionado à soma.
#O loop é executado até a penúltima dimensão do vetor, já que a última dimensão é usada para armazenar a classe
# da flor e não faz parte do cálculo da distância euclidiana.
#Finalmente, a raiz quadrada da soma é calculada usando a função "sqrt" do módulo "math" e retornada como
# saída da função.
#Em resumo, esta função calcula a distância euclidiana entre dois vetores de mesma dimensão.
import math
def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)

#Este código define uma função chamada "knn" que implementa o algoritmo K-NN (K-Nearest Neighbors) para
#classificação de dados. O algoritmo K-NN é uma técnica de aprendizado de máquina não paramétrica, que
#classifica uma nova amostra com base na classe dos K vizinhos mais próximos dessa amostra no conjunto de
#treinamento.
#A função "knn" recebe três argumentos: a lista de treinamento (um conjunto de vetores com suas respectivas
#classes), uma nova amostra (um vetor que precisa ser classificado) e o parâmetro K, que determina o número de
#vizinhos a serem considerados para a classificação.
#A função começa inicializando um dicionário "dists" e a variável "len_treino" com o comprimento da lista de
#treinamento. Em seguida, ela itera sobre cada vetor na lista de treinamento e calcula a distância euclidiana
#entre esse vetor e a nova amostra usando a função "dist_euclidiana" definida anteriormente. Essas distâncias
#são armazenadas no dicionário "dists", usando o índice do vetor na lista de treinamento como chave.
#Após calcular as distâncias para todos os vetores no conjunto de treinamento, a função seleciona os K índices
#de vetores com as menores distâncias (isto é, os K vizinhos mais próximos) usando a função "sorted" e o
#método "get" do dicionário "dists".
#Então, a função itera sobre os K vizinhos mais próximos e conta quantos deles pertencem a cada uma das três
#classes possíveis (Setosa, Versicolor e Virginica). Essas contagens são armazenadas nas variáveis
#"qtd_setosa", "qtd_versicolor" e "qtd_virginica".
#Finalmente, a função cria uma lista com as contagens das três classes e retorna o índice da classe com o
#maior número de vizinhos mais próximos, somando 1 para corresponder ao valor real da classe (1 para Setosa, 2
# para Versicolor e 3 para Virginica). Essa classe é a classe prevista para a nova amostra.


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

#Este é o trecho final de um algoritmo de classificação K-NN, que realiza a avaliação da precisão do modelo com
# uma porcentagem de acertos.
#A variável "acertos" é inicializada com 0 e é incrementada toda vez que a classe prevista pelo K-NN é igual à
#classe real da amostra de teste.
#O laço "for" percorre todas as amostras de teste e chama a função "knn" passando como parâmetros o conjunto de
#treinamento, a amostra de teste e o valor de K. Em seguida, a classe prevista é armazenada na variável "classe".
#Por fim, a precisão do modelo é calculada dividindo o número de acertos pelo total de amostras de teste e
#multiplicando por 100 para obter a porcentagem de acertos. O resultado é impresso na tela.

acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))



