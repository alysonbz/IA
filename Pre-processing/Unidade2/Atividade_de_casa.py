"""O k-NN (K — Nearest Neighbors) é um algoritmo de classificação, usado para classificar novas amostras baseando-se
em um conjunto de treinamento.Ele parte do princípio de mínima distância calculada entre vetores, ou seja,
quão similar um dado é do outro com base na distância entre eles."""


""""A lógica consiste em:
Receber uma amostra desconhecida ou não classificada;
Calcular a distância(Euclidiana por exemplo) entre a amostra desconhecida e os outras do amostras do conjunto de treinamento;
Identificar os K vizinhos mais próximos, ou seja, as amostras que tiveram a menor distância entre a amostra não classificada.
Entre os K vizinhos mais próximos, identificar a frequência de todas as classes;
Tomar como resultado a maior frequência, ou seja, a que mais apareceu dentre os dados que tiveram as menores distâncias;"""

"""Etapas
Obter o dataset
Pré-processamento
Dividir o conjunto entre treino e teste
Criar função que calcula distância entre dados
Aplicar o knn com base no conjunto de treinamento e classificar novas amostras.
"""

"""Dataset
Os dados foram obtidos do repositório de machine learning UCI. No dataset
utilizado há 150 instâncias e 5 colunas. Nas colunas, os atributos são:
comprimento e largura de sépalas, comprimento e largura de pétalas, classe
das flores. As classes são três: iris-setosa, iris-versicolor e iris-virginica. Cada uma delas contém 50 instâncias."""

# INICIO DO CODIGO

"Pré-Processamento"
lista=[]
with open('iris.data', 'r') as f:
    for linha in f.readlines():
        a=linha.replace('\n','').split(',')
        lista.append(a)

### Explicação: Foi criada uma lista para armazenar cada sub-lista. uma função para somente ler o arquivo e transformar a instancia em uma sub-lista

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        if len(lista[i]) >= 5 and lista[i][4] == 1.0:
            setosa += 1
        elif len(lista[i]) >= 5 and lista[i][4] == 2.0:
            versicolor += 1
        elif len(lista[i]) >= 5 and lista[i][4] == 3.0:
            virginica += 1
    return setosa, versicolor, virginica

### Este código define uma função chamada "countclasses" que recebe uma lista como argumento.
### A função percorre a lista e conta quantas flores pertencem a cada uma das três espécies: Setosa, Versicolor e Virginica. Isso é feito por meio do uso de três variáveis contadoras - "setosa", "versicolor" e "virginica" - que tem valor 0 no inicio da função
### É feito um loop (for) para repetir cada elemnto da lista. para cada item, a função verifica o indice 4 da lista, que contem a especie da flor. de acordo com a variavel é incrementada +1 em cada especie
### Retorna a lista contendo o número de cada espécie

"Dividindo o conjunto em teste e treino"
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
### Este código divide uma lista de dados em duas partes: treinamento e teste. Ele começa definindo um valor para a variável "p", que é definida como 0.6.
### ele chama a função "countclasses" que recebe a lista de dados como argumento e retorna uma lista com as contagens de cada classe.
### O código então define três variáveis contadoras - "total1", "total2" e "total3" - inicializadas com o valor zero.
### Para cada elemento, o código verifica se o valor da última posição (índice -1) é igual a 1.0, 2.0 ou 3.0, que são as três classes. dependendo do indice e caso o numero máximo do conjunto de treinamento ainda nao foi alcançado, o elemento é adicionado ao conjunto de treinamento.
### caso o elemento não atenda aos requisitos ele é adicionado a lista de testes por meio do "else"

import math
def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)

### o codigo cria uma função "dist_euclidiana", calcula a distancia euclidiana entre dois vetores (v1,v2)
### é definido a variavel "dim": dimensão do vetor v1; soma: recebe 0
### é criado um loop "for" para percorrer os elemnetos dos vetores, menos o ultimo elemento (-1)
### Após o loop, a função retorna a raiz quadrada de "soma", que é o resultado da distância euclidiana entre os dois vetores.

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


### é definida uma função "knn" com tres parametros: treinamento, nova_amostra, K.
### A função cria um dicionário "dists" que armazena as distâncias euclidianas entre a nova amostra e as amostras de treinamento.
### é criado um loop "for" para percorrer os itens de treinamento, calculando a distancia euclidiana usando a função anterior "dist_euclidiana"
### A função seleciona os "K" vizinhos mais próximos da nova amostra a partir do dicionário "dists", usando a função "sorted". O resultado é armazenado na lista "k_vizinhos".
### A função em sequida inicializa três variáveis para contar a quantidade de amostras de cada classe nos "K" vizinhos mais próximos. Em um loop "for", a função verifica a classe de cada vizinho selecionado e incrementa a contagem.
### A funão cria uma lista "a" com a contagem de cada classe. e retorna o indice de valor máximo, adicionando 1 ao resultado anterior.
### A função "knn" implementa o algoritmo KNN para classificar uma nova amostra com base nas classes das "K" mais próximas, calculando a distância euclidiana entre a nova amostra e as amostras de treinamento

acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))

### O código define o número de vizinhos K que serão usados para classificar a amostra de teste: 1
### incrementa sobre todas as amostras de teste e usa a função "knn" para classificar com base nos K vizinhos mais próximos encontrados no conjunto de treinamento.
### Se a classe atribuída pelo KNN for igual à classe da amostra de teste (ou seja, amostra[-1]), a contagem de acertos é incrementada
