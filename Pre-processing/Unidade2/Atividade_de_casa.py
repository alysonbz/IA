#leitura do arquivo
#o comando with open abre o arquivo iris_data em modo leitura por conta do 'r'
lista=[]
with open('iris.data', 'r') as f:
    for linha in f.readlines(): # o loop for lê cada linha usando o método readlines()
        a=linha.replace('\n','').split(',')
        lista.append(a) #adicionando cada sub lista em uma lista

#função countclasses para contar o número de sublistas
def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)): #percorrendo cada elemento da lista
        if lista[i][4] == 1.0:
            setosa += 1
        if lista[i][4] == 2.0:
            versicolor += 1
        if lista[i][4] == 3.0:
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
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0

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
