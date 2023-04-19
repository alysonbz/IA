##O k-NN é um algoritmo de classificação, usado para classificar novas amostras baseando-se em um conjunto de treinamento.
# Ele parte do princípio de mínima distância calculada entre vetores, ou seja, quão similar um dado é do outro com base na distância entre eles.


##Receber uma amostra desconhecida ou não classificada;
##Calcular a distância entre a amostra desconhecida e os outras do amostras do conjunto de treinamento;
##Identificar os K vizinhos mais próximos, ou seja, as amostras que tiveram a menor distância entre a amostra não classificada.
##Entre os K vizinhos mais próximos, identificar a frequência de todas as classes;
##Tomar como resultado a maior frequência, ou seja, a que mais apareceu dentre os dados que tiveram as menores distâncias;
#Ler o arquivo e verificar a distribuição dos dados#Transformar cada instância do conjunto em uma sub lista.


lista=[]
with open('iris.data.csv', 'r') as f:
    for linha in f.readlines():
        a=linha.replace('\n','').split(',')
        if a[-1] == "Iris-setosa":            a[-1] = 1
        if a[-1] == "Iris-versicolor":
            a[-1] = 2
        if a[-1] == "Iris-virginica":            a[-1] = 3
        try:           a = [float(i) for i in a]
        except:          continue
        lista.append(a)

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        if lista[i][-1] == 1:
            setosa += 1
        if lista[i][-1] == 2:
            versicolor += 1
        if lista[i][-1] == 3:
            virginica += 1
    return [setosa, versicolor, virginica]

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

acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))



