import math

lista=[]
with open('iris.data', 'r') as f:
    for linha in f.readlines():
        a=linha.replace('\n','').split(',')
        if a[-1] == 'Iris-setosa':
            a[-1] = 1.0
        if a[-1] == 'Iris-versicolor':
            a[-1] = 2.0
        if a[-1] == 'Iris-virginica':
            a[-1] = 3.0
        try :
            valores = [float(val) for val in a]
        except:
            print(valores)
        lista.append(valores)


def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for linha in lista:
        classe = linha[-1]  # A classe está na última posição de cada linha
        if classe == 1.0:
            setosa += 1
        elif classe == 2.0:
            versicolor += 1
        elif classe == 3.0:
            virginica += 1

    return [setosa, versicolor, virginica]

p=0.7
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

def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)


def knn(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_euclidiana(treinamento[i][:-1], nova_amostra[:-1])  # Excluímos a classe da amostra antes de calcular a distância
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

acertos, K = 0, 10
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))
