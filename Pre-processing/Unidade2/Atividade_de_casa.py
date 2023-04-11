# implementação KNN

# importei o método utilizado: KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# importei o módulo pickle para guardar as variáveis de treino e teste
import pickle

# abri a base de dados "credit"
with open('credit.pkl', 'rb') as f:
    # Dividi em treino e teste
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

# Usarei 1500 para treinar o algoritmo
X_credit_treinamento.shape, y_credit_treinamento.shape

# Usarei 500 para teste
X_credit_teste.shape, y_credit_teste.shape

# Criei uma variável que recebe o número de vizinhos (o padrão é: 5), 'minkowski' é como o cálculo da distância será feito
# p = 2 é a medida euclidiana
knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
knn_credit.fit(X_credit_treinamento, y_credit_treinamento)

# Realizando a previsão, e assim podemos ver as respostas do algoritmo
previsoes = knn_credit.predict(X_credit_teste)
previsoes

# Comparando com os dados reais (ex.: os últimos 3 valores estão ok)
y_credit_teste

# Calcular a acurária
from sklearn.metrics import accuracy_score, classification_report

# Utilizando a padronização (cálculo que deixa os valores no mesmo padrão de escala)
accuracy_score(y_credit_teste, previsoes)
# temos o valor de 98%

print(classification_report(y_credit_teste, previsoes))
# Analisando o 'classification_report' podemos observar os resultados por cada uma das classes
# Esse algoritmo detecta 99% dos clientes que pagam (zero), e quando ele identifica está certo em 99% das vezes
# No caso dos clientes que não pagam, o algoritmo detecta 95% desses clientes, e quando ele identifica está certo em 94% das vezes


# Distância Manhattan
from scipy.spatial.distance import cityblock


# Distância de Minkowski
from math import *
from decimal import Decimal

def p_root(value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) **
                 Decimal(root_value), 3)

def minkowski_distance(x, y, p_value):
    return (p_root(sum(pow(abs(a - b), p_value)
                       for a, b in zip(x, y)), p_value))

vector1 = [0, 2, 3, 4]
vector2 = [2, 4, 3, 7]
p = 3
print('\nDistância de Minkowski:')
print(minkowski_distance(vector1, vector2, p))
'''