from src.utils import process_diabetes
from src.utils import load_diabetes_clean_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


diabetes_df = load_diabetes_clean_dataset()

X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)

# Ajustar o modelo aos dados de treino
knn.fit(X_train, y_train)

# Prever os rótulos dos dados de teste: y_pred
y_pred = knn.predict(X_test)


def compute_acuraccy(y_pred ,y_test):
    # Inicializa a variável de contagem
    acertos = 0

    # Percorre os exemplos de teste
    for i in range(len(y_test)):
        # Verifica se o modelo acertou
        if y_test[i] == y_pred[i]:
            acertos += 1

    # Calcula a acurácia
    acuracia = acertos / len(y_test)
    return acuracia

def compute_recall(y_pred , y_test):
    # Inicializa a variáveis de contagem
    verdadeiros_positivos = 0
    falsos_negativos = 0

    # Percorre os exemplos de teste
    for i in range(len(y_test)):
        # Verifica se o modelo acertou
        if y_test[i] == 1 and y_pred[i] == 1:
            verdadeiros_positivos += 1
        # Verifica se o modelo errou
        elif y_test[i] == 1 and y_pred[i] == 0:
            falsos_negativos += 1

    # Calcula o recall
    recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)
    return recall

def compute_precision(y_pred, y_test):
    # Inicializa as variáveis de contagem
    verdadeiros_positivos = 0
    falsos_positivos = 0

    # Percorre os exemplos de teste
    for i in range(len(y_test)):
        # Verifica se o modelo previu positivo
        if y_pred[i] == 1:
            # Verifica se o modelo acertou
            if y_test[i] == 1:
                verdadeiros_positivos += 1
            else:
                falsos_positivos += 1

    # Calcula a precisão
    precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
    return precisao


def compute_f1_score(y_pred,y_test):
    # Inicializa as variáveis de contagem
    verdadeiros_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    # Percorre os exemplos de teste
    for i in range(len(y_test)):
        # Verifica se o modelo previu positivo
        if y_pred[i] == 1:
            # Verifica se o modelo acertou
            if y_test[i] == 1:
                verdadeiros_positivos += 1
            else:
                falsos_positivos += 1
        else:
            # Verifica se o modelo errou
            if y_test[i] == 1:
                falsos_negativos += 1

    # Calcula a precisão e o recall
    precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
    recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)

    # Calcula a pontuação F1
    f1_score = 2 * (precisao * recall) / (precisao + recall)
    return f1_score

def compute_confusion_matriz(y_pred,y_test):
    # Inicializa as variáveis de contagem
    verdadeiros_positivos = 0
    verdadeiros_negativos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    # Percorre os exemplos de teste
    for i in range(len(y_test)):
        # Verifica se o modelo previu positivo
        if y_pred[i] == 1:
            # Verifica se o modelo acertou
            if y_test[i] == 1:
                verdadeiros_positivos += 1
            else:
                falsos_positivos += 1
        else:
            # Verifica se o modelo acertou
            if y_test[i] == 0:
                verdadeiros_negativos += 1
            else:
                falsos_negativos += 1

    # Retorna a matriz de confusão como uma lista de listas
    return [[verdadeiros_positivos, falsos_positivos], [falsos_negativos, verdadeiros_negativos]]

y_pred ,y_test = process_diabetes()

print("Acurácia geral: {}".format(compute_acuraccy(y_pred, y_test)*100))
print("recall classe 0: {}".format(compute_recall(y_pred,y_test == 0)*100))
print("recall classe 1: {}".format(compute_recall(y_pred,y_test == 1)*100))
print("precision classe 0: {}".format(compute_precision(y_pred,y_test == 0)*100))
print("precision classe 1: {}".format(compute_precision(y_pred,y_test == 1)*100))
print("F1-score classe 0: {}".format(compute_f1_score(y_pred,y_test == 0)*100))
print("F1-score classe 1: {}".format(compute_f1_score(y_pred,y_test == 1)*100))

