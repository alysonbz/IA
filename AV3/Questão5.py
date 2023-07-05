import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregar o dataset pré-processado
df = pd.read_csv("mitbih_train.csv")
# Separar os atributos das classes
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classificador Regressão Logística
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

# Métricas de avaliação da Regressão Logística
logreg_accuracy = accuracy_score(y_test, logreg_pred)
logreg_precision = precision_score(y_test, logreg_pred, average='macro')
logreg_recall = recall_score(y_test, logreg_pred, average='macro')
logreg_f1 = f1_score(y_test, logreg_pred, average='macro')

# Imprimir as métricas de avaliação
print("Regressão Logística:")
print("Acurácia:", logreg_accuracy)
print("Precisão:", logreg_precision)
print("Recall:", logreg_recall)
print("F1-Score:", logreg_f1)
