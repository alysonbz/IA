from src.utils import log_reg_diabetes
from sklearn.metrics import classification_report, confusion_matrix
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

y_prob,y_test,y_pred = log_reg_diabetes()

# Calculate roc_auc_score
print(roc_auc_score(y_test, y_prob))

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test, y_pred))


# Relatório
''''# KNN
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix

diabetes_df = load_diabetes_clean_dataset()
pd.set_option('display.max_columns', None) # para mostrar todas as colunas
diabetes_df

X = diabetes_df[["pregnancies","glucose","diastolic","triceps","insulin","bmi","dpf","age"]].values
y = diabetes_df["diabetes"].values

# Dividindo em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criando vizinhos
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}
teste = True
for neighbor in neighbors:
    # Configurar um classificador KNN
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Ajuste do modelo
    knn.fit(X_train, y_train)

    # Precisão
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("Acurácia do treino: ",train_accuracies, '\n',"Acurácia do teste: ", test_accuracies)

# Adicionando um título
plt.title("\nKNN: Número variável de vizinhos")

# Plotando precisões de treinamento
plt.plot(neighbors, train_accuracies.values(), label="Precisão de treinamento")

# Plotando precisões de teste
plt.plot(neighbors, test_accuracies.values(), label="Precisão de teste")

plt.legend()
plt.xlabel("Número de vizinhos")
plt.ylabel("Precisão")

# Exibindo o gráfico
plt.show() # melhor k = 4


# Cross Validation
diabetes_cross = load_diabetes_clean_dataset()

X = diabetes_cross["dpf"].values.reshape(-1, 1)
y = diabetes_cross["diabetes"].values

# Criando o objeto Kfold
kf = KFold(n_splits=6, shuffle=True, random_state=5)

knn = KNeighborsClassifier()

# Calcular pontuações de validação cruzada 6 vezes
cv_scores = cross_val_score(knn, X, y, cv=kf)

# Print cv_scores
print("\nCross validation: ", cv_scores)

# Print da média
print("\nMédia: ", np.mean(cv_scores))

# Print o desvio padrão
print("\nDesvio padrão: ", np.std(cv_scores))

# Confusion Matrix
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

# Ajustando o modelo aos dados de treinamento
knn.fit(X_train, y_train)

# Prevendo os rótulos dos dados de teste: y_pred
y_pred = knn.predict(X_test)

# Gerando a matriz de confisão e o classification report
print("\nConfusion Matrix:\n ", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n ", classification_report(y_test, y_pred))'''




# REGRESSÃO LOGÍSTICA
'''from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from src.utils import log_reg_diabetes

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instanciando o modelo
logreg = LogisticRegression()

# Ajuste do modelo
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# Prevendo probabilidades
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print("\nProbabilidades previstas: ", y_pred_probs[:10])

# Gerando a matriz de confusão e o classification report
print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Criando X e y
X = diabetes_df["dpf"].values.reshape(-1, 1)
y = diabetes_df["diabetes"].values

# Criando o objeto KFold
kf = KFold(n_splits=6, shuffle=True, random_state=5)

Logreg = LogisticRegression()

# Calculando as pontuações de validação cruzada 6 vezes
cv_scores = cross_val_score(Logreg, X, y, cv=kf)

# Print cv_scores
print("\nCv score: ", cv_scores)

# Print da média
print("\nMédia: ", np.mean(cv_scores))

# Print do desvio padrão
print("\nDesvio padrão: ", np.std(cv_scores))


print("\nCurva ROC\n")

y_prob,y_test ,_= log_reg_diabetes()

# Gerar valores de curva ROC: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')

# Plotando tpr contra fpr
plt.plot(fpr, tpr)
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa Verdadeiros Positivos')
plt.title('Curva ROC para previsão de diabetes')
plt.show()'''