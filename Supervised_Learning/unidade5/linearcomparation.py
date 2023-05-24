#dataset
from src.utils import load_diabetes_clean_dataset
dados = load_diabetes_clean_dataset()

#bibliotecas
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#regressão logistica

X = dados.drop(['diabetes'], axis=1)
y = dados['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression
logreg = LogisticRegression(max_iter=1000)

# Fit the data to the model
logreg.fit(X_train, y_train)


# Usando train_test_split
accuracy = logreg.score(X_test, y_test)
print('Acurácia usando train_test_split:', accuracy)

# Usando cross-validation com k-fold
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(logreg, X, y, cv=k_fold)
print('Acurácia usando cross-validation e k-fold:', cv_accuracy.mean())
y_pred = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#grafico de barra
import matplotlib.pyplot as plt

# Acurácias da regressão logística e do KNN
acuracia_logreg = 0.746
acuracia_knn = 0.779
acuracia_logreg_Validation = 0.768
acuracia_knn_Validation = 0.735

# Preparação dos dados
models = ['Regressão Logística', 'KNN', 'Logreg_Valcross', 'KNN_Valcross']
accuracies = [acuracia_logreg, acuracia_knn, acuracia_logreg_Validation, acuracia_knn_Validation]

# Criando o gráfico de barras
plt.bar(models, accuracies, color=['pink', 'black'])
plt.ylim(0.7, 0.9)  # Definindo os limites do eixo y
plt.xlabel('Modelos')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia: Regressão Logística vs KNN')

# Exibindo o gráfico
plt.show()
