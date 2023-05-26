#dataset
#ruan
from src.utils import load_diabetes_clean_dataset
dados = load_diabetes_clean_dataset()

#bibliotecas
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print(dados.columns)
#KNN


X = dados.drop('diabetes', axis=1)
y = dados['diabetes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print('Acurácia do KNN sem cross-validation:', accuracy)

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(knn, X, y, cv=k_fold)
print('Acurácia usando cross-validation do KNN:', cv_accuracy.mean())



#Matriz de confusão
y_pred = knn.predict(X_test)
print('Matriz de confusão do KNN:')
print(confusion_matrix(y_test, y_pred))

#classification report
print('Classification report do KNN:')
print(classification_report(y_test, y_pred))

#regressão logistica

X = dados.drop(['diabetes'], axis=1)
y = dados['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression
logreg = LogisticRegression()


# Fit the data to the model
logreg.fit(X_train, y_train)


# Usando train_test_split
accuracy = logreg.score(X_test, y_test)
print('Acurácia da logistica de regressão sem cross-validation:', accuracy)


# Usando cross-validation com k-fold
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(logreg, X, y, cv=k_fold)
print('Acurácia usando cross-validation da logistica de regressão:', cv_accuracy.mean())

#Matriz de confusão
y_pred = logreg.predict(X_test)
print('Matriz de confusão de logística de regressão:')
print(confusion_matrix(y_test, y_pred))


#classification report
print('Classification report da regressão logistica:')
print(classification_report(y_test, y_pred))

#grafico de barra
import matplotlib.pyplot as plt


# Acurácias da regressão logística e do KNN
acuracia_logreg = 0.746
acuracia_knn = 0.779
acuracia_logreg_Validation = 0.768
acuracia_knn_Validation = 0.735


# Preparação dos dados
models = ['Regressão Logística', 'KNN']
accuracies = [acuracia_logreg, acuracia_knn]


# Criando o gráfico de barras
plt.bar(models, accuracies, color=['black', 'gray'])
plt.ylim(0.7, 0.9)  # Definindo os limites do eixo y
plt.xlabel('Modelos')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia: Regressão Logística vs KNN')


# Exibindo o gráfico
plt.show()

