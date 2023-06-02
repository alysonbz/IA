
#Questão 1
from src.utils import load_diabetes_clean_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Carregar os dados
diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Criar e ajustar o modelo
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

# Plotar a matriz de confusão
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.RdPu)  # Definir o colormap como 'RdPu' (rosa)
plt.title('Matriz de Confusão')
plt.colorbar()

classes = ['Classe 0', 'Classe 1']  # Nomes das classes

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Preenchimento da matriz de confusão com os valores
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
plt.show()
