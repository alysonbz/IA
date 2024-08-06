#Importe as bibliotecas necessárias.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
sc = pd.read_csv(r"C:\Users\jonna\IA\AV1\dataset\star_classification_atualizado.csv")
print(sc.info())

#Normalize com a melhor normalização o conjunto de dados se houver melhoria.
print("Valores ausentes antes da imputação:")
print(sc.isnull().sum())

imputer = SimpleImputer(strategy='mean')
sc_imputed = pd.DataFrame(imputer.fit_transform(sc), columns=sc.columns)

print("\nValores ausentes após a imputação:")
print(sc_imputed.isnull().sum())

# Imputando valores ausentes
imputer = SimpleImputer(strategy='mean')
sc_imputed = pd.DataFrame(imputer.fit_transform(sc), columns=sc.columns)

# Separando as features e o target
sc_imputed['class'] = sc_imputed['class'].astype(int)
X = sc_imputed.drop(columns=['class'])  
y = sc_imputed['class']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustando valores negativos ou zeros para logaritmo
X_train[X_train <= 0] = 1e-9
X_test[X_test <= 0] = 1e-9

# Aplicando a normalização logarítmica
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# Buscando o melhor valor de k
accuracies_log = []
k_values = range(1, 21)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_log, y_train)
    y_pred = knn.predict(X_test_log)
    accuracies_log.append(accuracy_score(y_test, y_pred))

# Plotando o gráfico
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies_log, marker='o')
plt.title('Acurácia em função do valor de k (KNN)')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.xticks(k_values)
plt.grid(True)
plt.show()
