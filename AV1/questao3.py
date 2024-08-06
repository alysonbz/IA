#Importe as bibliotecas necessárias.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
sc = pd.read_csv(r"C:\Users\jonna\IA\AV1\dataset\star_classification_atualizado.csv")
print(sc.info())

#Normalize o conjunto de dados com normalização logarítmica  e verifique a acurácia do knn.

print("Valores ausentes antes da imputação:")
print(sc.isnull().sum())

imputer = SimpleImputer(strategy='mean')
sc_imputed = pd.DataFrame(imputer.fit_transform(sc), columns=sc.columns)

print("\nValores ausentes após a imputação:")
print(sc_imputed.isnull().sum())

sc_imputed['class'] = sc_imputed['class'].astype(int)
X = sc_imputed.drop(columns=['class'])  
y = sc_imputed['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train[X_train <= 0] = 1e-9
X_test[X_test <= 0] = 1e-9

X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

print("\nValores ausentes após a normalização logarítmica (treino):")
print(pd.DataFrame(X_train_log, columns=X_train.columns).isnull().sum())
print("\nValores ausentes após a normalização logarítmica (teste):")
print(pd.DataFrame(X_test_log, columns=X_test.columns).isnull().sum())

knn_log = KNeighborsClassifier()
knn_log.fit(X_train_log, y_train)
y_pred_log = knn_log.predict(X_test_log)
accuracy_log = accuracy_score(y_test, y_pred_log)




#Normalize o conjunto de dados com normalização de media zero e variância unitária e verifique a acurácia do knn.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier()
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

#Print as duas acuracias lado a lado para comparar. 
print(f"Acurácia com normalização logarítmica: {accuracy_log}")
print(f"Acurácia com normalização de média zero e variância unitária: {accuracy_scaled}")