
# 1) Importar bibliotecas necessárias 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 2) Carregar o dataset
data = pd.read_csv('diabetes_atualizado.csv')
print(diabetes_atualizado.head())

 X = data.drop('Outcome', axis=1)  
 y = data['Outcome']


 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # 3) Normalização logarítmica
 scaler_log = MinMaxScaler()
 X_train_log = scaler_log.fit_transform(X_train)
 X_test_log = scaler_log.transform(X_test)


 knn_log = KNeighborsClassifier(n_neighbors=5)  
 knn_log.fit(X_train_log, y_train)


 y_pred_log = knn_log.predict(X_test_log)
 acuracia_log = accuracy_score(y_test, y_pred_log)

 # 4) Normalização de média zero e variância unitária
 scaler_std = StandardScaler()
 X_train_std = scaler_std.fit_transform(X_train)
 X_test_std = scaler_std.transform(X_test)


 knn_std = KNeighborsClassifier(n_neighbors=5)
 knn_std.fit(X_train_std, y_train)


 y_pred_std = knn_std.predict(X_test_std)
acuracia_std = accuracy_score(y_test, y_pred_std)

 # 5) Print das acurácias
 print(f'Acurácia com normalização logarítmica: {acuracia_log:.4f}')
 print(f'Acurácia com normalização de média zero e variância unitária: {acuracia_std:.4f}')