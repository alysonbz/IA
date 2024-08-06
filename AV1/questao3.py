#Neste exercicio você deve verificar se a normalização interfere nos resultados de sua classificação.

#Importe as bibliotecas necessárias.
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.utils import load_cancer_dataset_cleaned

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
cancer = load_cancer_dataset_cleaned()
print(cancer)
print('4.1) Normalize o conjunto de dados com normalização logarítmica  e verifique a acurácia do knn.')

#Também serve para separar os valores numéricos e assim, usar somente a variável x para normalizar.
x = cancer.drop(columns=['diagnosis'])
y = cancer['diagnosis']

#Aplicando a normalização Logarítimica
x_log_normalized = np.log1p(x)

#treino e testes
x_train, x_test, y_train, y_test = train_test_split(x_log_normalized, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
#treinando com os dados de treino
knn.fit(x_train, y_train)
#previsões nos dados de teste
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)


print('4.2) Normalize o conjunto de dados com normalização de media zero e variância unitária e  verifique a acurácia do knn.')
#normalização Z-score (média zero e variância unitária)
scaler = StandardScaler()
x_zscore_normalized = scaler.fit_transform(x)
# treino e teste
x_train_zscore, x_test_zscore, y_train_zscore, y_test_zscore = train_test_split(x_zscore_normalized, y, test_size=0.2, random_state=42)
knn_zscore = KNeighborsClassifier(n_neighbors=3)
#treinando com os dados de treino
knn_zscore.fit(x_train_zscore, y_train_zscore)
#previsões nos dados de teste
y_pred_zscore = knn_zscore.predict(x_test_zscore)
accuracy_zscore = accuracy_score(y_test_zscore, y_pred_zscore)


print('4.3) Print as duas acuracias lado a lado para comparar.')
print("Acurácia com normalização logarítmica é: {:.2f}%".format(accuracy * 100))
print(f"Acurácia com Normalização Z-score: {accuracy_zscore * 100:.2f}%")