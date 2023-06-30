# Importar as bibliotecas necessárias


import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Carregar o conjunto de dados

lol = pd.read_csv(r'C:\Users\eryka\Downloads\Master_Ranked_Games.csv\Master_Ranked_Games.csv')


# Realizar a amostragem aleatória
sample_size = 1000
lol_sample = lol.sample(n=sample_size, random_state=42)

# Salvar o dataset lol_sample em um arquivo CSV
lol_sample.to_csv('lol_sample.csv', index=False)

print(lol_sample.info())

# Definir as variáveis de entrada (X) e a variável de saída (y)
X = lol_sample.drop('blueWins', axis=1)
y = lol_sample['blueWins']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando uma instância do StandardScaler
scaler = StandardScaler()

# Ajustando o scaler aos dados de treinamento e normalizando-os
X_train_normalized = scaler.fit_transform(X_train)

#calcular o mergings
mergings = linkage(X_train_normalized, method='complete')

# Plotando o dendograma
dendrogram(mergings,
           labels=y_train.values,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()


