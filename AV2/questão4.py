import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Carregando o dataset
data = pd.read_csv(r'C:\\Users\\jonna\\IA\\AV2\\dataset\\carprice.csv') 

# Removendo valores nulos
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Convertendo colunas relevantes para o tipo numérico
data['horsepower'] = pd.to_numeric(data['horsepower'])
data['curb-weight'] = pd.to_numeric(data['curb-weight'])
data['engine-size'] = pd.to_numeric(data['engine-size'])
data['width'] = pd.to_numeric(data['width'])
data['length'] = pd.to_numeric(data['length'])
data['price'] = pd.to_numeric(data['price'])

# Selecionando os atributos relevantes
X = data[['curb-weight', 'engine-size', 'horsepower', 'width', 'length']]
Y = data['price']

# Configurando K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

# Plotando os dados reais e a linha de regressão
plt.figure(figsize=(15, 8))
plt.scatter(Y, Y, color='blue', label='Dados Reais', alpha=0.5)

# Realizando K-Fold e plotando a linha de regressão para cada fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    # Plotando a linha de regressão
    plt.plot(Y_test, Y_pred, marker='o', linestyle='', label='Previsão (Fold)')

plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.title('Regressão Linear com K-Fold Cross-Validation')
plt.legend()
plt.show()
