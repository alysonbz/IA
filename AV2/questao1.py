#1 importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

#2  Carregue o dataset definido para você
smartwatch = pd.read_csv('Smartwatchprices.csv')
print("\n Dataset: Smart watch prices")
print(smartwatch)

#3 Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe
print("\n Verificação da existência de células vazias ou NaN")
print(smartwatch.isna().sum())

smartwatch_sem_na = smartwatch.dropna()
print(smartwatch_sem_na.isna().sum())

#5 observe se a coluna de classes precisa ser renomeada para atributos numéricos, realizar a conversão, se necessário
# Renomear a coluna "Brand" (Todos os atributos)
smartwatch_sem_na["Marcas"] = pd.factorize(smartwatch_sem_na["Brand"])[0]

# Renomear a coluna "Operating System" (Todos os atributos)
smartwatch_sem_na["Sistema Operacional"] = pd.factorize(smartwatch_sem_na["Operating System"])[0]

# Mapear as categorias desejadas para valores numéricos
'''display_mapping = {"AMOLED": 1, "LCD": 2}
smartwatch_sem_na["Display"] = smartwatch_sem_na["Display Type"].map(display_mapping).fillna(3)'''
#(Com Todos os Atributos)
smartwatch_sem_na["Display"] = pd.factorize(smartwatch_sem_na["Display Type"])[0]

# Mapear as categorias desejadas para valores numéricos
smartwatch_sem_na["Conectividade"] = pd.factorize(smartwatch_sem_na["Connectivity"])[0]

# Mapear as categorias desejadas para valores numéricos
smartwatch_sem_na["Resolução"] = pd.factorize(smartwatch_sem_na["Resolution"])[0]

# Mapear as categorias desejadas para valores numéricos
smartwatch_sem_na["Modelo"] = pd.factorize(smartwatch_sem_na["Model"])[0]

# Mapear as categorias desejadas para valores numéricos
heart_rate_mapping = {"Yes": 1, "No": 0}  # Adicione os valores desejados de acordo com o dataset
smartwatch_sem_na["Monitor"] = smartwatch_sem_na["Heart Rate Monitor"].map(heart_rate_mapping).fillna(2)

# Mapear as categorias desejadas para valores numéricos
gps_mapping = {"Yes": 1, "No": 0}  # Adicione os valores desejados de acordo com o dataset
smartwatch_sem_na["GPS"] = smartwatch_sem_na["GPS"].map(gps_mapping).fillna(2)

# Mapear as categorias desejadas para valores numéricos
nfc_mapping = {"Yes": 1, "No": 0}  # Adicione os valores desejados de acordo com o dataset
smartwatch_sem_na["NFC"] = smartwatch_sem_na["NFC"].map(nfc_mapping).fillna(2)

# Renomear a coluna para 'Resistencia agua'
smartwatch_sem_na = smartwatch_sem_na.rename(columns={'Water Resistance (meters)': 'Resistencia agua'})
# Remover linhas com valores não numéricos na coluna 'Resistencia agua'
smartwatch_sem_na = smartwatch_sem_na[pd.to_numeric(smartwatch_sem_na['Resistencia agua'], errors='coerce').notnull()]


# Remover linhas com valores não numéricos na coluna 'Display Size (inches)'
smartwatch_sem_na = smartwatch_sem_na[pd.to_numeric(smartwatch_sem_na['Display Size (inches)'], errors='coerce').notnull()]
# Renomear a coluna 'Display Size (inches)' para 'Display Tamanho'
smartwatch_sem_na = smartwatch_sem_na.rename(columns={'Display Size (inches)': 'Display Tamanho'})

# Remover o cifrão ($) e a vírgula da coluna "Price"
smartwatch_sem_na["Preço"] = smartwatch_sem_na["Price (USD)"].str.replace("$", "").str.replace(",", "")

# Exibir o resultado
print(smartwatch_sem_na["Preço"])

#4 Verifique quais colunas são as mais relevantes e crie um novo dataframe.
smp_relevantes = smartwatch_sem_na[["Marcas", "Sistema Operacional", "Conectividade", "Display", "Display Tamanho", "Resistencia agua","NFC", "GPS", "Preço"]]

print("\n Dataset atualizado:")
print(smp_relevantes)

# Separar as colunas de recursos (X) e a coluna de destino (y)
X = smp_relevantes.drop(["Preço"], axis=1)
y = smp_relevantes["Preço"].astype(float)

# Instanciar o modelo Lasso e ajustá-lo aos dados
lasso = Lasso(alpha=0.3)
lasso_coef = lasso.fit(X, y).coef_

# Plotar os coeficientes do Lasso
plt.bar(X.columns, lasso_coef)
plt.xticks(rotation=45)
plt.suptitle("Colunas mais relevantes",fontsize=10,y=0.95)
plt.show()

#Salve o dataset atualizado se houver modificações.
smp_relevantes.to_csv("sm_prices.csv")