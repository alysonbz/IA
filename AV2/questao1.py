"""
Verifique qual atributo será o alvo para regressão no seu
dataset e faça uma análise de qual atributo é mais relevante para
realizar a regressão do alvo escolhido. Lembre de comprovar via gráfico.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre o resultado encontrado.
"""
from src.utils import load_laptopPrice_dataset
# Carregar o dataset
laptop = load_laptopPrice_dataset()  # ajuste o caminho do arquivo

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


# Verificar dados nulos
print("Dados nulos por coluna:")
print(laptop.isnull().sum())

# Transformando dados categóricos em numéricos com LabelEncoder
label_encoder = LabelEncoder()
categorical_cols = laptop.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    laptop[col] = label_encoder.fit_transform(laptop[col])

# Normalizando o dataset
scaler = MinMaxScaler()
laptop_scaled = pd.DataFrame(scaler.fit_transform(laptop), columns=laptop.columns)

# Verificando o dataset após normalização
print("Dataset após normalização:")
print(laptop_scaled.head())

# Análise de correlação para identificar os atributos mais relevantes para 'Price'
correlation = laptop_scaled.corr()

# Ordenar os atributos pela correlação com o preço
price_correlation = correlation['Price'].sort_values(ascending=False)
print("Correlação dos atributos com o preço:")
print(price_correlation)

# Visualizando os atributos mais correlacionados com 'Price'
plt.figure(figsize=(10, 6))
sns.barplot(x=price_correlation.index, y=price_correlation.values)
plt.title("Correlação dos atributos com o Preço")
plt.xticks(rotation=90)
plt.show()

laptop_scaled.to_csv('dataset/Laptop_Data_Cleaned.csv', index=False)
