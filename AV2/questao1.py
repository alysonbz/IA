"""
### Questão 1

Verifique qual atributo será o alvo para regressão no seu dataset
e faça uma análise de qual atributo é mais relevante para realizar a regressão do alvo escolhido.
Lembre de comprovar via gráfico.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre o resultado encontrado.

"""

import pandas as pd
from src.utils import load_smart_watch_prices_dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
swp = load_smart_watch_prices_dataset()

# Visualizar as primeiras linhas do dataset para entender a estrutura
print(swp.head())
print(swp.dtypes)

# Remover o símbolo de dólar e converter 'Price (USD)' para numérico
swp['Price (USD)'] = swp['Price (USD)'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Substituir valores como 'Unlimited' por NaN ou um valor numérico alto
swp['Battery Life (days)'] = swp['Battery Life (days)'].replace({'hours': '', 'days': '', 'Unlimited': 999}, regex=True)

# Agora, converter a coluna 'Battery Life (days)' para numérico, ignorando erros
swp['Battery Life (days)'] = pd.to_numeric(swp['Battery Life (days)'], errors='coerce')

# Selecionar apenas as colunas numéricas
swp_numeric = swp[['Display Size (inches)', 'Price (USD)', 'Battery Life (days)']]

# Calcular a matriz de correlação
correlation_matrix = swp_numeric.corr()

# Extrair a correlação com o preço ('Price (USD)')
target_correlation = correlation_matrix["Price (USD)"].drop("Price (USD)")

# Plotar um heatmap para visualizar as correlações
plt.figure(figsize=(8, 6))
sns.heatmap(target_correlation.to_frame(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlação dos Atributos com o Preço (Price USD)")
plt.show()

# Ordenar as correlações para encontrar o atributo mais relevante
print("Correlação com o Preço (USD):\n", target_correlation.sort_values(ascending=False))