import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_lenovo_share_prices  # Importar a função de utils.py

# 1. Carregar o dataset usando a função do arquivo utils.py
lenovo_df = load_lenovo_share_prices()

# 2. Matriz de correlação entre os atributos
correlation_matrix = lenovo_df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()

# Exibindo a matriz de correlação com um heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlação dos Atributos - Preço de Fechamento")
plt.show()

# 3. Gráfico de dispersão entre 'High' e 'Close'
plt.figure(figsize=(8, 6))
plt.scatter(lenovo_df['High'], lenovo_df['Close'], alpha=0.5)
plt.title("Relação entre Preço de Fechamento e Preço Mais Alto do Dia")
plt.xlabel("Preço Mais Alto (High)")
plt.ylabel("Preço de Fechamento (Close)")
plt.grid(True)
plt.show()

# 4. Exibição dos resultados
print("Matriz de correlação:")
print(correlation_matrix)
