import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv("LG Electronics (2023-24).csv")

# Excluir a coluna 'Date' antes de calcular a matriz de correlação
df_numeric = df.drop(columns=['Date'])

# Calcular a matriz de correlação
correlation_matrix = df_numeric.corr()

# Extrair a correlação com o alvo (Adj Close)
target_correlation = correlation_matrix["Adj Close"].drop("Adj Close")

# Plotar um heatmap para visualizar as correlações
plt.figure(figsize=(8, 6))
sns.heatmap(target_correlation.to_frame(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlação dos Atributos com o Preço Ajustado (Adj Close)")
plt.show()

# Ordenar as correlações para encontrar o atributo mais relevante
print("Correlação com o Preço Ajustado (Adj Close):\n", target_correlation.sort_values(ascending=False))
