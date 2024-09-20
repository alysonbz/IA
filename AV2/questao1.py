"""
- Questão 1
    Verifique qual atributo será o alvo para regressão no seu dataset
    e faça uma análise de qual atributo é mais relevante para realizar a regressão do alvo escolhido.

    Lembre de comprovar via gráfico.
    Obs: Registrar na seção de resultados a análise realizada e discutir sobre o resultado encontrado.
"""
from src.utils import load_activision_blizzard_dataset
import seaborn as sns
import matplotlib.pyplot as plt

# Lendo o dataset
act_blz = load_activision_blizzard_dataset()
print(act_blz.head())
print(act_blz.dtypes)

# Foi preciso criar um dataset sem a coluna Date, pois ela não entra no cálculo de correlações
act_blz_new = act_blz.select_dtypes(include=[float, int])

# Calcular a matriz de correlação
correlation_matrix = act_blz_new.corr()

# Extrair a correlação com o alvo (Adj Close)
target_correlation = correlation_matrix["Adj Close"].drop("Adj Close")

# Montando um gráfico heatmap para visualizar as correlações
plt.figure(figsize=(8, 6))
sns.heatmap(target_correlation.to_frame(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlação dos Atributos com o Atributo Alvo (Adj Close)")
plt.show()

# Mostrar a correlação com o alvo 'Adj Close'
print(target_correlation)
