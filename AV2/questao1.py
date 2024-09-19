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

# Foi preciso criar um dataset sem a coluna Date, pois ela não entra no cálculo de correlações
act_blz_new = act_blz.select_dtypes(include=[float, int])

# Calcular a correlação entre as variáveis
correlation_matrix = act_blz_new.corr()

# Extraindo as correlações com o alvo 'Adj Close'
target_corr = correlation_matrix['Adj Close'].drop('Adj Close')

# Visualizar as correlações via gráfico de barras
plt.figure(figsize=(8,6))
sns.barplot(x=target_corr.index, y=target_corr.values, palette='viridis')
plt.title('Correlação entre atributos e Adj Close')
plt.ylabel('Coeficiente de Correlação')
plt.xticks(rotation=45)
plt.show()

# Mostrar a matriz de correlação completa para análise
correlation_matrix['Adj Close'].sort_values(ascending=False)