import pandas as pd
import matplotlib.pyplot as plt

# Carregar o dataset
data = pd.read_csv(r'C:\Users\eryka\Downloads\archive\Samsung Electronics.csv')

# Remover a coluna "Date"
data = data.drop('Date', axis=1)

# Exibir as primeiras linhas do dataset para entender a estrutura dos dados
print(data.head())

# Verificar a correlação entre os atributos
correlation_matrix = data.corr()
print(correlation_matrix)

# Selecionar o atributo alvo
target_attribute = 'Close'

# Calcular a correlação entre os atributos e o atributo alvo
correlations = correlation_matrix[target_attribute]

# Ordenar as correlações em ordem decrescente
sorted_correlations = correlations.abs().sort_values(ascending=False)

# Plotar um gráfico de barras das correlações
plt.figure(figsize=(10, 6))
sorted_correlations.plot(kind='bar')
plt.xlabel('Atributo')
plt.ylabel('Correlação')
plt.title('Correlação entre os atributos e o atributo alvo (Close)')
plt.show()


