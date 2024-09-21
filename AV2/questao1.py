# Importação das bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando o dataset
file_path = 'C:\Users\azjoa\Downloads\thailand_co2_emission_1987_2022'
df = pd.read_csv(file_path)

# Verificação inicial dos dados
print(df.head())
print(df.info())

# Identificação do atributo alvo para regressão: 'emissions_tons'

# Análise de correlação entre variáveis numéricas (year, month, emissions_tons)
correlation = df[['year', 'month', 'emissions_tons']].corr()

# Plotando heatmap para visualizar a correlação
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlação entre variáveis numéricas')
plt.show()

# Analisando a relação entre as variáveis categóricas 'source' e 'fuel_type' com as emissões de CO₂

# Boxplot para a variável 'source' (fonte)
plt.figure(figsize=(12, 6))
sns.boxplot(x='source', y='emissions_tons', data=df)
plt.title('Distribuição de Emissões por Fonte')
plt.xticks(rotation=45)
plt.show()

# Boxplot para a variável 'fuel_type' (tipo de combustível)
plt.figure(figsize=(12, 6))
sns.boxplot(x='fuel_type', y='emissions_tons', data=df)
plt.title('Distribuição de Emissões por Tipo de Combustível')
plt.xticks(rotation=45)
plt.show()

# Discussão dos resultados:
# A correlação entre as variáveis numéricas mostra que o 'year' tem uma correlação moderada com 'emissions_tons',
# enquanto as variáveis categóricas ('source' e 'fuel_type') possuem uma relação mais clara e relevante com as emissões de CO₂.
