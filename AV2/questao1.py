
import pandas as pd
import matplotlib.pyplot as plt

#carregar dataset
file_path = 'C:\\Users\\Neto\\Downloads\\IA\\AV2\\Dataset\\Sample - Superstore.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

print(df.head())

#seleção das colunas
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

#calculo
correlation_matrix = df[numerical_columns].corr()

# Exibindo a correlação
correlation_with_sales = correlation_matrix["Sales"].sort_values(ascending=False)
print(correlation_with_sales)

# Gerando o gráfico
plt.figure(figsize=(8, 6))
plt.scatter(df['Profit'], df['Sales'], alpha=0.5)
plt.title('Relação entre Vendas (Sales) e Lucro (Profit)')
plt.xlabel('Profit')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
