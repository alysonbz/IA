import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


# Carregar o dataset
data = pd.read_csv(r'C:\\Users\\jonna\\IA\\AV2\\dataset\\carprice.csv') 

data.replace('?', pd.NA, inplace=True)

data.dropna(inplace=True)

for col in data.select_dtypes(include=['object']).columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

print(data.dtypes)

corr = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

print(corr['price'].sort_values(ascending=False))
