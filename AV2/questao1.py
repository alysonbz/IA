from src.utils import load_ferrari_dataset
import seaborn as sns
import matplotlib.pyplot as plt

# Lendo o dataset.
ferrari = load_ferrari_dataset()
print(ferrari.head())

# Foi preciso criar um dataset sem a coluna Date, pois ela não entra no cálculo de correlações.
ferrari_numeric = ferrari.select_dtypes(include=[float, int])

# Calculando a matriz de correlação.
correlation_matrix = ferrari_numeric.corr()

# Gráfico heatmap para visualizar a matriz e qual o alvo e atributo relevante.
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Stock Data")
plt.show()