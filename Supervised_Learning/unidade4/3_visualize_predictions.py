from src.utils import processing_sales_clean
import matplotlib.pyplot as plt

X, y, predictions = processing_sales_clean()

# Criar gráfico de dispersão.
plt.scatter(X, y, color="blue")

# Criar gráfico de linha.
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Exibir o gráfico.
plt.show()
