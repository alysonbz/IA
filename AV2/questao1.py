#bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder

#importação do dataset
dados = pd.read_csv('car_price_prediction.csv')







# Selecionar apenas as colunas relevantes
numeric_columns = ['Price', 'Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']
df = dados[numeric_columns]

# Remover linhas com valores ausentes
df = df.dropna()


correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Converter valores não numéricos para numéricos
label_encoder = LabelEncoder()
df['Levy'] = label_encoder.fit_transform(df['Levy'])

# Tratar coluna 'Engine volume'
df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '')  # Remover ' Turbo'
df['Engine volume'] = df['Engine volume'].str.replace(' L', '')  # Remover ' L'
Ev = df['Engine volume'].astype(float)  # Converter para float

# Tratar coluna 'Mileage'
df['Mileage'] = df['Mileage'].str.replace(' km', '')  # Remover ' km'
df['Mileage'] = df['Mileage'].str.replace(',', '')  # Remover vírgulas
df['Mileage'] = df['Mileage'].astype(float)  # Converter para float

# Separar atributos e alvo
X = df.drop(['Price'], axis=1)
y = df['Price']


# Ajustar o modelo Lasso
lasso = Lasso(alpha=0.3)
lasso.fit(X, y)

# Obter os coeficientes e plotar
lasso_coef = lasso.coef_
plt.bar(X.columns, lasso_coef)
plt.xticks(rotation=45)
plt.xlabel('Atributos')
plt.ylabel('Coeficientes')
plt.title('Coeficientes do modelo Lasso')
plt.show()


Ev.to_csv('dados.csv', index=False)
df.to_csv('dados1.csv', index=False)