#bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import seaborn as sns

##importação do dataset
dados = pd.read_csv(r"C:\Users\ruanr\Downloads\archive (1)\bodyfat.csv")
print(dados.columns)
print(dados.isna().sum())
# Selecionar apenas as colunas relevantes
df=dados[['Density',
            'BodyFat',
            'Age',
            'Weight',
          'Height']]
print(df)

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()
# Separar atributos e alvo
X = df.drop(['BodyFat'], axis=1)
y = df['BodyFat']


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

# Verificar a correlação entre os atributos
correlation_matrix = df.corr()
print(correlation_matrix)

# Selecionar o atributo alvo
target_attribute = 'BodyFat'

# Calcular a correlação entre os atributos e o atributo alvo
correlations = correlation_matrix[target_attribute]

# Ordenar as correlações em ordem decrescente
sorted_correlations = correlations.abs().sort_values(ascending=False)


df.to_csv('df.csv')

