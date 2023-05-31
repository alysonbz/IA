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
# Separar atributos e alvo
X = df.drop(['BodyFat'], axis=1)
y = df['BodyFat']

# Ajustar o modelo Lasso
lasso = Lasso(alpha=0.3)
lasso_coef = lasso.fit(X, y).coef_
# Obter os coeficientes e plotar
plt.bar(X.columns, lasso_coef)
plt.xticks(rotation=45)
plt.suptitle("Colunas mais relevantes",fontsize=10,y=0.95)
plt.show()
df.to_csv('df.csv')

