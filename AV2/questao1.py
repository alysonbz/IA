#importe as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

## Carregue o dataset definido para você
dados = pd.read_csv('laptopPrice.csv')



# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
dm = pd.DataFrame(dados[['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'Price', 'rating', 'Number of Ratings', 'Number of Reviews']])

'''#conversão de strting para int
dm['brand'].replace(['ASUS', 'Lenovo','acer', 'Avita', 'HP', 'DELL', 'MSI', 'APPLE'],
                        [0,1], inplace=True)'''



# Converter a coluna "brand" de volta para valores string
dados['brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'Price', 'rating', 'Number of Ratings', 'Number of Reviews'].astype(str)

# Exibir o DataFrame com a coluna "brand" convertida para string
print(dados)

'''#atributo mais relevante
X = dm.drop(['Price'], axis=1)
y = dados['Price'].values
dm_columns = X.columns
'''






'''# Instantiate a lasso regression model
lasso = Lasso(alpha=0.1)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(dm_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()'''