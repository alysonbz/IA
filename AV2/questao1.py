#importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

## Carregue o dataset definido para você
emissao = pd.read_csv('thailand_co2_emission_1987_2022.csv')
print('\n DataSet: Emissão de CO2 da Tailândia ')
print(emissao)


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(emissao.isna().sum())

#Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário

'''Mapear os valores da coluna source'''
'''TRANSFORMAÇÃO'''
mapping = {
    "industry": 1,
    "transport": 2
}
emissao["source"] = emissao["source"].map(mapping).fillna(3)  # Preencher valores não mapeados com 3

mapping = {
    "oil": 1,
    "natural_gas": 2
}
emissao["fuel_type"] = emissao["fuel_type"].map(mapping).fillna(3)  # Preencher valores não mapeados com 3

# Salvar o resultado em um novo arquivo CSV
emissao_novo= emissao
print(emissao_novo)


#Verifique quais colunas são as mais relevantes e crie um novo dataframe.
'''Verificação da coluna mais relevantes usando o Lasso'''
X = emissao_novo.drop(['emissions_tons'], axis = 1)
y = emissao_novo["emissions_tons"].values
emissao_novo_columns = X.columns
# Instantiate a lasso regression model
lasso = Lasso(alpha = 0.3)
# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(emissao_novo_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
'''O mais importante é tipo de combustivel usado'''
print('O mais importante é o tipo de combustivel usado, porem como ele possui poucos valores (1,2,3) vamos utilizar a tabela (year)')


#Salve o dataset atualizado se houver modificações.
emissao_new = emissao_novo
emissao_new.to_csv('emissao_new.csv')