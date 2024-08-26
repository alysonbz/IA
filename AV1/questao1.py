#importe as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Carregue o dataset definido para você

waterQuality = pd.read_csv('dataset/waterQuality1.csv')
print("Water Quality\n",waterQuality.info())
#Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
"""não existian celulas vazias ou nan, todavia, em duas colunas existiam strings que não permitiam converter 
as colunas para tipos numericos. Essas strings não foram sinalizadas na função info, então 
tranformei as strings em nan e dropei"""

waterQuality.replace('@NUM!', np.nan, inplace=True)
waterQuality = waterQuality.apply(pd.to_numeric, errors='coerce')
print(waterQuality.isna().sum())

waterQuality = waterQuality.dropna(how='any')

CleanWaterQuality = waterQuality


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.

"""todas a colunas do dataset são relevntes, pois são um conjunto de substancias que 
que tornam a agua segura de acordo com sua concentração e/ou presença"""

#Print o dataframe final e mostre a distribuição de classes que você deve classificar

print("dataframe final\n",CleanWaterQuality.info(), "istribuição de classes\n", CleanWaterQuality["is_safe"].value_counts())
# Contar a frequência de cada categoria

unique, counts = np.unique(CleanWaterQuality["is_safe"], return_counts=True)
cls = ['is_safe', 'not safe']
# Criar o gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(cls, counts, color='skyblue')

# Adicionar títulos e rótulos
plt.title('Contagem de Classificações')
plt.xlabel('Classificação')
plt.ylabel('Número de Ocorrências')
plt.show()# Mostrar o gráfico

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário

CleanWaterQuality["is_safe"] = CleanWaterQuality["is_safe"].astype(int)
CleanWaterQuality["ammonia"] = CleanWaterQuality["ammonia"].astype(float)


#Salve o dataset atualizado se houver modificações.

CleanWaterQuality.to_csv('dataset/CleanWaterQuality1.csv', index=False)