'''Verifique qual atributo será o alvo para regressão no seu dataset e faça uma análise
 de qual atributo é mais relevante para realizar a regressão do alvo escolhido.
 Lembre de comprovar via gráfico. Obs: Registrar na seção de resultados a análise realizada
 e discutir sobre o resultado encontrado.'''

# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Carregando o dataset definido para mim
car_price = pd.read_csv('carprice.csv')
pd.set_option('display.max_columns', None)
print("\n Dataset: Previsão do preço de carros")
print(car_price)


# Verificando se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("\nVerificando se existe células vazias ou Nan")
print(car_price.isna().sum())

# Excluindo os caracteres '?' da coluna 'normalized-losses' e 'num-of-doors'
car_price = car_price[~car_price['normalized-losses'].str.contains('\?')]
car_price.reset_index(drop=True, inplace=True)

car_price = car_price[~car_price['num-of-doors'].str.contains('\?')]
car_price.reset_index(drop=True, inplace=True)


# Breve análise exploratória
# Histogramas
colunas_numericas = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
                   'engine-size']

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
axes = axes.ravel()

for i, col in enumerate(colunas_numericas):
    axes[i].hist(car_price[col], bins=20, edgecolor='pink', color='lightpink')
    axes[i].set_title(col)
    axes[i].set_ylabel("Frequência", labelpad=2)
    axes[i].tick_params(axis='both', which='both', labelsize=8, pad=2)

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()


colunas_numericas_2 = ['bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
                     'city-mpg', 'highway-mpg', 'price']

fig, axes = plt.subplots(4, 2, figsize=(12, 8))
axes = axes.ravel()

for i, col in enumerate(colunas_numericas_2):
    axes[i].hist(car_price[col], bins=20, edgecolor='pink', color='lightpink')
    axes[i].set_title(col)
    axes[i].set_ylabel("Frequência", labelpad=2)
    axes[i].tick_params(axis='both', which='both', labelsize=8, pad=2)

plt.tight_layout()
plt.show()


# Gráficos de barras
colunas_categoricas = ['symboling', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style']

fig, axes = plt.subplots(2, 3, figsize=(12, 10))
axes = axes.ravel()

for i, col in enumerate(colunas_categoricas[:6]):
    car_price[col].value_counts().plot(kind='barh', edgecolor='pink', color='lightpink', ax=axes[i])
    axes[i].set_xlabel('Frequência')
    axes[i].set_ylabel(col)
    axes[i].tick_params(axis='y', labelrotation=0)

plt.tight_layout()
plt.show()


colunas_categoricas_2 = ['drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']

fig_2, axes_2 = plt.subplots(2, 3, figsize=(12, 10))
axes_2 = axes_2.ravel()

for i, col in enumerate(colunas_categoricas_2):
    car_price[col].value_counts().plot(kind='barh', edgecolor='pink', color='lightpink', ax=axes_2[i])
    axes_2[i].set_xlabel('Frequência')
    axes_2[i].set_ylabel(col)
    axes_2[i].tick_params(axis='y', labelrotation=0)

plt.tight_layout()
plt.show()


# Matriz de correlação
col_num_mc = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size',
              'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

matriz_corr = car_price[col_num_mc]
correlation_matrix = matriz_corr.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="RdPu")
plt.title("Matriz de Correlação")
plt.show()

# Tradução das colunas para ajudar na minha interpretação dos gráficos
'''
    'symboling': 'classificacao_risco',
    'normalized-losses': 'perdas_normalizadas',
    'make': 'marca',
    'fuel-type': 'tipo_combustivel',
    'aspiration': 'aspiracao',
    'num-of-doors': 'num_portas',
    'body-style': 'estilo_carroceria',
    'drive-wheels': 'tipo_tracao',
    'engine-location': 'localizacao_motor',
    'wheel-base': 'distancia_eixos',
    'length': 'comprimento',
    'width': 'largura',
    'height': 'altura',
    'curb-weight': 'peso',
    'engine-type': 'tipo_motor',
    'num-of-cylinders': 'num_cilindros',
    'engine-size': 'tamanho_motor',
    'fuel-system': 'sistema_combustivel',
    'bore': 'diametro_cilindro',
    'stroke': 'avc',
    'compression-ratio': 'taxa_compressao',
    'horsepower': 'cav_potencia',
    'peak-rpm': 'rotacao_maxima',
    'city-mpg': 'consumo_cidade',
    'highway-mpg': 'consumo_rodovia',
    'price': 'preco'
'''

# Fiz o ".unique()" para cada uma das colunas que precisei transformar em numérica
# print("\nUnique para saber os labels das colunas:")
# print(car_price['fuel-ty_ed'].unique())

# Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
map_make = {
    'audi': 0,
    'bmw': 1,
    'chevrolet': 2,
    'dodge': 3,
    'honda': 4,
    'jaguar': 5,
    'mazda': 6,
    'mercedes-benz': 7,
    'mitsubishi': 8,
    'nissan': 9,
    'peugot': 10,
    'plymouth': 11,
    'porsche': 12,
    'saab': 13,
    'subaru': 14,
    'toyota': 15,
    'volkswagen': 16,
    'volvo': 17
}
car_price['make_ed'] = car_price['make'].map(map_make)
print(car_price[['make', 'make_ed']].head())

map_ft = {
    'gas': 0,
    'diesel': 1
}
car_price['fuel-ty_ed'] = car_price['fuel-type'].map(map_ft)
print(car_price[['fuel-type', 'fuel-ty_ed']].head())

map_asp = {
    'std': 0,
    'turbo': 1
}
car_price['aspirat_ed'] = car_price['aspiration'].map(map_asp)
print(car_price[['aspiration', 'aspirat_ed']].head())

map_nod = {
    'four': 0,
    'two': 1
}
car_price['num-of-d_ed'] = car_price['num-of-doors'].map(map_nod)
print(car_price[['num-of-doors', 'num-of-d_ed']].head())

map_bs = {
    'sedan': 0,
    'hatchback': 1,
    'wagon': 2,
    'hardtop': 3,
    'convertible': 4
}
car_price['body-sty_ed'] = car_price['body-style'].map(map_bs)
print(car_price[['body-style', 'body-sty_ed']].head())

map_dw = {
    'fwd': 0,
    '4wd': 1,
    'rwd': 2
}
car_price['drive-whe_ed'] = car_price['drive-wheels'].map(map_dw)
print(car_price[['drive-wheels', 'drive-whe_ed']].head())

map_el = {
    'front': 1
}
car_price['engine-locat_ed'] = car_price['engine-location'].map(map_el)
print(car_price[['engine-location', 'engine-locat_ed']].head())

map_et = {
    'ohc': 0,
    'l': 1,
    'dohc': 2,
    'ohcv': 3,
    'ohcf': 4
}
car_price['engine-ty_ed'] = car_price['engine-type'].map(map_et)
print(car_price[['engine-type', 'engine-ty_ed']].head())

map_noc = {
    'four': 0,
    'five': 1,
    'six': 2,
    'three': 3,
    'eight': 4
}
car_price['num-of-cyl_ed'] = car_price['num-of-cylinders'].map(map_noc)
print(car_price[['num-of-cylinders', 'num-of-cyl_ed']].head())

map_noc = {
    'mpfi': 0,
    '2bbl': 1,
    'mfi': 2,
    '1bbl': 3,
    'idi': 4,
    'spdi': 5
}
car_price['fuel-syst_ed'] = car_price['fuel-system'].map(map_noc)
print(car_price[['fuel-system', 'fuel-syst_ed']].head())

# Label Encoder
colunas_transformadas = ["compression-ratio", "stroke", "bore", "width", "symboling", "price"]
le = LabelEncoder()
for coluna in colunas_transformadas:
    car_price[coluna] = le.fit_transform(car_price[coluna])
print(car_price.head())


car_price_att = car_price.drop(['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
                                'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system',
                                'horsepower', 'peak-rpm', 'engine-locat_ed', 'height', 'curb-weight'], axis=1)

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
X = car_price_att.drop(['price'], axis=1)  # Características (todas as colunas, exceto price)
y = car_price_att['price'].values  # Variável alvo ('price')

lasso = Lasso(alpha=0.1)  # Especificar o valor do hiperparâmetro de regularização (alpha)
lasso_coef = lasso.fit(X, y).coef_

colunas_relevantes = X.columns

sorted_cols = sorted(zip(colunas_relevantes, lasso_coef), key=lambda x: abs(x[1]), reverse=True)
colunas_relevantes, lasso_coef = zip(*sorted_cols)

plt.figure(figsize=(12, 8))
plt.barh(colunas_relevantes, lasso_coef, color='lightpink')
plt.xlabel('Coeficiente de Lasso')
plt.ylabel('Características')
plt.title('Importância da característica - Regressão de Lasso')
plt.show()


''' 
Após o gráfico, voltei para o dataset atualizado e exclui essas também que não são relevantes:
'horsepower', 'peak-rpm', 'engine-location_encoded', 'height', 'curb-weight' 
'''

# Salve o dataset atualizado se houver modificações.
car_price_atualizado = car_price_att[['num-of-cyl_ed', 'engine-ty_ed', 'drive-whe_ed', 'num-of-d_ed',
                                     'aspirat_ed', 'fuel-ty_ed', 'compression-ratio', 'stroke', 'bore',
                                     'width', 'symboling', 'price', 'make_ed', 'fuel-syst_ed']]

print("\nDataset atualizado somente com as colunas relevantes\n")
print(car_price_atualizado)
car_price_atualizado.to_csv('car_price_atualizado.csv', index=False)
