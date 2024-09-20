import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Função para carregar e limpar os dados
def carregar_dados(caminho):
    data = pd.read_csv(caminho)
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)
    
    # Convertendo colunas relevantes para o tipo numérico
    colunas_numericas = ['horsepower', 'curb-weight', 'engine-size', 'width', 'length', 'price']
    for coluna in colunas_numericas:
        data[coluna] = pd.to_numeric(data[coluna])
    
    return data

# Função para criar gráficos de dispersão
def criar_graficos(X, Y):
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    ax[0, 0].scatter(X['curb-weight'], Y)
    ax[0, 0].set_xlabel('curb-weight')
    ax[0, 0].set_ylabel('price')

    ax[0, 1].scatter(X['engine-size'], Y)
    ax[0, 1].set_xlabel('engine-size')
    ax[0, 1].set_ylabel('price')

    ax[0, 2].scatter(X['horsepower'], Y)
    ax[0, 2].set_xlabel('horsepower')
    ax[0, 2].set_ylabel('price')

    ax[1, 0].scatter(X['width'], Y)
    ax[1, 0].set_xlabel('width')
    ax[1, 0].set_ylabel('price')

    ax[1, 1].scatter(X['length'], Y)
    ax[1, 1].set_xlabel('length')
    ax[1, 1].set_ylabel('price')

    # Remove a posição do gráfico que não está sendo usada
    ax[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Função para treinar e avaliar o modelo com Ridge ou Lasso
def treinar_e_avaliar_modelo(X, Y, modelo, lambdas):
    scores = []
    for lamb in lambdas:
        modelo.alpha = lamb
        modelo.fit(X, Y)
        score = modelo.score(X, Y)
        scores.append(score)

    return scores

# Função para plotar os scores
def plotar_scores(lambdas, scores, nome_modelo):
    plt.scatter(lambdas, scores)
    plt.xlabel('Lambda')
    plt.ylabel(f'{nome_modelo} Score')
    plt.xscale('log')
    plt.xlim([min(lambdas), max(lambdas)])
    plt.ylim([min(scores) - 0.01, max(scores) + 0.01])
    plt.show()

data = carregar_dados(r'C:\\Users\\jonna\\IA\\AV2\\dataset\\carprice.csv')

# Selecionar atributos relevantes
X = data[['curb-weight', 'engine-size', 'horsepower', 'width', 'length']]
Y = data['price']

# Criar gráficos de dispersão
criar_graficos(X, Y)

# Testar modelos Ridge e Lasso
lambdas_ridge = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5]
lambdas_lasso = [0.0001, 0.001, 0.01, 0.1, 0.5, 1]

# Ridge
modelo_ridge = Ridge()
scores_ridge = treinar_e_avaliar_modelo(X, Y, modelo_ridge, lambdas_ridge)
plotar_scores(lambdas_ridge, scores_ridge, "Ridge")

# Lasso
modelo_lasso = Lasso()
scores_lasso = treinar_e_avaliar_modelo(X, Y, modelo_lasso, lambdas_lasso)
plotar_scores(lambdas_lasso, scores_lasso, "Lasso")
