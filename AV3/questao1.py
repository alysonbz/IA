#Importando Bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram

#Carregando o Dataset
csgo_round = pd.read_csv('csgo_round_snapshots.csv')
'''print("\n Dataset: csgo_round")
print(csgo_round)'''

#Verificando se há valores nulos
'''print("\n Verificação da existência de células vazias ou NaN")
print(csgo_round.isna().sum())'''

#Mapeando as categorias desejadas para valores numéricos
    #round_winner
map_round_winner = {
    'T': 0,
    'CT': 1
}
csgo_round['round_winner'] = csgo_round['round_winner'].map(map_round_winner)

'''pd.set_option('display.max_columns', None)
print(csgo_round)'''

#Salvando um novo dataset atualizado
csgo_round_df = csgo_round.drop(['map','bomb_planted'], axis = 1)
'''print(csgo_round_df)'''

X = csgo_round_df.drop(['round_winner'], axis = 1)
y = csgo_round_df["round_winner"].values
csgo_round_df_Lasso = X.columns

#Instantiate a lasso regression model
lasso = Lasso(alpha = 0.3)

#Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
plt.bar(csgo_round_df_Lasso, lasso_coef)
plt.xticks(rotation=45)
plt.show()

csgo_relevantes = csgo_round_df[['ct_health','t_health','ct_armor', 't_armor', 'ct_money', 't_money', 'round_winner']]
'''print(csgo_relevantes)'''

def load_csgo_relevantes():
    X = csgo_relevantes.drop(['round_winner'], axis=1)
    y = csgo_relevantes['round_winner'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

X_train, df, y_train, round_winner = load_csgo_relevantes()
'''
y = csgo_round_df["round_winner"].values
classe = normalize(csgo_relevantes)'''
mergings = linkage(df, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels= round_winner,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

#