#importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


## Carregue o dataset definido para você
sonodb = pd.read_csv(r"Sleep_Efficiency.csv")


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("\n Verificando se existem células vazias ou Nan:")
print(sonodb.isna().sum())
dbsono = sonodb.dropna(subset=["Awakenings","Caffeine consumption","Alcohol consumption","Exercise frequency"])


db_atualizado = pd.get_dummies(dbsono["Smoking status"])
db_atualizando = pd.concat((dbsono, db_atualizado), axis=1)
#Apagando as colunas desatualizadas:
db_atualizando = db_atualizando.drop(["Smoking status"], axis=1)
db_atualizando = db_atualizando.drop(["Yes"], axis=1)
db_atualizado = db_atualizando.rename(columns={"No": "Smoking status"})

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
db_final = db_atualizado.drop(["Gender", "Bedtime", "Wakeup time"], axis=1)
X = db_final.drop(["Sleep efficiency"], axis=1)
y = db_final["Sleep efficiency"].values
db_final_columns = X.columns


lasso = Lasso(alpha = 0.3)

lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(db_final_columns, lasso_coef, color = 'darkgreen')
plt.xticks(rotation=45)
plt.show()


#Salve o dataset atualizado se houver modificações.
db_final.to_csv('db_final.csv')
print(db_final)