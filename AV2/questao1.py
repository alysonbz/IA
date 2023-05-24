#importe as bibliotecas necessárias
import pandas as pd
import matplotlib as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from



## Carregue o dataset definido para você
alunos_notas = pd.read_csv(r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\Student_Marks.csv")


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(alunos_notas.isnull().sum())


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
#print(alunos_notas.shape)
X =  alunos_notas.drop("Marks", axis= 1).values
y = alunos_notas["marks"].values
nomes = alunos_notas.drop("Marks", axis= 1).columns
lasso = asso(alpha= 0.1)
lasso_coef = lasso.fit(X, y).coef_
plt.bar(nomes, lasso_coef)
plt.xticks(rotation = 45)
plt.show

#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(alunos_notas["number_courses"].value_counts()) #quantidade de cursos escolhidos por alunos
print(alunos_notas["time_study"].value_counts()) #tempo de estudo
print(alunos_notas["Marks"].value_counts()) #notas

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
# alunos_notas_ajustado.to_csv(r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\Student_Marks.csv")