#importe as bibliotecas necessárias
import pandas as pd

# Carregue o dataset definido para você
read_cancer = pd.read_csv("Cancer_Data.csv")


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(read_cancer.isnull().sum())
print(read_cancer.isna().sum())
X = read_cancer.drop("Unnamed: 32", axis=1)
print(X.isnull().sum())


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.

cancer = X[['id', 'diagnosis', 'radius_mean', 'area_mean', 'fractal_dimension_worst']]
cancer = pd.DataFrame(X[['id', 'diagnosis', 'radius_mean', 'area_mean', 'fractal_dimension_worst']])
print(cancer)

#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(cancer['id'].value_counts(),'\n','\n')
print(cancer['diagnosis'].value_counts(),'\n','\n')
print(cancer['radius_mean'].value_counts(),'\n','\n')
print(cancer['area_mean'].value_counts(),'\n','\n')
print(cancer['fractal_dimension_worst'].value_counts(),'\n','\n')


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
cancer['diagnosis'].replace(['B', 'M'],
                        [0, 1], inplace=True)

#Salve o dataset atualizado se houver modificações.')
print(cancer)


cancer.to_csv("dados_preprocessados.csv", index=False)

