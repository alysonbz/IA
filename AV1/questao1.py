# Importe as bibliotecas necessárias.
import pandas as pd

# Carregue o dataset definido para você.
dados_avc = pd.read_csv('healthcare-dataset-stroke-data.csv')
print("\nBase de dados inicial:")
print(dados_avc)

print("Número de linhas: ", dados_avc.shape[0],"\nNúmero de colunas: ", dados_avc.shape[1])

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("\nVerificando se existem células vazias ou Nan:")
print(dados_avc.isna().sum())

dados_avc_sem_na = dados_avc.dropna()

print("\nBase de dados sem valores vazios:")
print(dados_avc_sem_na.head())
print("Linhas: ", dados_avc_sem_na.shape[0],"\nColunas: ", dados_avc_sem_na.shape[1])

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
dados_avc_sem_col_irrel = dados_avc_sem_na.drop(["id", "ever_married", "work_type", "Residence_type"], axis = 1)

print("\nBase de dados sem as colunas irrelevantes:")
print(dados_avc_sem_col_irrel.head())
print("Linhas: ", dados_avc_sem_col_irrel.shape[0],"\nColunas: ", dados_avc_sem_col_irrel.shape[1])

# Print o dataframe final e mostre a distribuição de classes que você deve classificar.
print("\nIrei usar as colunas 'gender' e 'smoking_status', mas para isso precisarei modificá-las:")
print(dados_avc_sem_col_irrel.head())
print("Linhas: ", dados_avc_sem_col_irrel.shape[0],"\nColunas: ", dados_avc_sem_col_irrel.shape[1])

print("\nDistribuição de classes:")
print(dados_avc_sem_col_irrel['stroke'].value_counts())


# Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário.
dados_avc_sem_col_irrel["gender"] = dados_avc_sem_col_irrel["gender"].map({"Male": 0, "Female": 1, "Other": 2})
dados_avc_sem_col_irrel["smoking_status"] = dados_avc_sem_col_irrel["smoking_status"].map({"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3})

print("\nBase de dados com as colunas 'gender' e 'smoking_status' transformadas para numéricas:")
print(dados_avc_sem_col_irrel.head())

# Salve o dataset atualizado se houver modificações.
avc_ajustado = dados_avc_sem_col_irrel
avc_ajustado.to_csv = ('avc_ajustado.csv')

print("\nBase de dados após todo o pré processamento:")
print(avc_ajustado.head())