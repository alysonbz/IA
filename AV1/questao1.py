# importe as bibliotecas necessárias
import pandas as pd

path = "dataset/flavors_of_cacao.csv"
df = pd.read_csv(path)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("Células vazias ou NaN antes da remoção:")
print(df.isnull().sum())
df_cleaned = df.dropna()
print("\nCélulas vazias ou NaN após a remoção:")
print(df_cleaned.isnull().sum())
print(df_cleaned.head())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print("Colunas do dataframe:")
print(df_cleaned.columns)

# Selecionar colunas relevantes
columns_relevant = [
    'Specific Bean Origin\r\nor Bar Name',
    'Cocoa\r\nPercent',
    'Company\r\nLocation',
    'Rating'
]
df_relevant = df_cleaned[columns_relevant]

# Print do dataframe final
print("Dataframe final:")
print(df_relevant.head())

# Mostrar a distribuição de classes (Rating)
print("\nDistribuição de classes:")
print(df_relevant['Rating'].value_counts())
# observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
# Converter a coluna 'Cocoa_Percent' para float
df_relevant.loc[:, 'Cocoa\r\nPercent'] = df_relevant['Cocoa\r\nPercent'].str.replace('%', '').astype(float)

# Salve o dataset atualizado se houver modificações.
df_relevant.to_csv("dataset/flavors_of_cacao_ajustado.csv", index=False)
print("\nDataset salvo como 'flavors_of_cacao_ajustado'")
