#importe as bibliotecas necessárias
import pandas as pd # type: ignore

## Carregue o dataset definido para você
sc = pd.read_csv(r"C:\Users\jonna\IA\AV1\dataset\star_classification.csv")
print(sc.info())

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
star_classification_new = sc.dropna()

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
sc.drop(['obj_ID','run_ID','rerun_ID','field_ID','fiber_ID'], axis=1, inplace=True)

#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(sc.info())

# Mostrando a distribuição das classes
distribuicao_classes = sc['class'].value_counts()
print(distribuicao_classes)


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário

print(sc['class'].dtype)

if sc['class'].dtype == 'object':
    sc['class'] = sc['class'].astype('category').cat.codes

print(sc)


sc.to_csv('star_classification_atualizado.csv', index=False)