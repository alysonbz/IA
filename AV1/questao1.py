#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
bt = pd.read_csv('data_brain_tumor.csv')
print('\n DataSet: Tumor Cerebral')
print(bt)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print('\n Verificando a existência de NA')
print(bt.isna().sum())
print('\n Nenhum NA encontrado')

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
bt_colunas_relevantes = bt
print('\n selecionei a a coluna "y" que é a classificadora, e todas as demais para uma classificação mais precisa')


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(bt_colunas_relevantes)
print('\n Irei utilizar a coluna "y", porém, antes, vou ter que transformar a coluna em valores inteiros.')


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
print('\n Transformando "y" em inteiro')
bt_modifica = pd.get_dummies(bt_colunas_relevantes["y"])
print(bt_modifica)
bt_modificando = pd.concat((bt_modifica, bt_colunas_relevantes), axis=1)
bt_modificando = bt_modificando.drop(["y"], axis=1)
bt_modificando = bt_modificando.drop(["tumor"], axis=1)
bt_modificado = bt_modificando.rename(columns={"Normal": "classe"})
print('\n Print do dataset modificado')
print(bt_modificado)


print('DISTRIBUIÇÃO DE CLASSE, QUANTIDADE DE 1 E QUANTIDADE DE 0:')
print(bt_modificado['classe'].value_counts())


#Salve o dataset atualizado se houver modificações.
bt_modificado.to_csv('bt_novo.csv')