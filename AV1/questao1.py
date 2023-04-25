#importe as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder



## Carregue o dataset definido para você
star_class = pd.read_csv(r"C:\Users\Aluno\Documents\Thais\IA\AV1\dataset\star_classification.csv")
print(star_class)


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(star_class.isna().sum())#o dataset ja estava limpo



# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
# colunas que achei mais relevante(9/18): obj_id, alfa, delta, g. r, z, run_id, rereun_ID, class)
star_class_new = (star_class.drop(["u", "i", "cam_col", "field_ID", "spec_obj_ID", "redshift", "plate", "MJD", "fiber_ID"], axis=1))
print(star_class_new)

#Print o dataframe final e mostre a distribuição de classes que você deve classificar

print(star_class_new["class"].value_counts(),'\n','\n')



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário




#Salve o dataset atualizado se houver modificações.

#star_class_new.to_csv(r"C:\Users\Aluno\Documents\Thais\IA\AV1\dataset\star_class_new.csv")

