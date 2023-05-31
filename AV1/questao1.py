import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


## Carregue o dataset definido para você
star_class = pd.read_csv(r"C:\Users\Aluno\Downloads\BIANCA\IA\AV1\dataset\star_classification.csv")





# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(star_class.isna().sum())#o dataset ja estava limpo



# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
# colunas que achei mais relevante(9/18):
star_class_new = (star_class.drop(["u", "i", "cam_col", "field_ID", "spec_obj_ID", "redshift", "plate", "MJD", "fiber_ID"], axis=1))
#print(star_class_new)
star_class_new = star_class_new.rename(columns={"class":"classe"})
print(star_class_new)

#Print o dataframe final e mostre a distribuição de classes que você deve classificar

print(star_class_new["classe"].value_counts(),'\n','\n')



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário

star_dic = {'classe':['GALAXY',"STAR","QSO"]}# dicionario
star_class_new= pd.DataFrame(star_dic)
#print(star_class_new)
lb = LabelEncoder()
star_class_new['classe'] = lb.fit_transform(star_class_new["classe"])
print(star_class_new)