import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv(r"C:\Users\Aluno\Documents\Thais\IA\AV2\averrage sceen time by country.csv")

print(dataset.isna().sum())#o dataset ja estava limpo
#dataset_new = (dataset.drop(["Flag"], axis=1))
#print(dataset_new["Country"].value_counts(),'\n','\n')
#print(dataset)
data_dic = {'Country':['Argentina','Australia', 'Austria', 'Belgium', 'Brazil', 'Canada', 'Chile', 'China', 'Colombia', 'Czechia', 'Denmark', 'Egypt', 'France', 'Germany', 'Greece', 'Hong Kong', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Malaysia', 'Mexico', 'Netherlands', 'New Zealand', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Russia', 'Saudi Arabia', 'Singapore', 'South Africa', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Turkey', 'U.A.E', 'UK', 'USA', 'Vietnam', 'Worldwide']}# dicionario
dataset_new = pd.DataFrame(data_dic)
lb = LabelEncoder()
dataset_new['Country'] = lb.fit_transform(dataset_new["Country"])
print(dataset_new)
