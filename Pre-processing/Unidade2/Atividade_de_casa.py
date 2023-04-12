import pandas as pd

lista=[]
with open('iris_data.csv', 'r') as f:
    for linha in f.readlines():
        a=linha.replace('\n','').split(',')
        lista.append(a)