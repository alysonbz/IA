import pandas as pd


def load_volunteer_dataset():
    return pd.read_csv('../dataset/opportunities.csv')

def load_wine_dataset():
    wine = pd.read_csv('../dataset/wine.csv')
    return wine

def load_hiking_dataset():
    return pd.read_json('../dataset/hiking.json')

def load_df1_unidade1():
    return pd.read_csv('../dataset/df1_unidade1.csv')


def load_df2_unidade1():
    return pd.read_csv('../dataset/df2_unidade1.csv')
