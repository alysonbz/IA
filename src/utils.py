import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


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

def load_df1_unidade2():
    return pd.read_csv('../dataset/df1_unidade2.csv')

def load_df2_unidade2():
    return pd.read_csv('../dataset/df2_unidade2.csv')

def load_churn_dataset():
    df = pd.read_csv('../dataset/churn_train.csv')
    le = LabelEncoder()
    df['churn'] = le.fit_transform(df['churn'])
    return df

def load_iris_dataset():
    df = pd.read_csv('../dataset/iris.csv')
    return df

def load_sales_clean_dataset():
    df = pd.read_csv('../dataset/sales_clean.csv')
    return df

def load_diabetes_clean_dataset():
    df = pd.read_csv('../dataset/diabetes_clean.csv')
    return df

def processing_sales_clean():
    sales_df = load_sales_clean_dataset()
    y = sales_df["sales"].values
    X = sales_df["radio"].values.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X, y)
    predictions = reg.predict(X)
    return X,y,predictions
