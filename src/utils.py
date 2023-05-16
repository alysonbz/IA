import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.model_selection import train_test_split



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

def processing_all_features_sales_clean():
    sales_df = load_sales_clean_dataset()
    X = sales_df.drop(["sales", "influencer"], axis=1)
    y = sales_df["sales"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    reg = LinearRegression()
    reg.fit(X_test, y_test)
    predictions = reg.predict(X)
    return X, y, predictions

def process_diabetes():
    from src.utils import load_diabetes_clean_dataset
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    diabetes_df = load_diabetes_clean_dataset()
    X = diabetes_df.drop(['diabetes'], axis=1)
    y = diabetes_df['diabetes'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=6)

    # Fit the model to the training data
    knn.fit(X_train, y_train)

    # Predict the labels of the test data: y_pred
    y_pred = knn.predict(X_test)

    return y_pred , y_test

def log_reg_diabetes():

    diabetes_df = load_diabetes_clean_dataset()
    X = diabetes_df.drop(['diabetes'], axis=1)
    y = diabetes_df['diabetes'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Instantiate the model
    logreg = LogisticRegression()

    # Fit the model
    logreg.fit(X_train, y_train)

    teste = logreg.predict_proba(X_test)
    # Predict probabilities
    y_pred_probs = logreg.predict_proba(X_test)[:, 1]

    y_pred = logreg.predict(X_test)

    return y_pred_probs , y_test, y_pred

