from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

churn_df = load_churn_dataset()
print(churn_df)
print(churn_df.describe)
print(churn_df.info)
print(churn_df["churn"].value_counts())


