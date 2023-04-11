from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

churn_df = load_churn_dataset()

#função para contar elementos
print(churn_df['churn'].value_counts())