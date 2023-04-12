from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

churn_df = load_churn_dataset()

print(churn_df)

#print(churn_df['churn'])
#print(churn_df['account_length'])

#print(churn_df['total_day_minutes'])

#print(churn_df['churn'].value_counts())

