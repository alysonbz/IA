from src.utils import load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()

print("Dimensões do dataset:", volunteer.shape)

print("Datatypes:\n", volunteer.dtypes)

print(volunteer['locality'].isnull().sum())

volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis=1)

volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

print("Shape:", volunteer_subset.shape)
