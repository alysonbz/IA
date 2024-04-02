from src.utils import load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()

print(volunteer['hits'].head())

print(volunteer['hits'].dtype())


volunteer['hits'] = volunteer['hits'].astype('int32')

print(volunteer['hits'].head())
