from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

print(volunteer['hits'].head())

print(volunteer['hits'].dtypes)

volunteer['hits'] = volunteer['hits'].astype('int32')

print(volunteer['hits'].head())









