from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

print(volunteer['hits'].head(5))

print(volunteer['hits'].describe())

volunteer['hits']=volunteer['hits'].astype(int)
print(volunteer['hits'])
