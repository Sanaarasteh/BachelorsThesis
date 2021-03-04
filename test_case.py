import pandas as pd

from source.utils import series_to_sequence

dataset = pd.read_csv('datasets/AppliancesEnergy/energydata_complete.csv')

print(dataset.shape)

new_samples, new_labels = series_to_sequence(dataset, 'Appliances', 10, 1)

print(new_samples.shape)
print(new_labels.shape)
